import AVFoundation
import CoreMedia
import Foundation
import Speech

@available(macOS 26.0, *)
final class AudioFeeder: @unchecked Sendable {
    private let queue = DispatchQueue(label: "LiveTranslator.AudioFeeder")
    private let lock = NSLock()
    private var source: AVAudioFormat?
    private var target: AVAudioFormat?
    private var converter: AVAudioConverter?
    private var continuation: AsyncStream<AnalyzerInput>.Continuation?
    private var frames: Int64 = 0

    var seconds: Double {
        lock.lock()
        defer { lock.unlock() }
        return Double(frames) / 16000
    }

    func configure(target: AVAudioFormat, continuation: AsyncStream<AnalyzerInput>.Continuation) {
        queue.sync {
            let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
            self.source = source
            self.target = target
            self.converter = source == target ? nil : AVAudioConverter(from: source, to: target)
            self.continuation = continuation
        }
    }

    func append(_ data: Data) {
        queue.async { [self] in feed(data) }
    }

    func finish() {
        queue.sync {
            continuation?.finish()
            continuation = nil
        }
    }

    private func feed(_ data: Data) {
        guard let source, let target, let continuation else { return }
        let count = AVAudioFrameCount(data.count / MemoryLayout<Float>.size)
        guard count > 0, let buffer = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: count) else { return }
        buffer.frameLength = count
        data.withUnsafeBytes { raw in
            buffer.floatChannelData![0].update(from: raw.bindMemory(to: Float.self).baseAddress!, count: Int(count))
        }
        var input = buffer
        if let converter {
            let capacity = AVAudioFrameCount(Double(count) * target.sampleRate / source.sampleRate) + 32
            guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var consumed = false
            var error: NSError?
            converter.convert(to: output, error: &error) { _, status in
                if consumed {
                    status.pointee = .noDataNow
                    return nil
                }
                consumed = true
                status.pointee = .haveData
                return buffer
            }
            guard error == nil, output.frameLength > 0 else { return }
            input = output
        }
        continuation.yield(AnalyzerInput(buffer: input))
        lock.lock()
        frames += Int64(count)
        lock.unlock()
    }
}

@available(macOS 26.0, *)
@MainActor
final class SpeechEngine {
    struct Item {
        let text: String
        let start: Double
        let end: Double
        let queued: Date
        let key: String
    }

    struct Lane {
        var chunk = 0
        var open = false
        var chunkStart = -1.0
        var committed = 0
        var parts: [String] = []
        var terminated = 0
        var rangeEnd = 0.0
        var seen: [String: (Date, Double)] = [:]
        var queue: [Item] = []
    }

    static let languages = ["ja": "ja-JP", "en": "en-US"]
    static let settleSeconds = 0.8
    static let lonelySettleSeconds = 1.6
    static let englishWait = 1.0
    static let englishMaxWait = 4.0
    static let japaneseCharsPerSecond = 2.0
    static let englishWordRatio = 0.6
    static let dictionary: Set<String> = {
        guard let words = try? String(contentsOfFile: "/usr/share/dict/words", encoding: .utf8) else { return [] }
        return Set(words.split(separator: "\n").map { $0.lowercased() })
    }()
    static let fillers: Set<String> = ["オー", "おー", "えっと", "えーと", "ええと", "えー", "え", "あー", "あ", "あの", "あのー", "うーん", "うん", "ん", "んー", "まあ", "um", "uh", "uhm", "umm", "hmm", "hm", "mm", "mhm", "er", "ah", "oh", "huh"]

    var onPartial: (String) -> Void = { _ in }
    var onFinal: (Int, String, String) -> Void = { _, _, _ in }
    var onRevise: (Int, String) -> Void = { _, _ in }
    var onError: (String) -> Void = { _ in }
    // 術語表裡的正確寫法，給 SpeechAnalyzer 當 contextual strings，讓辨識在第一次就偏向這些詞；prepare() 之前設定
    var contextualStrings: [String] = []

    nonisolated let feeder = AudioFeeder()
    private var analyzer: SpeechAnalyzer?
    private var resultTasks: [Task<Void, Never>] = []
    private var ticker: Task<Void, Never>?
    private var lanes: [String: Lane] = ["ja": Lane(), "en": Lane()]
    private var japaneseBase = 0
    private var japaneseHistory: [(Double, Int)] = [(0, 0)]
    private var japaneseSpans: [(Double, Double)] = []
    private var lastPartial = ""
    private var nextId = 1
    private var emitted: [String: (Int, String)] = [:]
    private var finishing = false
    private let debug = ProcessInfo.processInfo.environment["LIVE_TRANSLATOR_DEBUG"] == "1"

    func prepare() async throws {
        var modules: [SpeechTranscriber] = []
        var named: [(String, SpeechTranscriber)] = []
        for (language, identifier) in Self.languages {
            let transcriber = SpeechTranscriber(
                locale: Locale(identifier: identifier),
                transcriptionOptions: [],
                reportingOptions: [.volatileResults, .fastResults],
                attributeOptions: [.audioTimeRange]
            )
            modules.append(transcriber)
            named.append((language, transcriber))
        }
        if let request = try await AssetInventory.assetInstallationRequest(supporting: modules) {
            try await request.downloadAndInstall()
        }
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: modules) else {
            throw NSError(domain: "SpeechEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "沒有可用的音訊格式"])
        }
        for (language, transcriber) in named {
            resultTasks.append(Task { [weak self] in
                do {
                    for try await result in transcriber.results {
                        var timed: [(Character, Double, Double)] = []
                        if result.isFinal {
                            for run in result.text.runs {
                                let start = run.audioTimeRange?.start.seconds ?? timed.last?.2 ?? result.range.start.seconds
                                let end = run.audioTimeRange?.end.seconds ?? start
                                for character in result.text[run.range].characters { timed.append((character, start, end)) }
                            }
                        }
                        self?.receive(text: String(result.text.characters), timed: timed, start: result.range.start.seconds, end: result.range.end.seconds, isFinal: result.isFinal, language: language)
                    }
                } catch {
                    self?.onError(error.localizedDescription)
                }
            })
        }
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        feeder.configure(target: format, continuation: continuation)
        let analyzer = SpeechAnalyzer(modules: modules, options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .processLifetime))
        self.analyzer = analyzer
        if !contextualStrings.isEmpty {
            let context = AnalysisContext()
            context.contextualStrings[.general] = contextualStrings
            try await analyzer.setContext(context)
        }
        try await analyzer.prepareToAnalyze(in: format)
        try await analyzer.start(inputSequence: stream)
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
                self?.update()
            }
        }
    }

    nonisolated func append(_ data: Data) {
        feeder.append(data)
    }

    func flush() async {
        try? await analyzer?.finalize(through: nil)
        update()
    }

    func finish() async {
        feeder.finish()
        try? await analyzer?.finalizeAndFinishThroughEndOfInput()
        for task in resultTasks { await task.value }
        ticker?.cancel()
        finishing = true
        update()
    }

    func cancel() {
        feeder.finish()
        ticker?.cancel()
        for task in resultTasks { task.cancel() }
        let analyzer = self.analyzer
        Task { await analyzer?.cancelAndFinishNow() }
    }

    private func receive(text raw: String, timed: [(Character, Double, Double)], start: Double, end: Double, isFinal: Bool, language: String) {
        var lane = lanes[language]!
        if !lane.open {
            lane.open = true
            lane.chunk += 1
            lane.chunkStart = start
            lane.committed = 0
            lane.seen = [:]
        }
        let stamp = isFinal ? end : max(end, feeder.seconds - 0.4)
        let (sentences, tail) = Self.split(Self.clean(raw, language: language), language: language)
        lane.parts = tail.isEmpty ? sentences : sentences + [tail]
        lane.terminated = isFinal ? lane.parts.count : sentences.count
        lane.rangeEnd = stamp
        if language == "ja" {
            let count = japaneseBase + Self.japaneseCount(raw)
            japaneseHistory.append((stamp, count))
            if isFinal { japaneseBase = count }
        }
        if isFinal {
            for index in 0..<min(lane.committed, lane.parts.count) {
                let key = "\(language)|\(lane.chunk)|\(index)"
                if let (id, text) = emitted[key], Self.bare(text) != Self.bare(lane.parts[index]), !lane.parts[index].isEmpty {
                    emitted[key] = (id, lane.parts[index])
                    if debug { print("  revise \(id): \(text) -> \(lane.parts[index])") }
                    onRevise(id, lane.parts[index])
                }
            }
            let spans = Self.spans(timed, language: language)
            var itemStart = lane.queue.last.map { max($0.end, start) } ?? start
            for index in lane.committed..<max(lane.committed, lane.parts.count) {
                let span = index < spans.count && spans.count == lane.parts.count ? spans[index] : (itemStart, end)
                lane.queue.append(Item(text: lane.parts[index], start: span.0, end: span.1, queued: Date(), key: "\(language)|\(lane.chunk)|\(index)"))
                itemStart = span.1
            }
            lane.committed = 0
            lane.parts = []
            lane.terminated = 0
            lane.seen = [:]
            lane.open = false
        } else {
            let now = Date()
            for index in lane.committed..<max(lane.committed, sentences.count) {
                let key = "\(index)|\(sentences[index])"
                if lane.seen[key] == nil { lane.seen[key] = (now, stamp) }
            }
        }
        if debug { print(String(format: "  raw %@ final=%d %.2f-%.2f @%.2f %@", language, isFinal ? 1 : 0, start, end, stamp, raw)) }
        lanes[language] = lane
        update()
    }

    private func settle(_ language: String) {
        var lane = lanes[language]!
        let now = Date()
        while lane.committed < lane.terminated {
            let sentence = lane.parts[lane.committed]
            guard let (first, end) = lane.seen["\(lane.committed)|\(sentence)"] else { break }
            let age = now.timeIntervalSince(first)
            let followed = lane.committed + 1 < lane.parts.count
            guard finishing || age >= (followed ? Self.settleSeconds : Self.lonelySettleSeconds) else { break }
            let itemStart = lane.committed == 0 ? lane.chunkStart : (lane.queue.last?.end ?? lane.chunkStart)
            lane.queue.append(Item(text: sentence, start: max(itemStart, lane.chunkStart), end: end, queued: now, key: "\(language)|\(lane.chunk)|\(lane.committed)"))
            lane.committed += 1
        }
        lanes[language] = lane
    }

    private func pendingText(_ language: String) -> String {
        let lane = lanes[language]!
        let separator = language == "en" ? " " : ""
        let parts = lane.queue.map(\.text) + (lane.committed < lane.parts.count ? Array(lane.parts[lane.committed...]) : [])
        return parts.joined(separator: separator)
    }

    private func japaneseChars(at time: Double) -> Int {
        japaneseHistory.last { $0.0 <= time }?.1 ?? 0
    }

    private func window(_ item: Item) -> (Double, Double) {
        (item.start - 0.5, max(item.end, item.start + 1.0) + 0.5)
    }

    private func japaneseRate(_ item: Item) -> Double {
        let (from, to) = window(item)
        return Double(japaneseChars(at: to) - japaneseChars(at: from)) / (to - from)
    }

    private func isEnglishRegion(_ item: Item) -> Bool {
        let duration = max(item.end - item.start, 0.3)
        if japaneseSpans.contains(where: { min($0.1, item.end) - max($0.0, item.start) >= duration * 0.4 }) { return false }
        let (from, to) = window(item)
        if Self.wordCount(item.text) <= 1 && japaneseChars(at: to) - japaneseChars(at: from) > 0 { return false }
        return japaneseRate(item) < Self.japaneseCharsPerSecond && Self.englishRatio(item.text) >= Self.englishWordRatio
    }

    private func update() {
        settle("ja")
        settle("en")
        var progressed = true
        while progressed {
            progressed = false
            if let item = lanes["ja"]!.queue.first {
                lanes["ja"]!.queue.removeFirst()
                progressed = true
                if !Self.isLatin(item.text) {
                    japaneseSpans.append((item.start, item.end))
                    emit(item, "ja")
                }
                continue
            }
            if let item = lanes["en"]!.queue.first {
                let waited = Date().timeIntervalSince(item.queued)
                let jaReached = (japaneseHistory.last?.0 ?? 0) >= window(item).1
                let jaSpeaking = Self.japaneseCount(pendingText("ja")) >= 2
                guard finishing || waited >= Self.englishMaxWait || (!jaSpeaking && (jaReached || waited >= Self.englishWait)) else { break }
                lanes["en"]!.queue.removeFirst()
                progressed = true
                if isEnglishRegion(item) {
                    emit(item, "en")
                } else if debug {
                    print(String(format: "  drop en %.2f-%.2f jaRate=%.1f ratio=%.2f: %@", item.start, item.end, japaneseRate(item), Self.englishRatio(item.text), item.text))
                }
            }
        }
        publishPartial()
    }

    private func emit(_ item: Item, _ language: String) {
        guard !item.text.isEmpty, !Self.isFiller(item.text) else { return }
        let id = nextId
        nextId += 1
        emitted[item.key] = (id, item.text)
        if debug { print("  emit \(id) \(language): \(item.text)") }
        onFinal(id, item.text, language)
    }

    private func publishPartial() {
        let japanese = pendingText("ja")
        let english = pendingText("en")
        let recent = japaneseChars(at: .greatestFiniteMagnitude) - japaneseChars(at: max(0, feeder.seconds - 2.0))
        let text = Self.isLatin(japanese) || (japanese.isEmpty && recent == 0 && Self.englishRatio(english) >= Self.englishWordRatio) ? english : japanese
        guard text != lastPartial else { return }
        lastPartial = text
        onPartial(text)
    }

    static func bare(_ text: String) -> String {
        String(text.filter { $0.isLetter || $0.isNumber }).lowercased()
    }

    static func spans(_ timed: [(Character, Double, Double)], language: String) -> [(Double, Double)] {
        let terminators: Set<Character> = language == "en" ? [".", "?", "!"] : ["。", "！", "？", "!", "?"]
        var result: [(Double, Double)] = []
        var first: Double?
        var last = 0.0
        for (character, start, end) in timed {
            if !character.isWhitespace {
                if first == nil { first = start }
                last = end
            }
            if terminators.contains(character), let begin = first {
                result.append((begin, last))
                first = nil
            }
        }
        if let begin = first { result.append((begin, last)) }
        return result
    }

    static func japaneseCount(_ text: String) -> Int {
        text.unicodeScalars.filter { (0x3040...0x30FF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value) }.count
    }

    static func words(_ text: String) -> [String] {
        text.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "'" }.map { word -> String in
            var value = String(word)
            for suffix in ["'s", "'re", "'ll", "'ve", "'d", "'m", "n't"] where value.hasSuffix(suffix) {
                value = String(value.dropLast(suffix.count))
            }
            return value
        }.filter { !$0.isEmpty }
    }

    static func wordCount(_ text: String) -> Int {
        words(text).count
    }

    static func isEnglishWord(_ word: String) -> Bool {
        if word.first!.isNumber || ["i", "a", "ok", "okay", "qa", "ai", "api", "ui", "ux"].contains(word) || dictionary.contains(word) { return true }
        let rules: [(String, String)] = [("ies", "y"), ("es", ""), ("s", ""), ("ied", "y"), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e"), ("er", ""), ("est", ""), ("ly", "")]
        for (suffix, replacement) in rules where word.hasSuffix(suffix) && word.count > suffix.count + 2 {
            if dictionary.contains(String(word.dropLast(suffix.count)) + replacement) { return true }
        }
        return false
    }

    static func englishRatio(_ text: String) -> Double {
        let list = words(text)
        guard !list.isEmpty else { return 0 }
        return Double(list.filter(isEnglishWord).count) / Double(list.count)
    }

    static func isLatin(_ text: String) -> Bool {
        var japanese = 0
        var latin = 0
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x3040...0x30FF, 0x4E00...0x9FFF: japanese += 1
            case 0x41...0x5A, 0x61...0x7A: latin += 1
            default: break
            }
        }
        return latin >= 4 && japanese * 3 < latin
    }

    static func split(_ text: String, language: String) -> ([String], String) {
        let terminators: Set<Character> = language == "en" ? [".", "?", "!"] : ["。", "！", "？", "!", "?"]
        var sentences: [String] = []
        var current = ""
        for character in text {
            current.append(character)
            if terminators.contains(character) {
                let piece = current.trimmingCharacters(in: .whitespaces)
                if !piece.isEmpty { sentences.append(piece) }
                current = ""
            }
        }
        return (sentences, current.trimmingCharacters(in: .whitespaces))
    }

    static func clean(_ text: String, language: String) -> String {
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if language == "ja" {
            value = value.replacingOccurrences(of: #"(?<=[\p{Han}\p{Hiragana}\p{Katakana}ー、。！？「」])\s+|\s+(?=[\p{Han}\p{Hiragana}\p{Katakana}ー、。！？「」])"#, with: "", options: .regularExpression)
        }
        return value
    }

    static func isFiller(_ text: String) -> Bool {
        let pieces = text.lowercased().components(separatedBy: CharacterSet(charactersIn: "、。！？!?…ー .,")).filter { !$0.isEmpty }
        return !pieces.isEmpty && pieces.allSatisfy { fillers.contains($0) }
    }
}

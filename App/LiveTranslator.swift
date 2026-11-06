import AppKit
import AudioToolbox
import QuartzCore
import CoreMedia
import ScreenCaptureKit

func installCrashLogging(root: URL) {
    let directory = root.appendingPathComponent("Logs")
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let path = directory.appendingPathComponent("app.log").path
    if let file = fopen(path, "a") {
        dup2(fileno(file), STDERR_FILENO)
        fclose(file)
    }
    setvbuf(stderr, nil, _IONBF, 0)
    let stamp = ISO8601DateFormatter().string(from: Date())
    fputs("\n===== launch \(stamp) =====\n", stderr)
    NSSetUncaughtExceptionHandler { exception in
        fputs("UNCAUGHT EXCEPTION: \(exception.name.rawValue): \(exception.reason ?? "")\n", stderr)
        for line in exception.callStackSymbols { fputs(line + "\n", stderr) }
        fflush(stderr)
    }
    for signalNumber in [SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGABRT, SIGFPE] {
        signal(signalNumber) { number in
            let message = "FATAL SIGNAL \(number)\n"
            write(STDERR_FILENO, message, message.utf8.count)
            var frames = [UnsafeMutableRawPointer?](repeating: nil, count: 64)
            let count = backtrace(&frames, Int32(frames.count))
            backtrace_symbols_fd(&frames, count, STDERR_FILENO)
            signal(number, SIG_DFL)
            raise(number)
        }
    }
}

final class AudioResampler {
    private var previous: Float?
    private var phase = 0.0
    private var sourceRate = 0.0

    func convert(_ mono: [Float], sampleRate: Double) -> Data {
        guard !mono.isEmpty else { return Data() }
        if sampleRate == 16000 {
            previous = nil
            phase = 0
            sourceRate = sampleRate
            return mono.withUnsafeBytes { Data($0) }
        }
        if sourceRate != sampleRate {
            sourceRate = sampleRate
            previous = nil
            phase = 1
        }
        var samples = [Float]()
        samples.reserveCapacity(mono.count + 1)
        samples.append(previous ?? mono[0])
        samples.append(contentsOf: mono)
        let step = sampleRate / 16000
        var output = [Float]()
        output.reserveCapacity(Int(Double(mono.count) / step) + 2)
        while phase < Double(samples.count - 1) {
            let lower = Int(phase)
            let fraction = Float(phase - Double(lower))
            output.append(samples[lower] * (1 - fraction) + samples[lower + 1] * fraction)
            phase += step
        }
        phase -= Double(samples.count - 1)
        previous = samples.last
        return output.withUnsafeBytes { Data($0) }
    }
}

// 自動增益：把會議語音拉到約 -23 dBFS，VAD 與辨識模型才拿到穩定的音量。只放大不衰減，尾端軟限幅避免削波。
// 實測（2026-10-05 迴路測試）擷取路徑固定少 6 dB，真實會議語音落在 -22 到 -34 dBFS。
final class AudioLeveler {
    private let target: Float = -23      // dBFS
    private let gate: Float = -50        // 低於此視為無聲，不更新音量估計
    private let maxGain: Float = 24      // dB
    private let attack = 0.15            // 秒：音量變大時收增益的速度
    private let release = 2.5            // 秒：音量變小時放增益的速度，慢一點才不會把停頓時的底噪抬上來
    private(set) var level: Float = -29  // 估計的語音 RMS（dBFS）；起始值對應 +6 dB，先補回路徑損失
    private(set) var gain: Float = 6     // dB

    private static let ceiling: Float = 0.7  // -3 dBFS：峰值蓋與軟限幅的門檻
    private static let rampSamples = 80      // 5 ms：增益漸變的長度

    func process(_ samples: inout [Float]) {
        guard !samples.isEmpty else { return }
        var sum: Float = 0
        var peak: Float = 0
        for sample in samples {
            sum += sample * sample
            peak = max(peak, abs(sample))
        }
        let db = 20 * log10(max(sqrt(sum / Float(samples.count)), 1e-9))
        if db > gate {
            let tau = db > level ? attack : release
            let alpha = Float(1 - exp(-Double(samples.count) / 16000 / tau))
            level += (db - level) * alpha
        }
        let previous = gain
        gain = min(max(target - level, 0), maxGain)
        // 峰值蓋：突然變大聲（換人講話）時慢速 AGC 來不及收，這裡即時把增益壓到峰值不超過 -3 dBFS
        if peak > 0 {
            gain = min(gain, max(20 * log10(Self.ceiling / peak), 0))
        }
        // 5 ms 線性漸變到新增益，避免拉鍊聲；之後整塊用新增益
        let start = pow(10, previous / 20)
        let end = pow(10, gain / 20)
        let ramp = min(Self.rampSamples, samples.count)
        for index in 0..<samples.count {
            let factor = index < ramp ? start + (end - start) * Float(index) / Float(ramp) : end
            samples[index] = Self.limit(samples[index] * factor)
        }
    }

    // 軟限幅（安全網）：-3 dBFS 以上用 tanh 壓縮，峰值不會到 0 dBFS
    private static func limit(_ value: Float) -> Float {
        let magnitude = abs(value)
        guard magnitude > ceiling else { return value }
        let compressed = ceiling + (1 - ceiling) * tanh((magnitude - ceiling) / (1 - ceiling))
        return value < 0 ? -compressed : compressed
    }
}

final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let queue = DispatchQueue(label: "LiveTranslator.Audio")
    private let resampler = AudioResampler()
    private let leveler = AudioLeveler()
    private var samplesSinceLog = 0
    private let onAudio: (Data) -> Void
    private let onError: (String) -> Void
    private var stream: SCStream?
    private var reportedFormatError = false

    init(onAudio: @escaping (Data) -> Void, onError: @escaping (String) -> Void) {
        self.onAudio = onAudio
        self.onError = onError
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first(where: { $0.displayID == CGMainDisplayID() }) ?? content.displays.first else {
            throw NSError(domain: "LiveTranslator", code: 1, userInfo: [NSLocalizedDescriptionKey: "找不到可擷取的顯示器"])
        }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
    }

    func stop() async {
        guard let stream else { return }
        try? await stream.stopCapture()
        self.stream = nil
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        onError(error.localizedDescription)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferDataIsReady(sampleBuffer),
              let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let format = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee else { return }
        guard format.mFormatID == kAudioFormatLinearPCM,
              format.mBitsPerChannel == 32,
              format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              format.mSampleRate >= 8000,
              format.mChannelsPerFrame >= 1 else {
            if !reportedFormatError {
                reportedFormatError = true
                onError("系統音訊格式不支援")
            }
            return
        }

        var required = 0
        let sizeStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &required,
            bufferListOut: nil,
            bufferListSize: 0,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: 0,
            blockBufferOut: nil
        )
        guard sizeStatus == noErr, required > 0 else { return }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: required, alignment: 16)
        defer { raw.deallocate() }
        let list = raw.bindMemory(to: AudioBufferList.self, capacity: 1)
        var block: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: list,
            bufferListSize: required,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &block
        )
        guard status == noErr else { return }
        let buffers = Array(UnsafeMutableAudioBufferListPointer(list))
        let channels = Int(format.mChannelsPerFrame)
        var mono = [Float]()
        if buffers.count == 1, let pointer = buffers[0].mData {
            let frames = Int(buffers[0].mDataByteSize) / (MemoryLayout<Float>.size * channels)
            let values = pointer.assumingMemoryBound(to: Float.self)
            mono.reserveCapacity(frames)
            if channels == 1 {
                mono.append(contentsOf: UnsafeBufferPointer(start: values, count: frames))
            } else {
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels { sum += values[frame * channels + channel] }
                    mono.append(sum / Float(channels))
                }
            }
        } else if buffers.count >= channels {
            let frames = buffers.prefix(channels).map { Int($0.mDataByteSize) / MemoryLayout<Float>.size }.min() ?? 0
            mono.reserveCapacity(frames)
            for frame in 0..<frames {
                var sum: Float = 0
                for channel in 0..<channels {
                    guard let pointer = buffers[channel].mData else { continue }
                    sum += pointer.assumingMemoryBound(to: Float.self)[frame]
                }
                mono.append(sum / Float(channels))
            }
        }
        withExtendedLifetime(block) {}
        guard !mono.isEmpty else { return }
        deliver(mono, sampleRate: format.mSampleRate)
    }

    // 統一轉成 16 kHz 單聲道 float32，過自動增益後送出；每 60 秒記一次原始音量與增益
    private func deliver(_ mono: [Float], sampleRate: Double) {
        var samples: [Float]
        if sampleRate == 16000 {
            samples = mono
        } else {
            let data = resampler.convert(mono, sampleRate: sampleRate)
            samples = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        guard !samples.isEmpty else { return }
        leveler.process(&samples)
        samplesSinceLog += samples.count
        if samplesSinceLog >= 16000 * 60 {
            samplesSinceLog = 0
            NSLog("audio level: speech %.1f dBFS, gain %+.1f dB", leveler.level, leveler.gain)
        }
        onAudio(samples.withUnsafeBytes { Data($0) })
    }
}

struct WorkerEvent: Decodable {
    let kind: String
    let utterance: Int
    let text: String
    let translation: String
    let message: String
    // warning 的種類：translator_load、refine_load、polish_load、translate_failed、finalize_failed
    let code: String?
}

// 每個 worker 各自一份：切換時舊 worker 還在收尾，兩邊的輸出不能混在同一個緩衝裡
final class OutputBuffer {
    var data = Data()
}

final class AudioSink {
    private let queue = DispatchQueue(label: "LiveTranslator.AudioSink")
    private var handle: FileHandle?

    init(_ handle: FileHandle) {
        self.handle = handle
    }

    func write(_ data: Data) {
        queue.async { [self] in
            guard let handle else { return }
            try? handle.write(contentsOf: data)
        }
    }

    func close() {
        queue.async { [self] in
            try? handle?.close()
            handle = nil
        }
    }
}

// 把擷取到的 16 kHz 單聲道 float32 音訊存成 16-bit WAV，與逐字稿放在同一個資料夾、同名
final class AudioRecorder {
    private let queue = DispatchQueue(label: "LiveTranslator.AudioRecorder")
    private var handle: FileHandle?
    private var url: URL?
    private var writing = false
    private var dataBytes: UInt32 = 0

    func open(at target: URL) {
        queue.sync {
            guard FileManager.default.createFile(atPath: target.path, contents: Self.header(dataBytes: 0)),
                  let opened = try? FileHandle(forWritingTo: target) else {
                NSLog("recorder: cannot create %@", target.path)
                return
            }
            _ = try? opened.seekToEnd()
            handle = opened
            url = target
            writing = false
            dataBytes = 0
        }
    }

    func resume() {
        queue.sync { writing = handle != nil }
    }

    func pause() {
        queue.sync { writing = false }
    }

    func append(_ data: Data) {
        queue.async { [self] in
            guard writing, let handle else { return }
            let pcm = data.withUnsafeBytes { raw -> Data in
                let samples = raw.bindMemory(to: Float.self)
                var output = [Int16](repeating: 0, count: samples.count)
                for index in 0..<samples.count {
                    output[index] = Int16((max(-1, min(1, samples[index])) * 32767).rounded()).littleEndian
                }
                return output.withUnsafeBytes { Data($0) }
            }
            guard (try? handle.write(contentsOf: pcm)) != nil else { return }
            dataBytes &+= UInt32(pcm.count)
            updateHeader()
        }
    }

    func move(to target: URL) {
        queue.sync {
            guard let current = url, current != target else { return }
            do {
                try FileManager.default.moveItem(at: current, to: target)
                url = target
            } catch {
                NSLog("recorder: move failed: %@", error.localizedDescription)
            }
        }
    }

    // 資料夾改名後，已開啟的檔案仍可繼續寫入，只需更新記錄的路徑
    func relocate(to target: URL) {
        queue.sync { if url != nil { url = target } }
    }

    func finish(deleteFile: Bool) {
        queue.sync {
            try? handle?.close()
            handle = nil
            writing = false
            if deleteFile, let url { try? FileManager.default.removeItem(at: url) }
            url = nil
        }
    }

    private func updateHeader() {
        guard let handle, let end = try? handle.offset() else { return }
        var riff = (dataBytes &+ 36).littleEndian
        var size = dataBytes.littleEndian
        try? handle.seek(toOffset: 4)
        try? handle.write(contentsOf: Data(bytes: &riff, count: 4))
        try? handle.seek(toOffset: 40)
        try? handle.write(contentsOf: Data(bytes: &size, count: 4))
        try? handle.seek(toOffset: end)
    }

    private static func header(dataBytes: UInt32) -> Data {
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36) &+ dataBytes)
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(UInt32(16000))
        append(UInt32(16000 * 2))
        append(UInt16(2))
        append(UInt16(16))
        data.append(contentsOf: Array("data".utf8))
        append(dataBytes)
        return data
    }
}

final class SubtitlePanel: NSPanel {
    override func cancelOperation(_ sender: Any?) {}
}

enum PanelState {
    case preparing
    case listening
    case paused
    case ended
    case problem(String, String)
}

struct TranscriptEntry {
    let utterance: Int
    let time: String
    var text: String
    var translation: String?
    // worker 已送回定稿（Qwen 重解、刪贅詞、術語表都處理完）；之後 SpeechAnalyzer 的修正不再覆蓋
    var finalized = false
    // SpeechAnalyzer 判定的語言（ja 或 en）；修正後重送 worker 時要帶上，worker 才不用從文字猜
    var language: String?
    // 這句屬於第幾個 worker；worker 結束時用來把它沒送回定稿的句子標成已定稿
    var generation = 0
}

@MainActor
final class AppController: NSObject, NSApplicationDelegate, NSWindowDelegate, NSTextFieldDelegate {
    private var panel: NSPanel!
    private var statusLabel: NSTextField!
    private var statusDot: NSView!
    private var spinner: NSProgressIndicator!
    private var emptyLabel: NSTextField!
    private var state: PanelState = .preparing
    private var scrollView: NSScrollView!
    private var transcriptView: NSTextView!
    private var toggleButton: NSButton!
    private var endButton: NSButton!
    private var translationButton: NSButton!
    private var refineButton: NSButton!
    private var activeRefine = false
    private var polishButton: NSButton!
    private var activePolish = false
    // 潤稿模型載入失敗（worker 回報 polish_load）：按鈕顯示「載入失敗」，原因放 tooltip；重新開始聆聽時清掉
    private var polishFailed = false
    private var polishNotice = ""
    private var activeTranslation = ""
    private var activeTarget = ""
    private var workerNotice = ""
    private var translateNotice = ""
    private var translatorFailed = false
    private var switching = false
    private var switchToken = 0
    private var workerGeneration = 0
    private let recorder = AudioRecorder()
    private var nameField: NSTextField!
    private var nameWidth: NSLayoutConstraint!
    private var suffixLabel: NSTextField!
    private var nameBeforeEdit = ""
    private var savedURL: URL?
    private var endingWorker = false
    private var worker: Process?
    private var inputPipe: Pipe?
    private var sink: AudioSink?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?
    private var workerError = ""
    private var capture: AudioCapture?
    private var starting = false
    private var running = false
    private var entries: [TranscriptEntry] = []
    private var committedCount = 0
    private var committedLength = 0
    private var liveText = ""
    private var liveStable = ""
    private var engineBox: AnyObject?
    private var utteranceBase = 0
    private let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
    private let textAttributes: [NSAttributedString.Key: Any]
    private let textAloneAttributes: [NSAttributedString.Key: Any]
    private let liveAttributes: [NSAttributedString.Key: Any]
    private let tentativeAttributes: [NSAttributedString.Key: Any]
    private let translationAttributes: [NSAttributedString.Key: Any]

    override init() {
        let textFont = NSFont.systemFont(ofSize: 14)
        let liveFont = NSFont.systemFont(ofSize: 16)
        let translationFont = NSFont.systemFont(ofSize: 18, weight: .semibold)
        let translationColor = NSColor(red: 1.0, green: 0.85, blue: 0.42, alpha: 1.0)
        let tightStyle = NSMutableParagraphStyle()
        tightStyle.paragraphSpacing = 2
        let looseStyle = NSMutableParagraphStyle()
        looseStyle.paragraphSpacing = 18
        textAttributes = [.font: textFont, .foregroundColor: NSColor.white.withAlphaComponent(0.62), .paragraphStyle: tightStyle]
        textAloneAttributes = [.font: textFont, .foregroundColor: NSColor.white.withAlphaComponent(0.62), .paragraphStyle: looseStyle]
        liveAttributes = [.font: liveFont, .foregroundColor: NSColor.white.withAlphaComponent(0.92), .paragraphStyle: looseStyle]
        tentativeAttributes = [.font: liveFont, .foregroundColor: NSColor.white.withAlphaComponent(0.45), .paragraphStyle: looseStyle]
        translationAttributes = [.font: translationFont, .foregroundColor: translationColor, .paragraphStyle: looseStyle]
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installCrashLogging(root: projectRoot())
        migrateLegacySettings()
        makePanel()
        NSApp.activate(ignoringOtherApps: true)
        start(fresh: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        NSLog("applicationShouldTerminate")
        return .terminateNow
    }

    func applicationWillTerminate(_ notification: Notification) {
        NSLog("applicationWillTerminate")
        refreshSavedFile()
        closeSession()
        cancelEngine()
        sink?.close()
        if worker?.isRunning == true { worker?.terminate() }
    }

    func windowWillClose(_ notification: Notification) {
        NSLog("windowWillClose")
        NSApp.terminate(nil)
    }

    // 舊版的 Models/polish-model.txt（重解開關）改名為 final-model.txt：舊設定自動打開重解，檔案刪掉
    private func migrateLegacySettings() {
        let models = projectRoot().appendingPathComponent("Models")
        let legacy = models.appendingPathComponent("polish-model.txt")
        guard FileManager.default.fileExists(atPath: legacy.path) else { return }
        let hasModel = FileManager.default.fileExists(atPath: models.appendingPathComponent(Self.refineModel).path)
        if !FileManager.default.fileExists(atPath: refineChoiceURL.path), hasModel {
            try? "\(Self.refineModel)\n".write(to: refineChoiceURL, atomically: true, encoding: .utf8)
            NSLog("migrate: polish-model.txt -> final-model.txt (%@), polish-model.txt removed", Self.refineModel)
        } else {
            NSLog("migrate: polish-model.txt removed (final-model.txt exists or %@ missing)", Self.refineModel)
        }
        try? FileManager.default.removeItem(at: legacy)
    }

    private func makeLabel(size: CGFloat, color: NSColor, weight: NSFont.Weight = .regular, lines: Int = 2) -> NSTextField {
        let label = NSTextField(wrappingLabelWithString: "")
        label.font = NSFont.systemFont(ofSize: size, weight: weight)
        label.textColor = color
        label.maximumNumberOfLines = lines
        label.lineBreakMode = .byWordWrapping
        label.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return label
    }

    private func makePanel() {
        let rect = NSRect(x: 0, y: 0, width: 760, height: 460)
        panel = SubtitlePanel(contentRect: rect, styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        panel.title = "即時翻譯"
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isMovableByWindowBackground = true
        panel.hidesOnDeactivate = false
        panel.isFloatingPanel = true
        panel.level = .screenSaver
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.minSize = NSSize(width: 480, height: 280)
        panel.standardWindowButton(.zoomButton)?.isHidden = true
        panel.standardWindowButton(.closeButton)?.toolTip = "結束並儲存逐字稿"
        panel.delegate = self
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true

        let effect = NSVisualEffectView(frame: rect)
        effect.material = .hudWindow
        effect.blendingMode = .behindWindow
        effect.state = .active
        effect.wantsLayer = true
        effect.layer?.cornerRadius = 14
        effect.layer?.masksToBounds = true
        panel.contentView = effect

        statusDot = NSView()
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 4
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isDisplayedWhenStopped = false
        spinner.translatesAutoresizingMaskIntoConstraints = false
        statusLabel = NSTextField(labelWithString: "")
        statusLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        statusLabel.textColor = NSColor.white.withAlphaComponent(0.72)
        statusLabel.lineBreakMode = .byTruncatingTail
        statusLabel.maximumNumberOfLines = 1
        toggleButton = makeIconButton(action: #selector(toggle))
        endButton = makeIconButton(action: #selector(end))
        setSymbol(endButton, "stop.fill", tip: "終止這一段")
        nameField = NSTextField(string: "")
        nameField.isBordered = false
        nameField.drawsBackground = false
        nameField.focusRingType = .none
        nameField.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        nameField.textColor = NSColor(srgbRed: 1.0, green: 0.85, blue: 0.42, alpha: 1.0)
        nameField.usesSingleLineMode = true
        nameField.lineBreakMode = .byClipping
        nameField.alignment = .right
        nameField.delegate = self
        nameField.toolTip = "點一下修改檔名，逐字稿與錄音一起改"
        nameField.translatesAutoresizingMaskIntoConstraints = false
        suffixLabel = NSTextField(labelWithString: ".txt")
        suffixLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        suffixLabel.textColor = NSColor.white.withAlphaComponent(0.45)
        refineButton = NSButton(title: "", target: self, action: #selector(toggleRefine))
        refineButton.isBordered = false
        refineButton.bezelStyle = .regularSquare
        refineButton.translatesAutoresizingMaskIntoConstraints = false
        updateRefineButton()
        polishButton = NSButton(title: "", target: self, action: #selector(togglePolish))
        polishButton.isBordered = false
        polishButton.bezelStyle = .regularSquare
        polishButton.translatesAutoresizingMaskIntoConstraints = false
        updatePolishButton()
        translationButton = NSButton(title: "", target: self, action: #selector(showTranslationMenu))
        translationButton.isBordered = false
        translationButton.bezelStyle = .regularSquare
        translationButton.translatesAutoresizingMaskIntoConstraints = false
        updateTranslationButton()

        let indicator = NSView()
        indicator.translatesAutoresizingMaskIntoConstraints = false
        indicator.addSubview(statusDot)
        indicator.addSubview(spinner)
        let top = NSStackView(views: [indicator, statusLabel, refineButton, polishButton, translationButton, nameField, suffixLabel, endButton, toggleButton])
        top.orientation = .horizontal
        top.alignment = .centerY
        top.distribution = .fill
        top.spacing = 8
        top.setCustomSpacing(10, after: refineButton)
        top.setCustomSpacing(10, after: polishButton)
        top.setCustomSpacing(12, after: translationButton)
        top.setCustomSpacing(0, after: nameField)
        top.setCustomSpacing(12, after: suffixLabel)
        top.setCustomSpacing(2, after: endButton)
        statusLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        statusLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        toggleButton.setContentHuggingPriority(.required, for: .horizontal)
        endButton.setContentHuggingPriority(.required, for: .horizontal)
        translationButton.setContentHuggingPriority(.required, for: .horizontal)
        refineButton.setContentHuggingPriority(.required, for: .horizontal)
        refineButton.setContentCompressionResistancePriority(.required, for: .horizontal)
        polishButton.setContentHuggingPriority(.required, for: .horizontal)
        polishButton.setContentCompressionResistancePriority(.required, for: .horizontal)
        translationButton.setContentCompressionResistancePriority(.required, for: .horizontal)
        suffixLabel.setContentHuggingPriority(.required, for: .horizontal)
        nameWidth = nameField.widthAnchor.constraint(equalToConstant: 100)
        NSLayoutConstraint.activate([
            indicator.widthAnchor.constraint(equalToConstant: 16),
            indicator.heightAnchor.constraint(equalToConstant: 16),
            statusDot.widthAnchor.constraint(equalToConstant: 8),
            statusDot.heightAnchor.constraint(equalToConstant: 8),
            statusDot.centerXAnchor.constraint(equalTo: indicator.centerXAnchor),
            statusDot.centerYAnchor.constraint(equalTo: statusLabel.firstBaselineAnchor, constant: -(statusLabel.font?.capHeight ?? 9) / 2),
            spinner.centerXAnchor.constraint(equalTo: indicator.centerXAnchor),
            spinner.centerYAnchor.constraint(equalTo: statusDot.centerYAnchor),
            toggleButton.widthAnchor.constraint(equalToConstant: 28),
            toggleButton.heightAnchor.constraint(equalToConstant: 24),
            endButton.widthAnchor.constraint(equalToConstant: 28),
            endButton.heightAnchor.constraint(equalToConstant: 24),
            nameWidth
        ])

        let divider = NSBox()
        divider.boxType = .custom
        divider.borderWidth = 0
        divider.fillColor = NSColor.white.withAlphaComponent(0.08)
        divider.translatesAutoresizingMaskIntoConstraints = false

        scrollView = NSTextView.scrollableTextView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.setContentHuggingPriority(NSLayoutConstraint.Priority(1), for: .vertical)
        scrollView.setContentCompressionResistancePriority(NSLayoutConstraint.Priority(1), for: .vertical)
        top.setHuggingPriority(.required, for: .vertical)
        transcriptView = scrollView.documentView as? NSTextView
        transcriptView.isEditable = false
        transcriptView.isSelectable = true
        transcriptView.drawsBackground = false
        transcriptView.textContainerInset = NSSize(width: 0, height: 4)
        transcriptView.textContainer?.lineFragmentPadding = 0

        emptyLabel = makeLabel(size: 14, color: NSColor.white.withAlphaComponent(0.38), lines: 2)
        emptyLabel.alignment = .center
        emptyLabel.stringValue = "播放含有日文或英文的影片或會議\n字幕會即時出現在這裡"
        emptyLabel.translatesAutoresizingMaskIntoConstraints = false

        let stack = NSStackView(views: [scrollView, divider, top])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.distribution = .fill
        stack.spacing = 8
        stack.setCustomSpacing(12, after: scrollView)
        stack.translatesAutoresizingMaskIntoConstraints = false
        effect.addSubview(stack)
        effect.addSubview(emptyLabel)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: effect.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: effect.trailingAnchor, constant: -14),
            stack.topAnchor.constraint(equalTo: effect.topAnchor, constant: 32),
            stack.bottomAnchor.constraint(equalTo: effect.bottomAnchor, constant: -8),
            top.widthAnchor.constraint(equalTo: stack.widthAnchor),
            top.heightAnchor.constraint(equalToConstant: 28),
            divider.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: 6),
            divider.heightAnchor.constraint(equalToConstant: 1),
            scrollView.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -6),
            emptyLabel.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor),
            emptyLabel.centerYAnchor.constraint(equalTo: scrollView.centerYAnchor),
            emptyLabel.widthAnchor.constraint(lessThanOrEqualTo: scrollView.widthAnchor)
        ])
        setState(.preparing)
        if let screen = NSScreen.main {
            let frame = screen.visibleFrame
            panel.setFrameOrigin(NSPoint(x: frame.midX - rect.width / 2, y: frame.minY + 80))
        } else {
            panel.center()
        }
        panel.makeKeyAndOrderFront(nil)
    }

    @objc private func toggle() {
        switch state {
        case .preparing, .listening:
            pause()
        case .paused:
            resume()
        case .ended:
            startNewSession()
        case .problem:
            start(fresh: !hasContent)
        }
    }

    @objc private func end() {
        NSLog("action: end (state=%@, entries=%d)", String(describing: state), entries.count)
        switching = false
        switchToken += 1
        starting = false
        running = false
        let oldCapture = capture
        capture = nil
        let oldSink = sink
        sink = nil
        inputPipe = nil
        endingWorker = worker != nil
        let oldEngine = engineBox
        engineBox = nil
        recorder.pause()
        Task { @MainActor in
            await oldCapture?.stop()
            if #available(macOS 26.0, *), let engine = oldEngine as? SpeechEngine {
                await engine.finish()
            }
            oldSink?.close()
        }
        refreshSavedFile()
        nameField.stringValue = savedURL?.deletingPathExtension().lastPathComponent ?? ""
        updateNameWidth()
        setState(.ended)
    }

    private var hasContent: Bool {
        !entries.isEmpty || !liveText.isEmpty
    }

    private func pause() {
        NSLog("action: pause")
        guard starting || running else { return }
        starting = false
        running = false
        let oldCapture = capture
        capture = nil
        let engine = engineBox
        recorder.pause()
        Task { @MainActor in
            await oldCapture?.stop()
            if #available(macOS 26.0, *), let engine = engine as? SpeechEngine {
                await engine.flush()
            }
        }
        setState(.paused)
    }

    private func resume() {
        NSLog("action: resume")
        guard worker?.isRunning == true, sink != nil, activeRefine == refineEnabled, activePolish == polishEnabled,
              activeTranslation == (preferredTranslation ?? ""), activeTarget == translationTarget else {
            start(fresh: false)
            return
        }
        starting = true
        setState(.preparing)
        beginCapture()
    }

    // 設定改變時，正在聽就立刻重啟 worker；暫停、已終止、出錯時下次開始或繼續時套用
    private func restartIfListening() {
        // 切換中改的設定，等舊 worker 收完尾、新 worker 啟動時自然讀到
        guard !switching else { return }
        switch state {
        case .preparing, .listening:
            restartEngine()
        case .paused, .ended, .problem:
            break
        }
    }

    // 比照 end() 優雅收尾：停止擷取 → 讓 SpeechEngine 把未定稿的句子定稿送出 → 關舊 worker 的 stdin 讓它 EOF，
    // 舊 worker 的輸出 handler 繼續把剩下的定稿與譯文套到 entries → 舊程序結束後才用新設定啟動。
    // 這段期間（收尾約數秒，加上新模型載入）不擷取系統音訊，也不錄音
    private func restartEngine() {
        NSLog("action: switching worker (drain old)")
        starting = false
        running = false
        switching = true
        switchToken += 1
        let token = switchToken
        let oldCapture = capture
        capture = nil
        let oldEngine = engineBox
        let oldSink = sink
        sink = nil
        let oldWorker = worker
        // 舊 worker 的 handler 自己留著處理到 EOF；這裡只放開對它的引用，之後 start() 不會去動它
        worker = nil
        inputPipe = nil
        outputPipe = nil
        errorPipe = nil
        endingWorker = false
        recorder.pause()
        setState(.preparing)
        Task { @MainActor in
            await oldCapture?.stop()
            if #available(macOS 26.0, *), let engine = oldEngine as? SpeechEngine {
                await engine.finish()
            }
            if self.engineBox === oldEngine { self.engineBox = nil }
            oldSink?.close()
            if let oldWorker {
                // 收尾不應超過 60 秒；真的卡住就強制結束，entries 由 terminationHandler 補標定稿
                DispatchQueue.main.asyncAfter(deadline: .now() + 60) {
                    if oldWorker.isRunning { oldWorker.terminate() }
                }
                await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                    DispatchQueue.global().async {
                        oldWorker.waitUntilExit()
                        continuation.resume()
                    }
                }
            }
            // 切換期間按了終止或又換了設定：這次不啟動，交給最後一次
            guard token == self.switchToken else { return }
            self.switching = false
            self.liveText = ""
            self.liveStable = ""
            self.start(fresh: false)
        }
    }

    private func startNewSession() {
        start(fresh: true)
    }

    private static let refineModel = "Qwen3-ASR-1.7B-8bit"

    private var refineChoiceURL: URL {
        projectRoot().appendingPathComponent("Models/final-model.txt")
    }

    // 啟動 worker 時讀 final-model.txt，開啟才傳 --refine；內容為空或 none 視為關閉
    private var refineEnabled: Bool {
        guard let value = try? String(contentsOf: refineChoiceURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) else { return false }
        return !value.isEmpty && value != "none"
    }

    private func updateRefineButton() {
        let on = refineEnabled
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12, weight: .medium),
            .foregroundColor: on ? NSColor(srgbRed: 1.0, green: 0.85, blue: 0.42, alpha: 1.0) : NSColor.white.withAlphaComponent(0.3)
        ]
        refineButton.attributedTitle = NSAttributedString(string: on ? "Qwen 重解：開" : "Qwen 重解：關", attributes: attributes)
        let effect = "較準但定稿晚約 1 秒"
        refineButton.toolTip = on
            ? "每句定稿後用 Qwen3-ASR 重新辨識，\(effect)、多佔 2.5 GB 記憶體。點一下關閉"
            : "點一下開啟：每句定稿後用 Qwen3-ASR 重新辨識，\(effect)、多佔 2.5 GB 記憶體"
    }

    @objc private func toggleRefine() {
        if refineEnabled {
            try? FileManager.default.removeItem(at: refineChoiceURL)
        } else {
            let model = projectRoot().appendingPathComponent("Models/\(Self.refineModel)")
            guard FileManager.default.fileExists(atPath: model.path) else {
                let alert = NSAlert()
                alert.messageText = "尚未下載 Qwen3-ASR 模型"
                alert.informativeText = "缺少 Models/\(Self.refineModel)。下載方式見 Docs/SETUP.md 第 9 節「Qwen 重解（預設開啟）」。"
                alert.runModal()
                return
            }
            try? "\(Self.refineModel)\n".write(to: refineChoiceURL, atomically: true, encoding: .utf8)
        }
        NSLog("action: refine %@", refineEnabled ? "on" : "off")
        updateRefineButton()
        restartIfListening()
    }

    // 翻譯前潤稿（預設關閉）：設定檔用 prepolish-model.txt，polish-model.txt 是舊版的重解開關，migrateLegacySettings 會刪掉
    private static let polishModel = "Qwen3-4B-4bit"

    private var polishChoiceURL: URL {
        projectRoot().appendingPathComponent("Models/prepolish-model.txt")
    }

    private var polishInstalled: Bool {
        FileManager.default.fileExists(atPath: projectRoot().appendingPathComponent("Models/\(Self.polishModel)/config.json").path)
    }

    // 啟動 worker 時讀 prepolish-model.txt；缺檔、空、none 或模型目錄不存在都視為關閉
    private var polishEnabled: Bool {
        guard let value = try? String(contentsOf: polishChoiceURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) else { return false }
        return !value.isEmpty && value != "none" && polishInstalled
    }

    private func updatePolishButton() {
        let on = polishEnabled
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12, weight: .medium),
            .foregroundColor: on ? NSColor(srgbRed: 1.0, green: 0.85, blue: 0.42, alpha: 1.0) : NSColor.white.withAlphaComponent(0.3)
        ]
        let failed = on && polishFailed
        polishButton.attributedTitle = NSAttributedString(string: failed ? "潤稿：載入失敗" : on ? "潤稿：開" : "潤稿：關", attributes: attributes)
        let effect = "翻譯前先用 Qwen3-4B 刪贅詞、補標點、修正聽錯的術語（不翻譯、不改意思），譯文晚約 2 秒、多佔約 2.4 GB 記憶體"
        var tip = on ? "\(effect)。點一下關閉" : "點一下開啟：\(effect)"
        if failed, !polishNotice.isEmpty { tip += "\n\(polishNotice)" }
        polishButton.toolTip = tip
    }

    @objc private func togglePolish() {
        if polishEnabled {
            try? FileManager.default.removeItem(at: polishChoiceURL)
        } else {
            guard polishInstalled else {
                let alert = NSAlert()
                alert.messageText = "尚未下載潤稿模型"
                alert.informativeText = "缺少 Models/\(Self.polishModel)。下載方式見 Docs/SETUP.md 第 9 節「翻譯前潤稿（選用，預設關閉）」。"
                alert.runModal()
                return
            }
            try? "\(Self.polishModel)\n".write(to: polishChoiceURL, atomically: true, encoding: .utf8)
        }
        NSLog("action: polish %@", polishEnabled ? "on" : "off")
        // 設定一改，之前的載入失敗就作廢（下次開始聆聽才會重新載入）
        polishFailed = false
        polishNotice = ""
        updatePolishButton()
        restartIfListening()
    }

    // Models/glossary.txt 的正確寫法（「聽錯=正確」取等號右邊，單獨一行的整行採用），worker 讀同一個檔做替換與翻譯提示
    private func glossaryTerms() -> [String] {
        let url = projectRoot().appendingPathComponent("Models/glossary.txt")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        var terms: [String] = []
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            let term = (line.split(separator: "=", maxSplits: 1).last.map(String.init) ?? line).trimmingCharacters(in: .whitespaces)
            if !term.isEmpty, !terms.contains(term) { terms.append(term) }
        }
        return terms
    }

    // 已知的 PLaMo 翻譯模型，依優先順序；選單只列出已下載的
    private static let translationModels: [(name: String, label: String, detail: String)] = [
        ("plamo-3-translate-2b-MLX-4bit", "PLaMo 2B", "預設，快"),
        ("plamo-3-translate-8b-MLX-4bit", "PLaMo 8B", "較準，多約 3.7 GB 記憶體、較慢")
    ]

    private static let translationTargets: [(value: String, label: String)] = [
        ("zh-hant", "繁體中文"),
        ("ja-en", "日英互譯")
    ]

    private var translationChoiceURL: URL {
        projectRoot().appendingPathComponent("Models/translation-model.txt")
    }

    private var translationTargetURL: URL {
        projectRoot().appendingPathComponent("Models/translation-target.txt")
    }

    private func translationInstalled(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: projectRoot().appendingPathComponent("Models/\(name)/config.json").path)
    }

    private var installedTranslations: [String] {
        var names = Self.translationModels.map(\.name)
        if let chosen = try? String(contentsOf: translationChoiceURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           chosen.hasPrefix("plamo"), !names.contains(chosen) {
            names.append(chosen)
        }
        return names.filter(translationInstalled)
    }

    // translation-model.txt 指定了 plamo 開頭且已下載的就用它（舊的 Hy-MT2 設定直接忽略），否則用第一個已下載的
    private var preferredTranslation: String? {
        if let chosen = try? String(contentsOf: translationChoiceURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           chosen.hasPrefix("plamo"), translationInstalled(chosen) {
            return chosen
        }
        return installedTranslations.first
    }

    // translation-target.txt：zh-hant 或 ja-en；缺檔或不認得都當 zh-hant
    private var translationTarget: String {
        let value = (try? String(contentsOf: translationTargetURL, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return Self.translationTargets.contains { $0.value == value } ? value : "zh-hant"
    }

    private func translationLabel(_ name: String) -> String {
        Self.translationModels.first { $0.name == name }?.label ?? name
    }

    private func targetLabel(_ value: String) -> String {
        Self.translationTargets.first { $0.value == value }?.label ?? value
    }

    private func updateTranslationButton() {
        let model = preferredTranslation
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12, weight: .medium),
            .foregroundColor: NSColor.white.withAlphaComponent(model == nil ? 0.3 : 0.55)
        ]
        let status = model == nil ? "未安裝" : translatorFailed ? "載入失敗" : targetLabel(translationTarget)
        translationButton.attributedTitle = NSAttributedString(string: "譯文：\(status)", attributes: attributes)
        var tip = model.map { "翻譯模型：\(translationLabel($0))，點一下切換譯文與模型" } ?? "尚未下載 PLaMo 翻譯模型，只有逐字稿。下載方式見 Docs/SETUP.md"
        if !workerNotice.isEmpty { tip += "\n\(workerNotice)" }
        if !translateNotice.isEmpty { tip += "\n\(translateNotice)" }
        translationButton.toolTip = tip
    }

    @objc private func showTranslationMenu() {
        let menu = NSMenu()
        menu.addItem(NSMenuItem.sectionHeader(title: "譯文"))
        for target in Self.translationTargets {
            let item = NSMenuItem(title: target.label, action: #selector(chooseTarget(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = target.value
            item.state = target.value == translationTarget ? .on : .off
            menu.addItem(item)
        }
        menu.addItem(.separator())
        menu.addItem(NSMenuItem.sectionHeader(title: "翻譯模型"))
        let current = preferredTranslation
        if installedTranslations.isEmpty {
            let item = NSMenuItem(title: "尚未下載（見 Docs/SETUP.md）", action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        for name in installedTranslations {
            let detail = Self.translationModels.first { $0.name == name }?.detail
            let item = NSMenuItem(title: detail.map { "\(translationLabel(name))　\($0)" } ?? name, action: #selector(chooseTranslation(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = name
            item.state = name == current ? .on : .off
            menu.addItem(item)
        }
        // 字幕視窗在 screenSaver 層級，比選單高；選單打開期間暫時降下來，否則選單會被蓋住
        let level = panel.level
        panel.level = .floating
        menu.popUp(positioning: nil, at: NSPoint(x: 0, y: translationButton.bounds.height + 4), in: translationButton)
        panel.level = level
    }

    @objc private func chooseTarget(_ sender: NSMenuItem) {
        guard let value = sender.representedObject as? String, value != translationTarget else { return }
        NSLog("action: switch translation target to %@", value)
        try? "\(value)\n".write(to: translationTargetURL, atomically: true, encoding: .utf8)
        updateTranslationButton()
        restartIfListening()
    }

    @objc private func chooseTranslation(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String, name != preferredTranslation else { return }
        NSLog("action: switch translation model to %@", name)
        try? "\(name)\n".write(to: translationChoiceURL, atomically: true, encoding: .utf8)
        updateTranslationButton()
        restartIfListening()
    }

    // 一開始就建立 Transcripts/<名稱>/ 資料夾，裡面放同名的逐字稿與錄音檔，之後邊聽邊寫入
    private func openSession() {
        let name = uniqueName(base: dateName())
        let folder = transcriptsDirectory().appendingPathComponent(name)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("\(name).txt")
        savedURL = url
        writeTranscript(to: url, title: name)
        recorder.open(at: folder.appendingPathComponent("\(name).wav"))
    }

    // 整段沒有任何字幕時，整個資料夾一併刪掉
    private func closeSession() {
        let empty = !hasContent
        recorder.finish(deleteFile: empty)
        if empty, let url = savedURL { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        savedURL = nil
    }

    private func renameSession(to name: String) {
        guard let current = savedURL else { return }
        let currentName = current.deletingPathExtension().lastPathComponent
        guard !name.isEmpty, name != currentName else {
            nameField.stringValue = currentName
            updateNameWidth()
            return
        }
        let finalName = uniqueName(base: name)
        let oldFolder = current.deletingLastPathComponent()
        let newFolder = transcriptsDirectory().appendingPathComponent(finalName)
        // 先在原資料夾裡把兩個檔案改名，再改資料夾名
        do {
            try FileManager.default.moveItem(at: current, to: oldFolder.appendingPathComponent("\(finalName).txt"))
        } catch {
            NSLog("rename failed: %@", error.localizedDescription)
            nameField.stringValue = currentName
            updateNameWidth()
            return
        }
        recorder.move(to: oldFolder.appendingPathComponent("\(finalName).wav"))
        var folder = oldFolder
        do {
            try FileManager.default.moveItem(at: oldFolder, to: newFolder)
            folder = newFolder
        } catch {
            NSLog("rename folder failed: %@", error.localizedDescription)
        }
        recorder.relocate(to: folder.appendingPathComponent("\(finalName).wav"))
        let url = folder.appendingPathComponent("\(finalName).txt")
        savedURL = url
        refreshSavedFile()
        nameField.stringValue = finalName
        updateNameWidth()
        statusLabel.toolTip = url.path
        NSLog("renamed session to %@", finalName)
    }

    private func dateName() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: Date())
    }

    private func transcriptsDirectory() -> URL {
        let directory = projectRoot().appendingPathComponent("Transcripts")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func uniqueName(base: String) -> String {
        var name = base
        var index = 2
        while FileManager.default.fileExists(atPath: transcriptsDirectory().appendingPathComponent(name).path) {
            name = "\(base)-\(index)"
            index += 1
        }
        return name
    }

    private func sanitizedName(_ raw: String) -> String {
        var name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.lowercased().hasSuffix(".txt") { name = String(name.dropLast(4)) }
        name = name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
        return name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func updateNameWidth() {
        let width = nameField.attributedStringValue.size().width + 10
        nameWidth.constant = max(40, min(width, 320))
    }

    func controlTextDidBeginEditing(_ notification: Notification) {
        nameBeforeEdit = nameField.stringValue
    }

    func controlTextDidChange(_ notification: Notification) {
        updateNameWidth()
    }

    func controlTextDidEndEditing(_ notification: Notification) {
        renameSession(to: sanitizedName(nameField.stringValue))
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
            nameField.stringValue = nameBeforeEdit
            updateNameWidth()
            panel.makeFirstResponder(nil)
            return true
        }
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            panel.makeFirstResponder(nil)
            return true
        }
        return false
    }


    private func makeIconButton(action: Selector) -> NSButton {
        let button = NSButton(title: "", target: self, action: action)
        button.isBordered = false
        button.bezelStyle = .regularSquare
        button.imagePosition = .imageOnly
        button.contentTintColor = NSColor.white.withAlphaComponent(0.75)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }

    private func setSymbol(_ button: NSButton, _ name: String, tip: String) {
        let config = NSImage.SymbolConfiguration(pointSize: 13, weight: .semibold)
        button.image = NSImage(systemSymbolName: name, accessibilityDescription: tip)?.withSymbolConfiguration(config)
        button.toolTip = tip
    }

    private func setDot(_ color: NSColor?, pulse: Bool) {
        statusDot.isHidden = color == nil
        statusDot.layer?.backgroundColor = color?.cgColor
        statusDot.layer?.removeAnimation(forKey: "pulse")
        if pulse {
            let animation = CABasicAnimation(keyPath: "opacity")
            animation.fromValue = 1.0
            animation.toValue = 0.35
            animation.duration = 1.1
            animation.autoreverses = true
            animation.repeatCount = .infinity
            animation.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            statusDot.layer?.add(animation, forKey: "pulse")
        }
    }

    private func setState(_ newState: PanelState) {
        if case .listening = newState, case .listening = state, statusDot.layer?.animation(forKey: "pulse") != nil { return }
        NSLog("state -> %@", String(describing: newState))
        state = newState
        var ended = false
        if case .ended = newState { ended = true }
        nameField.isHidden = !ended
        suffixLabel.isHidden = !ended
        endButton.isHidden = ended
        switch newState {
        case .preparing:
            spinner.startAnimation(nil)
            setDot(nil, pulse: false)
            statusLabel.stringValue = switching ? "切換中…" : "準備中…"
            statusLabel.toolTip = nil
            setSymbol(toggleButton, "pause.fill", tip: "暫停")
        case .listening:
            spinner.stopAnimation(nil)
            setDot(NSColor(srgbRed: 0.30, green: 0.85, blue: 0.47, alpha: 1), pulse: true)
            statusLabel.stringValue = "聆聽中"
            statusLabel.toolTip = nil
            setSymbol(toggleButton, "pause.fill", tip: "暫停")
        case .paused:
            spinner.stopAnimation(nil)
            setDot(NSColor.white.withAlphaComponent(0.35), pulse: false)
            statusLabel.stringValue = "已暫停"
            statusLabel.toolTip = nil
            setSymbol(toggleButton, "play.fill", tip: "繼續")
        case .ended:
            spinner.stopAnimation(nil)
            setDot(NSColor.white.withAlphaComponent(0.35), pulse: false)
            statusLabel.stringValue = savedURL != nil && hasContent ? "已儲存" : "已終止"
            statusLabel.toolTip = savedURL?.path
            setSymbol(toggleButton, "play.fill", tip: "開始新的一段")
        case .problem(let message, let detail):
            spinner.stopAnimation(nil)
            setDot(NSColor(srgbRed: 1.0, green: 0.62, blue: 0.25, alpha: 1), pulse: false)
            statusLabel.stringValue = message
            statusLabel.toolTip = detail.isEmpty ? nil : detail
            setSymbol(toggleButton, "arrow.clockwise", tip: "重試")
        }
    }

    private func projectRoot() -> URL {
        Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
    }

    private func transcriptLines(includeLive: Bool) -> [String] {
        var lines: [String] = []
        for entry in entries {
            lines.append("[\(entry.time)] \(entry.text)")
            if let translation = entry.translation, !translation.isEmpty {
                lines.append("           \(translation)")
            }
        }
        if includeLive && !liveText.isEmpty {
            lines.append("[\(clock.string(from: Date()))] \(liveText)（未定稿）")
        }
        return lines
    }

    private func resetTranscript() {
        entries.removeAll()
        committedCount = 0
        committedLength = 0
        liveText = ""
        liveStable = ""
        transcriptView.textStorage?.setAttributedString(NSAttributedString())
        emptyLabel?.isHidden = false
    }

    private func writeTranscript(to url: URL, title: String) {
        let lines = ["# 即時翻譯逐字稿 \(title)", ""] + transcriptLines(includeLive: true)
        try? (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private func refreshSavedFile() {
        guard let url = savedURL else { return }
        writeTranscript(to: url, title: url.deletingPathExtension().lastPathComponent)
    }

    private func renderTranscript() {
        guard let storage = transcriptView.textStorage else { return }
        let wasAtBottom = scrollView.documentVisibleRect.maxY >= transcriptView.bounds.height - 40
        storage.beginEditing()
        storage.deleteCharacters(in: NSRange(location: committedLength, length: storage.length - committedLength))
        for entry in entries[committedCount...] {
            if let translation = entry.translation, !translation.isEmpty {
                storage.append(NSAttributedString(string: entry.text + "\n", attributes: textAttributes))
                storage.append(NSAttributedString(string: translation + "\n", attributes: translationAttributes))
            } else {
                storage.append(NSAttributedString(string: entry.text + "\n", attributes: textAloneAttributes))
            }
        }
        if !liveText.isEmpty {
            let stable = liveText.hasPrefix(liveStable) ? liveStable : ""
            storage.append(NSAttributedString(string: stable, attributes: liveAttributes))
            storage.append(NSAttributedString(string: String(liveText.dropFirst(stable.count)) + "\n", attributes: tentativeAttributes))
        }
        storage.endEditing()
        emptyLabel.isHidden = storage.length > 0
        while committedCount < entries.count, entries[committedCount].finalized {
            committedLength += renderedLength(entries[committedCount])
            committedCount += 1
        }
        if wasAtBottom {
            transcriptView.scrollRangeToVisible(NSRange(location: storage.length, length: 0))
        }
    }

    private func start(fresh: Bool) {
        guard !starting && !running else { return }
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        errorPipe?.fileHandleForReading.readabilityHandler = nil
        sink?.close()
        sink = nil
        if worker?.isRunning == true { worker?.terminate() }
        cancelEngine()
        worker = nil
        inputPipe = nil
        outputPipe = nil
        errorPipe = nil
        endingWorker = false
        switching = false
        workerGeneration += 1
        let generation = workerGeneration
        starting = true
        if fresh {
            closeSession()
            resetTranscript()
            openSession()
        }
        setState(.preparing)

        let root = projectRoot()
        activeRefine = refineEnabled
        activePolish = polishEnabled
        utteranceBase = entries.map(\.utterance).max() ?? 0
        workerNotice = ""
        translateNotice = ""
        translatorFailed = false
        polishFailed = false
        polishNotice = ""
        updatePolishButton()
        let python = root.appendingPathComponent(".venv/bin/python")
        let script = Bundle.main.resourceURL!.appendingPathComponent("worker.py")
        activeTranslation = preferredTranslation ?? ""
        activeTarget = translationTarget
        updateTranslationButton()
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: script.path) else {
            starting = false
            setState(.problem("尚未完成安裝", "請先在專案目錄執行 Scripts/setup.sh"))
            return
        }

        let process = Process()
        process.executableURL = python
        // worker 只做逐字稿；Qwen 重解、翻譯前潤稿與翻譯各自開著時才帶模型路徑，可同時開
        var arguments = [script.path, "--transcribe", root.appendingPathComponent("Models").path]
        if activeRefine { arguments += ["--refine", root.appendingPathComponent("Models/\(Self.refineModel)").path] }
        // 潤稿開著且模型已下載才帶 --polish
        if activePolish { arguments += ["--polish", root.appendingPathComponent("Models/\(Self.polishModel)").path] }
        if let model = preferredTranslation {
            arguments += ["--translate", root.appendingPathComponent("Models/\(model)").path, "--target", activeTarget]
        }
        process.arguments = arguments
        process.environment = ProcessInfo.processInfo.environment.merging([
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1"
        ]) { _, new in new }
        let input = Pipe()
        let output = Pipe()
        let errors = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
        worker = process
        inputPipe = input
        sink = AudioSink(input.fileHandleForWriting)
        outputPipe = output
        errorPipe = errors
        workerError = ""
        let buffer = OutputBuffer()

        // handler 直接用 output / errors：切換時 Pipe 要活到 worker 收完尾（EOF 時 handler 清掉，引用才放開）
        output.fileHandleForReading.readabilityHandler = { [weak self] _ in
            let handle = output.fileHandleForReading
            let data = handle.availableData
            guard !data.isEmpty else { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async { self?.receive(data, buffer: buffer, process: process) }
        }
        errors.fileHandleForReading.readabilityHandler = { [weak self] _ in
            let handle = errors.fileHandleForReading
            let data = handle.availableData
            guard !data.isEmpty else { handle.readabilityHandler = nil; return }
            guard let value = String(data: data, encoding: .utf8), !value.isEmpty else { return }
            fputs(value, stderr)
            DispatchQueue.main.async {
                guard let self else { return }
                self.workerError = String((self.workerError + value).suffix(1200))
            }
        }
        process.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                guard let self else { return }
                // 不論怎麼結束，它還沒送回定稿的句子都不會再有了；標成已定稿，commit 才不會卡住
                self.finalizeOrphans(generation: generation)
                guard self.worker === process else { return }
                self.worker = nil
                if self.endingWorker {
                    self.endingWorker = false
                    return
                }
                self.starting = false
                self.running = false
                self.capture = nil
                let detail = self.workerError.trimmingCharacters(in: .whitespacesAndNewlines)
                self.setState(.problem("翻譯引擎已停止", detail.isEmpty ? "結束代碼 \(process.terminationStatus)" : String(detail.suffix(600))))
            }
        }
        do {
            try process.run()
        } catch {
            starting = false
            worker = nil
            setState(.problem("無法啟動翻譯引擎", error.localizedDescription))
        }
    }

    // process 不是目前的 worker 時（切換中舊 worker 在收尾）：只套用 final 與譯文，不碰狀態與提示
    private func receive(_ data: Data, buffer: OutputBuffer, process: Process) {
        let active = worker === process
        buffer.data.append(data)
        while let newline = buffer.data.firstIndex(of: 10) {
            let line = buffer.data.prefix(upTo: newline)
            buffer.data.removeSubrange(...newline)
            guard let event = try? JSONDecoder().decode(WorkerEvent.self, from: line) else { continue }
            let utterance = event.utterance
            switch event.kind {
            case "status":
                break
            case "ready":
                guard active else { break }
                if #available(macOS 26.0, *) {
                    prepareAnalyzer()
                }
            case "final":
                if let index = entries.lastIndex(where: { $0.utterance == utterance }) {
                    invalidateRendering(from: index)
                    if event.translation.isEmpty {
                        // 逐字稿定稿事件不帶譯文：文字沒變就保留已有的譯文，變了才清掉（修正後會重新翻譯）
                        if entries[index].text != event.text { entries[index].translation = nil }
                        entries[index].text = event.text
                    } else if entries[index].text == event.text {
                        entries[index].translation = event.translation
                    }
                    // 文字對不上的譯文是修正前那句的，丟掉
                    entries[index].finalized = true
                } else {
                    entries.append(TranscriptEntry(utterance: utterance, time: clock.string(from: Date()), text: event.text, translation: event.translation.isEmpty ? nil : event.translation, finalized: true, generation: workerGeneration))
                }
                // 這句翻成功了，之前單句失敗的提示就不用留著
                if active, !event.translation.isEmpty, !translateNotice.isEmpty {
                    translateNotice = ""
                    updateTranslationButton()
                }
                renderTranscript()
                refreshSavedFile()
                if running { setState(.listening) }
            case "warning":
                NSLog("worker warning: %@", event.message)
                guard active else { break }
                switch event.code {
                case "translator_load":
                    translatorFailed = true
                    workerNotice = event.message
                case "polish_load":
                    polishFailed = true
                    polishNotice = event.message
                case "translate_failed", "finalize_failed":
                    translateNotice = event.message
                default:
                    workerNotice = event.message
                }
                updateTranslationButton()
                updatePolishButton()
            case "error":
                guard active else { break }
                setState(.problem("處理時發生錯誤", event.message))
            default:
                break
            }
        }
    }

    // 畫面上 entry 佔的字元數（含換行）；renderTranscript 的 commit 迴圈與 invalidateRendering 共用同一個公式
    private func renderedLength(_ entry: TranscriptEntry) -> Int {
        let translation = entry.translation ?? ""
        return (entry.text as NSString).length + 1 + (translation.isEmpty ? 0 : (translation as NSString).length + 1)
    }

    // index 之前的句子畫面不變，只退回到 index；renderTranscript 會把 storage 從 committedLength 之後截掉重畫
    private func invalidateRendering(from index: Int) {
        if index < committedCount {
            committedCount = index
            committedLength = entries[..<index].reduce(0) { $0 + renderedLength($1) }
        }
    }

    // worker 結束後（正常收尾、被終止或當掉），該 worker 還沒送回定稿的句子不會再有了
    private func finalizeOrphans(generation: Int) {
        var changed = false
        for index in entries.indices where entries[index].generation == generation && !entries[index].finalized {
            entries[index].finalized = true
            changed = true
        }
        guard changed else { return }
        NSLog("worker ended with unfinalized entries (generation %d), marked finalized", generation)
        renderTranscript()
        refreshSavedFile()
    }

    private func cancelEngine() {
        if #available(macOS 26.0, *), let engine = engineBox as? SpeechEngine {
            engine.cancel()
        }
        engineBox = nil
    }

    @available(macOS 26.0, *)
    private func prepareAnalyzer() {
        guard starting, let sink else { return }
        let engine = SpeechEngine()
        engineBox = engine
        engine.contextualStrings = glossaryTerms()
        NSLog("glossary: %d contextual strings", engine.contextualStrings.count)
        utteranceBase = entries.map(\.utterance).max() ?? 0
        engine.onPartial = { [weak self] text in self?.analyzerPartial(text) }
        engine.onFinal = { [weak self, weak engine] id, text, language, start, end in
            self?.analyzerFinal(id: id, text: text, language: language, start: start, end: end, sink: sink, engine: engine)
        }
        engine.onRevise = { [weak self] id, text in self?.analyzerRevise(id: id, text: text, sink: sink) }
        engine.onError = { [weak self] message in
            NSLog("speech engine error: %@", message)
            self?.setState(.problem("語音辨識發生錯誤", message))
        }
        Task { @MainActor in
            do {
                try await engine.prepare()
                guard self.engineBox === engine else { return }
                self.beginCapture()
            } catch {
                guard self.engineBox === engine else { return }
                self.starting = false
                self.engineBox = nil
                self.setState(.problem("無法啟動語音辨識", error.localizedDescription))
            }
        }
    }

    private func analyzerPartial(_ text: String) {
        liveStable = String(zip(liveText, text).prefix { $0 == $1 }.map(\.0))
        liveText = text
        renderTranscript()
        if running { setState(.listening) }
    }

    // language 是 ja 或 en；沒給時 worker 自己從文字判斷。audio 是該句的 16 kHz float32 音訊，給 Qwen 重解
    private func sendToWorker(utterance: Int, text: String, language: String? = nil, audio: Data? = nil, sink: AudioSink) {
        var job: [String: Any] = ["utterance": utterance, "text": text]
        if let language { job["language"] = language }
        if let audio, !audio.isEmpty { job["audio"] = audio.base64EncodedString() }
        guard let line = try? JSONSerialization.data(withJSONObject: job) else { return }
        sink.write(line + Data([10]))
    }

    // 重解時前後多留一點音訊：SpeechAnalyzer 的時間戳常把句首、句尾的音切掉一些。
    // 留太多會吃進前後句：兩段真實會議實測 0.2/0.3 秒有 12 句混入下一句，0.1/0.1 秒 7 句。worker 的 REFINE_PADDING_SECONDS 要同步
    private static let refineLeadSeconds = 0.1
    private static let refineTailSeconds = 0.1

    @available(macOS 26.0, *)
    private func analyzerFinal(id: Int, text: String, language: String, start: Double, end: Double, sink: AudioSink, engine: SpeechEngine?) {
        let utterance = utteranceBase + id
        entries.append(TranscriptEntry(utterance: utterance, time: clock.string(from: Date()), text: text, language: language, generation: workerGeneration))
        renderTranscript()
        refreshSavedFile()
        var audio: Data?
        // 用傳進來的 engine 而不是 engineBox：收尾時 engineBox 已清掉，最後幾句才拿得到音訊
        if activeRefine, let engine {
            audio = engine.feeder.audio(from: start - Self.refineLeadSeconds, to: end + Self.refineTailSeconds)
        }
        sendToWorker(utterance: utterance, text: text, language: language, audio: audio, sink: sink)
    }

    // SpeechAnalyzer 事後修正已送出的句子。Qwen 重解開著時以重解結果為準：已定稿就不動，未定稿只更新畫面。
    // 重解沒開時改完重送 worker，worker 會重新刪贅詞並重新翻譯
    private func analyzerRevise(id: Int, text: String, sink: AudioSink) {
        let utterance = utteranceBase + id
        guard let index = entries.lastIndex(where: { $0.utterance == utterance }) else { return }
        if activeRefine && entries[index].finalized { return }
        invalidateRendering(from: index)
        entries[index].text = text
        renderTranscript()
        refreshSavedFile()
        if !activeRefine { sendToWorker(utterance: utterance, text: text, language: entries[index].language, sink: sink) }
    }

    private func beginCapture() {
        guard starting, sink != nil else { return }
        let recorder = recorder
        var onAudio: (Data) -> Void = { data in recorder.append(data) }
        if #available(macOS 26.0, *), let engine = engineBox as? SpeechEngine {
            let feeder = engine.feeder
            onAudio = { data in feeder.append(data); recorder.append(data) }
        }
        let capture = AudioCapture(onAudio: onAudio, onError: { [weak self] message in
            DispatchQueue.main.async { self?.setState(.problem("系統音訊中斷", message)) }
        })
        self.capture = capture
        Task { @MainActor in
            do {
                try await capture.start()
                guard self.starting else { await capture.stop(); return }
                self.starting = false
                self.running = true
                self.recorder.resume()
                self.setState(.listening)
            } catch {
                self.starting = false
                self.running = false
                self.capture = nil
                self.setState(.problem("需要「螢幕與系統音訊錄製」權限", "系統設定 → 隱私權與安全性 → 螢幕與系統音訊錄製，允許「即時翻譯」後重新開啟 App。\n\(error.localizedDescription)"))
            }
        }
    }

}

@main
enum LiveTranslatorApp {
    @MainActor static var controller: AppController?

    @MainActor static func main() {
        signal(SIGPIPE, SIG_IGN)
        let app = NSApplication.shared
        let controller = AppController()
        self.controller = controller
        app.delegate = controller
        app.setActivationPolicy(.regular)
        app.run()
    }
}

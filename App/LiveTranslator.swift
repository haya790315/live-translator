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

final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let queue = DispatchQueue(label: "LiveTranslator.Audio")
    private let resampler = AudioResampler()
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
        if buffers.count == 1, channels == 1, format.mSampleRate == 16000,
           let pointer = buffers[0].mData, buffers[0].mDataByteSize > 0 {
            onAudio(Data(bytes: pointer, count: Int(buffers[0].mDataByteSize)))
        } else if buffers.count == 1, let pointer = buffers[0].mData {
            let frames = Int(buffers[0].mDataByteSize) / (MemoryLayout<Float>.size * channels)
            let values = pointer.assumingMemoryBound(to: Float.self)
            var mono = [Float]()
            mono.reserveCapacity(frames)
            for frame in 0..<frames {
                var sum: Float = 0
                for channel in 0..<channels { sum += values[frame * channels + channel] }
                mono.append(sum / Float(channels))
            }
            let data = resampler.convert(mono, sampleRate: format.mSampleRate)
            if !data.isEmpty { onAudio(data) }
        } else if buffers.count >= channels {
            let frames = buffers.prefix(channels).map { Int($0.mDataByteSize) / MemoryLayout<Float>.size }.min() ?? 0
            var mono = [Float]()
            mono.reserveCapacity(frames)
            for frame in 0..<frames {
                var sum: Float = 0
                for channel in 0..<channels {
                    guard let pointer = buffers[channel].mData else { continue }
                    sum += pointer.assumingMemoryBound(to: Float.self)[frame]
                }
                mono.append(sum / Float(channels))
            }
            let data = resampler.convert(mono, sampleRate: format.mSampleRate)
            if !data.isEmpty { onAudio(data) }
        }
        withExtendedLifetime(block) {}
    }
}

struct WorkerEvent: Decodable {
    let kind: String
    let utterance: Int
    var japanese: String
    let chinese: String
    let message: String
    let stable: String?
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
    var japanese: String
    var chinese: String?
}

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
    private var saveButton: NSButton!
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
    private var pendingOutput = Data()
    private var workerError = ""
    private var capture: AudioCapture?
    private var starting = false
    private var running = false
    private var currentUtterance = 0
    private var entries: [TranscriptEntry] = []
    private var committedCount = 0
    private var committedLength = 0
    private var liveJapanese = ""
    private var liveStable = ""
    private let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
    private let japaneseAttributes: [NSAttributedString.Key: Any]
    private let japaneseAloneAttributes: [NSAttributedString.Key: Any]
    private let liveAttributes: [NSAttributedString.Key: Any]
    private let tentativeAttributes: [NSAttributedString.Key: Any]
    private let chineseAttributes: [NSAttributedString.Key: Any]

    override init() {
        let japaneseFont = NSFont.systemFont(ofSize: 14)
        let liveFont = NSFont.systemFont(ofSize: 16)
        let chineseFont = NSFont.systemFont(ofSize: 18, weight: .semibold)
        let chineseColor = NSColor(red: 1.0, green: 0.85, blue: 0.42, alpha: 1.0)
        let tightStyle = NSMutableParagraphStyle()
        tightStyle.paragraphSpacing = 2
        let looseStyle = NSMutableParagraphStyle()
        looseStyle.paragraphSpacing = 18
        japaneseAttributes = [.font: japaneseFont, .foregroundColor: NSColor.white.withAlphaComponent(0.62), .paragraphStyle: tightStyle]
        japaneseAloneAttributes = [.font: japaneseFont, .foregroundColor: NSColor.white.withAlphaComponent(0.62), .paragraphStyle: looseStyle]
        liveAttributes = [.font: liveFont, .foregroundColor: NSColor.white.withAlphaComponent(0.92), .paragraphStyle: looseStyle]
        tentativeAttributes = [.font: liveFont, .foregroundColor: NSColor.white.withAlphaComponent(0.45), .paragraphStyle: looseStyle]
        chineseAttributes = [.font: chineseFont, .foregroundColor: chineseColor, .paragraphStyle: looseStyle]
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installCrashLogging(root: projectRoot())
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
        autosaveIfNeeded()
        sink?.close()
        if worker?.isRunning == true { worker?.terminate() }
    }

    func windowWillClose(_ notification: Notification) {
        NSLog("windowWillClose")
        NSApp.terminate(nil)
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
        saveButton = makeIconButton(action: #selector(save))
        setSymbol(saveButton, "square.and.arrow.down", tip: "儲存逐字稿")
        saveButton.contentTintColor = NSColor(srgbRed: 1.0, green: 0.85, blue: 0.42, alpha: 1.0)
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
        nameField.toolTip = "點一下修改檔名"
        nameField.translatesAutoresizingMaskIntoConstraints = false
        suffixLabel = NSTextField(labelWithString: ".txt")
        suffixLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        suffixLabel.textColor = NSColor.white.withAlphaComponent(0.45)

        let indicator = NSView()
        indicator.translatesAutoresizingMaskIntoConstraints = false
        indicator.addSubview(statusDot)
        indicator.addSubview(spinner)
        let top = NSStackView(views: [indicator, statusLabel, nameField, suffixLabel, saveButton, endButton, toggleButton])
        top.orientation = .horizontal
        top.alignment = .centerY
        top.distribution = .fill
        top.spacing = 8
        top.setCustomSpacing(0, after: nameField)
        top.setCustomSpacing(12, after: suffixLabel)
        top.setCustomSpacing(2, after: saveButton)
        top.setCustomSpacing(2, after: endButton)
        statusLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        statusLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        toggleButton.setContentHuggingPriority(.required, for: .horizontal)
        endButton.setContentHuggingPriority(.required, for: .horizontal)
        saveButton.setContentHuggingPriority(.required, for: .horizontal)
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
            saveButton.widthAnchor.constraint(equalToConstant: 28),
            saveButton.heightAnchor.constraint(equalToConstant: 24),
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
        starting = false
        running = false
        let oldCapture = capture
        capture = nil
        let oldSink = sink
        sink = nil
        inputPipe = nil
        endingWorker = worker != nil
        Task { @MainActor in
            await oldCapture?.stop()
            oldSink?.close()
        }
        savedURL = nil
        nameField.stringValue = uniqueName(base: dateName())
        updateNameWidth()
        setState(.ended)
    }

    @objc private func save() {
        NSLog("action: save")
        panel.makeFirstResponder(nil)
        guard hasContent else {
            statusLabel.stringValue = "沒有可儲存的內容"
            return
        }
        var name = sanitizedName(nameField.stringValue)
        if name.isEmpty { name = dateName() }
        var url = transcriptsDirectory().appendingPathComponent("\(name).txt")
        if url != savedURL, FileManager.default.fileExists(atPath: url.path) {
            name = uniqueName(base: name)
            url = transcriptsDirectory().appendingPathComponent("\(name).txt")
        }
        nameField.stringValue = name
        updateNameWidth()
        writeTranscript(to: url, title: name)
        savedURL = url
        setState(.ended)
    }

    private var hasContent: Bool {
        !entries.isEmpty || !liveJapanese.isEmpty
    }

    private func pause() {
        NSLog("action: pause")
        guard starting || running else { return }
        starting = false
        running = false
        let oldCapture = capture
        capture = nil
        Task { @MainActor in await oldCapture?.stop() }
        setState(.paused)
    }

    private func resume() {
        NSLog("action: resume")
        guard worker?.isRunning == true, sink != nil else {
            start(fresh: false)
            return
        }
        starting = true
        setState(.preparing)
        beginCapture()
    }

    private func startNewSession() {
        if savedURL == nil && hasContent {
            let alert = NSAlert()
            alert.messageText = "這段紀錄尚未儲存"
            alert.informativeText = "開始新的一段會清掉目前的字幕。要放棄這段紀錄嗎？"
            alert.alertStyle = .warning
            alert.addButton(withTitle: "放棄並開始")
            alert.addButton(withTitle: "取消")
            guard alert.runModal() == .alertFirstButtonReturn else { return }
        }
        start(fresh: true)
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
        while FileManager.default.fileExists(atPath: transcriptsDirectory().appendingPathComponent("\(name).txt").path) {
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
        let name = sanitizedName(nameField.stringValue)
        nameField.stringValue = name.isEmpty ? nameBeforeEdit : name
        updateNameWidth()
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
        saveButton.isHidden = !ended
        endButton.isHidden = ended
        switch newState {
        case .preparing:
            spinner.startAnimation(nil)
            setDot(nil, pulse: false)
            statusLabel.stringValue = "準備中…"
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
            statusLabel.stringValue = savedURL == nil ? "已終止" : "已儲存"
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
            lines.append("[\(entry.time)] \(entry.japanese)")
            if let chinese = entry.chinese, !chinese.isEmpty {
                lines.append("           \(chinese)")
            }
        }
        if includeLive && !liveJapanese.isEmpty {
            lines.append("[\(clock.string(from: Date()))] \(liveJapanese)（未定稿）")
        }
        return lines
    }

    private func resetTranscript() {
        entries.removeAll()
        committedCount = 0
        committedLength = 0
        currentUtterance = 0
        liveJapanese = ""
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

    private func autosaveIfNeeded() {
        guard hasContent else { return }
        if let url = savedURL {
            writeTranscript(to: url, title: url.deletingPathExtension().lastPathComponent)
            return
        }
        let name = uniqueName(base: dateName())
        writeTranscript(to: transcriptsDirectory().appendingPathComponent("\(name).txt"), title: name)
    }

    private func renderTranscript() {
        guard let storage = transcriptView.textStorage else { return }
        let wasAtBottom = scrollView.documentVisibleRect.maxY >= transcriptView.bounds.height - 40
        storage.beginEditing()
        storage.deleteCharacters(in: NSRange(location: committedLength, length: storage.length - committedLength))
        for entry in entries[committedCount...] {
            if let chinese = entry.chinese, !chinese.isEmpty {
                storage.append(NSAttributedString(string: entry.japanese + "\n", attributes: japaneseAttributes))
                storage.append(NSAttributedString(string: chinese + "\n", attributes: chineseAttributes))
            } else {
                storage.append(NSAttributedString(string: entry.japanese + "\n", attributes: japaneseAloneAttributes))
            }
        }
        if !liveJapanese.isEmpty {
            let stable = liveJapanese.hasPrefix(liveStable) ? liveStable : ""
            storage.append(NSAttributedString(string: stable, attributes: liveAttributes))
            storage.append(NSAttributedString(string: String(liveJapanese.dropFirst(stable.count)) + "\n", attributes: tentativeAttributes))
        }
        storage.endEditing()
        emptyLabel.isHidden = storage.length > 0
        while committedCount < entries.count, let chinese = entries[committedCount].chinese {
            committedLength += (entries[committedCount].japanese as NSString).length + 1 + (chinese.isEmpty ? 0 : (chinese as NSString).length + 1)
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
        worker = nil
        inputPipe = nil
        outputPipe = nil
        errorPipe = nil
        endingWorker = false
        starting = true
        currentUtterance = 0
        if fresh {
            resetTranscript()
            savedURL = nil
        }
        setState(.preparing)

        let root = projectRoot()
        let python = root.appendingPathComponent(".venv/bin/python")
        let script = Bundle.main.resourceURL!.appendingPathComponent("worker.py")
        let speech = root.appendingPathComponent("Models/parakeet-tdt_ctc-0.6b-ja")
        var translationNames = ["Hy-MT2-1.8B-8bit", "Hunyuan-MT-7B-4bit", "Hy-MT2-1.8B-4bit", "Qwen3-4B-4bit"]
        if let chosen = try? String(contentsOf: root.appendingPathComponent("Models/translation-model.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !chosen.isEmpty {
            translationNames.insert(chosen, at: 0)
        }
        let translation = translationNames.map { root.appendingPathComponent("Models/\($0)") }
            .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.safetensors").path) }
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: script.path),
              FileManager.default.fileExists(atPath: speech.appendingPathComponent("model.safetensors").path),
              let translation else {
            starting = false
            setState(.problem("尚未完成安裝", "請先在專案目錄執行 Scripts/setup.sh"))
            return
        }

        let process = Process()
        process.executableURL = python
        let english = root.appendingPathComponent("Models/parakeet-tdt-0.6b-v3")
        var arguments = [script.path, speech.path, translation.path]
        if FileManager.default.fileExists(atPath: english.appendingPathComponent("model.safetensors").path) {
            arguments.append(english.path)
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
        pendingOutput.removeAll()
        workerError = ""

        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async { self?.receive(data) }
        }
        errors.fileHandleForReading.readabilityHandler = { [weak self] handle in
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
                guard let self, self.worker === process else { return }
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

    private func receive(_ data: Data) {
        pendingOutput.append(data)
        while let newline = pendingOutput.firstIndex(of: 10) {
            let line = pendingOutput.prefix(upTo: newline)
            pendingOutput.removeSubrange(...newline)
            guard let event = try? JSONDecoder().decode(WorkerEvent.self, from: line) else { continue }
            switch event.kind {
            case "status":
                break
            case "ready":
                beginCapture()
            case "partial_japanese":
                guard event.utterance >= currentUtterance else { break }
                currentUtterance = event.utterance
                liveJapanese = event.japanese
                liveStable = event.stable ?? ""
                renderTranscript()
                if running { setState(.listening) }
            case "final_japanese":
                entries.append(TranscriptEntry(utterance: event.utterance, time: clock.string(from: Date()), japanese: event.japanese, chinese: nil))
                if event.utterance >= currentUtterance {
                    currentUtterance = event.utterance + 1
                    liveJapanese = ""
                    liveStable = ""
                }
                renderTranscript()
                refreshSavedFile()
            case "revised_japanese":
                if let index = entries.lastIndex(where: { $0.utterance == event.utterance }) {
                    entries[index].japanese = event.japanese
                    renderTranscript()
                    refreshSavedFile()
                }
            case "final":
                if let index = entries.lastIndex(where: { $0.utterance == event.utterance }) {
                    entries[index].japanese = event.japanese
                    entries[index].chinese = event.chinese
                } else {
                    entries.append(TranscriptEntry(utterance: event.utterance, time: clock.string(from: Date()), japanese: event.japanese, chinese: event.chinese))
                }
                renderTranscript()
                refreshSavedFile()
                if running { setState(.listening) }
            case "error":
                setState(.problem("處理時發生錯誤", event.message))
            default:
                break
            }
        }
    }

    private func beginCapture() {
        guard starting, let sink else { return }
        let capture = AudioCapture(onAudio: { data in
            sink.write(data)
        }, onError: { [weak self] message in
            DispatchQueue.main.async { self?.setState(.problem("系統音訊中斷", message)) }
        })
        self.capture = capture
        Task { @MainActor in
            do {
                try await capture.start()
                guard self.starting else { await capture.stop(); return }
                self.starting = false
                self.running = true
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

signal(SIGPIPE, SIG_IGN)
let app = NSApplication.shared
let controller = AppController()
app.delegate = controller
app.setActivationPolicy(.regular)
app.run()

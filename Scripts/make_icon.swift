import AppKit

let arguments = CommandLine.arguments
guard arguments.count == 2 else {
    FileHandle.standardError.write("usage: swift make_icon.swift <output.iconset>\n".data(using: .utf8)!)
    exit(2)
}
let output = URL(fileURLWithPath: arguments[1])
try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> NSColor {
    NSColor(
        srgbRed: CGFloat((hex >> 16) & 0xff) / 255,
        green: CGFloat((hex >> 8) & 0xff) / 255,
        blue: CGFloat(hex & 0xff) / 255,
        alpha: alpha
    )
}

func bubble(_ rect: NSRect, radius: CGFloat, tailAt tail: NSPoint, tailBase: (CGFloat, CGFloat)) -> NSBezierPath {
    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    let tailPath = NSBezierPath()
    tailPath.move(to: NSPoint(x: tailBase.0, y: rect.minY + 1))
    tailPath.line(to: tail)
    tailPath.line(to: NSPoint(x: tailBase.1, y: rect.minY + 1))
    tailPath.close()
    path.append(tailPath)
    return path
}

func drawGlyph(_ glyph: String, font: NSFont, color: NSColor, center: NSPoint) {
    let text = NSAttributedString(string: glyph, attributes: [.font: font, .foregroundColor: color])
    let bounds = text.boundingRect(with: NSSize(width: 4000, height: 4000), options: [.usesLineFragmentOrigin, .usesFontLeading])
    let line = CTLineCreateWithAttributedString(text)
    let ink = CTLineGetImageBounds(line, nil)
    let origin = NSPoint(x: center.x - ink.midX, y: center.y - ink.midY)
    _ = bounds
    guard let context = NSGraphicsContext.current?.cgContext else { return }
    context.textPosition = origin
    CTLineDraw(line, context)
}

func font(_ names: [String], size: CGFloat) -> NSFont {
    for name in names {
        if let found = NSFont(name: name, size: size) { return found }
    }
    return NSFont.systemFont(ofSize: size, weight: .bold)
}

func render(pixels: Int) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    )!
    rep.size = NSSize(width: 1024, height: 1024)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSGraphicsContext.current?.imageInterpolation = .high

    let body = NSRect(x: 100, y: 100, width: 824, height: 824)
    let shape = NSBezierPath(roundedRect: body, xRadius: 185, yRadius: 185)

    NSGraphicsContext.saveGraphicsState()
    let shadow = NSShadow()
    shadow.shadowColor = NSColor.black.withAlphaComponent(0.35)
    shadow.shadowOffset = NSSize(width: 0, height: -10)
    shadow.shadowBlurRadius = 24
    shadow.set()
    color(0x141B33).setFill()
    shape.fill()
    NSGraphicsContext.restoreGraphicsState()

    NSGraphicsContext.saveGraphicsState()
    shape.addClip()
    NSGradient(colors: [color(0x2A3766), color(0x0F1428)])!.draw(in: body, angle: -70)
    NSGradient(colors: [NSColor.white.withAlphaComponent(0.10), NSColor.white.withAlphaComponent(0)])!
        .draw(in: NSRect(x: body.minX, y: body.midY, width: body.width, height: body.height / 2), angle: -90)
    NSGraphicsContext.restoreGraphicsState()

    let back = NSRect(x: 190, y: 470, width: 420, height: 330)
    let backPath = bubble(back, radius: 92, tailAt: NSPoint(x: 250, y: 385), tailBase: (262, 360))
    NSGraphicsContext.saveGraphicsState()
    let backShadow = NSShadow()
    backShadow.shadowColor = NSColor.black.withAlphaComponent(0.25)
    backShadow.shadowOffset = NSSize(width: 0, height: -8)
    backShadow.shadowBlurRadius = 18
    backShadow.set()
    color(0xF2F4FA).setFill()
    backPath.fill()
    NSGraphicsContext.restoreGraphicsState()
    drawGlyph("あ", font: font(["HiraginoSans-W6", "HiraKakuProN-W6"], size: 230), color: color(0x2A3766), center: NSPoint(x: back.midX, y: back.midY + 6))

    let front = NSRect(x: 400, y: 215, width: 440, height: 345)
    let frontPath = bubble(front, radius: 96, tailAt: NSPoint(x: 770, y: 130), tailBase: (680, 790))
    NSGraphicsContext.saveGraphicsState()
    let frontShadow = NSShadow()
    frontShadow.shadowColor = NSColor.black.withAlphaComponent(0.4)
    frontShadow.shadowOffset = NSSize(width: 0, height: -12)
    frontShadow.shadowBlurRadius = 26
    frontShadow.set()
    NSGradient(colors: [color(0xFFE38A), color(0xFFC53D)])!.draw(in: frontPath, angle: -90)
    NSGraphicsContext.restoreGraphicsState()
    NSGradient(colors: [color(0xFFE38A), color(0xFFC53D)])!.draw(in: frontPath, angle: -90)
    drawGlyph("中", font: font(["PingFangSC-Semibold", "HiraginoSans-W6"], size: 240), color: color(0x141B33), center: NSPoint(x: front.midX, y: front.midY + 4))

    NSGraphicsContext.restoreGraphicsState()
    return rep
}

let sizes: [(String, Int)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024),
]
for (name, pixels) in sizes {
    let data = render(pixels: pixels).representation(using: .png, properties: [:])!
    try data.write(to: output.appendingPathComponent("\(name).png"))
}
print("wrote \(sizes.count) images to \(output.path)")

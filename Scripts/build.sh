#!/bin/zsh
set -euo pipefail

cd "${0:A:h:h}"
staging="Build/LiveTranslator.app.building"
rm -rf "$staging"
mkdir -p "$staging/Contents/MacOS" "$staging/Contents/Resources"
swiftc -O -parse-as-library -swift-version 5 -target arm64-apple-macos14.0 App/LiveTranslator.swift App/SpeechEngine.swift -o "$staging/Contents/MacOS/LiveTranslator"
cp App/Info.plist "$staging/Contents/Info.plist"
cp Inference/worker.py "$staging/Contents/Resources/worker.py"
cp App/AppIcon.icns "$staging/Contents/Resources/AppIcon.icns"
identity="-"
if security find-identity -v -p codesigning 2>/dev/null | grep -q '"LiveTranslator Dev"'; then identity="LiveTranslator Dev"; fi
codesign --force --deep --timestamp=none --sign "$identity" "$staging"
rm -rf Build/LiveTranslator.app.previous
[ -d Build/LiveTranslator.app ] && mv Build/LiveTranslator.app Build/LiveTranslator.app.previous
mv "$staging" Build/LiveTranslator.app
rm -rf Build/LiveTranslator.app.previous
if [ "$identity" = "-" ]; then
    print "已建立 Build/LiveTranslator.app（臨時簽章；執行 zsh Scripts/make_signing_identity.sh 可固定簽章，重編後不必再授權）"
else
    print "已建立 Build/LiveTranslator.app（以「$identity」簽章）"
fi

#!/bin/zsh
set -uo pipefail

name="LiveTranslator Dev"
keychain="$(security login-keychain | tr -d '[:space:]"')"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

has_valid_identity() {
    security find-identity -v -p codesigning "$keychain" 2>/dev/null | grep -q "\"$name\""
}

case "${1:-install}" in
install)
    if has_valid_identity; then
        print "簽章身分「$name」已存在且可用，不需重做。"
        exit 0
    fi
    if ! security find-certificate -c "$name" "$keychain" >/dev/null 2>&1; then
        cat > "$work/cert.cnf" <<EOF
[req]
distinguished_name = dn
prompt = no
[dn]
CN = $name
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
subjectKeyIdentifier = hash
EOF
        openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$work/cert.cnf" -extensions ext \
            -keyout "$work/key.pem" -out "$work/cert.pem" 2>/dev/null || { print "產生憑證失敗" >&2; exit 1 }
        openssl pkcs12 -export -inkey "$work/key.pem" -in "$work/cert.pem" -name "$name" \
            -passout pass:temp -out "$work/identity.p12" || { print "打包憑證失敗" >&2; exit 1 }
        security import "$work/identity.p12" -k "$keychain" -P temp -T /usr/bin/codesign -T /usr/bin/security >/dev/null \
            || { print "匯入登入鑰匙圈失敗" >&2; exit 1 }
        print "已把憑證與私鑰放進登入鑰匙圈（有效期 10 年）。"
    fi
    security find-certificate -c "$name" -p "$keychain" > "$work/trust.pem"
    print "接下來 macOS 會跳出視窗要求輸入 Mac 登入密碼，用來把這張憑證設為「程式碼簽章可信任」。"
    security add-trusted-cert -r trustRoot -p basic -p codeSign -k "$keychain" "$work/trust.pem"
    if has_valid_identity; then
        print "完成。之後 Scripts/build.sh 會自動用「$name」簽章。"
        print "第一次簽章時若跳出「codesign 想要使用鑰匙圈中的金鑰」，請按「永遠允許」。"
    else
        print "憑證已在鑰匙圈，但尚未被信任，build.sh 仍會用臨時簽章；再執行一次本指令即可補做信任步驟。" >&2
        exit 1
    fi
    ;;
remove)
    if ! security find-certificate -c "$name" -p "$keychain" > "$work/trust.pem" 2>/dev/null; then
        print "鑰匙圈裡沒有「$name」。"
        exit 0
    fi
    security remove-trusted-cert "$work/trust.pem" 2>/dev/null || true
    security delete-identity -c "$name" "$keychain" >/dev/null && print "已移除「$name」的憑證與私鑰。"
    ;;
*)
    print "用法：zsh Scripts/make_signing_identity.sh [install|remove]" >&2
    exit 2
    ;;
esac

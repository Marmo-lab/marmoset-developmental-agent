#!/bin/bash
# Build MarmoLabel's APK with the bare Android SDK (no Gradle): aapt2 -> javac -> d8 -> zipalign -> apksigner.
# Needs JDK 17 and the SDK's platforms;android-34 and build-tools;34.0.0 (brew install --cask android-commandlinetools,
# then sdkmanager). Output: build/marmolabel.apk, which tools/labels/phone_server.py serves at /app.apk.
# Signed with the standard debug key (~/.android/debug.keystore, created if missing); keep using the same key,
# or the phone refuses to install an update over the previous version.
set -euo pipefail
cd "$(dirname "$0")"
SDK="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
BT="$SDK/build-tools/34.0.0"
JAR="$SDK/platforms/android-34/android.jar"
export JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 17)}"
export PATH="$JAVA_HOME/bin:$PATH"
VERSION_CODE="$(date +%y%m%d%H%M | cut -c1-9)"
KS="$HOME/.android/debug.keystore"

rm -rf build && mkdir -p build/classes build/dex build/gen
"$BT/aapt2" compile --dir res -o build/res.zip
"$BT/aapt2" link -I "$JAR" --manifest AndroidManifest.xml -o build/unsigned.apk build/res.zip --java build/gen \
    --min-sdk-version 29 --target-sdk-version 34 --version-code "$VERSION_CODE" --version-name "1.0.$VERSION_CODE"
javac -source 8 -target 8 -Xlint:-options -encoding UTF-8 -classpath "$JAR" -d build/classes \
    $(find src build/gen -name '*.java')
"$BT/d8" --min-api 29 --lib "$JAR" --output build/dex $(find build/classes -name '*.class')
cp build/unsigned.apk build/app.apk
(cd build/dex && zip -q ../app.apk classes.dex)
"$BT/zipalign" -f 4 build/app.apk build/aligned.apk
if [ ! -f "$KS" ]; then
    mkdir -p "$(dirname "$KS")"
    keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias androiddebugkey \
        -dname "CN=Android Debug,O=Android,C=US" -keyalg RSA -keysize 2048 -validity 10000 >/dev/null
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out build/marmolabel.apk build/aligned.apk
rm -f build/app.apk build/aligned.apk build/unsigned.apk build/res.zip
echo "built $(pwd)/build/marmolabel.apk ($(du -h build/marmolabel.apk | cut -f1), version $VERSION_CODE)"

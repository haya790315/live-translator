"""檢查錄下的會議音訊品質：音量、頻寬、削波、斷音。

用法（在專案目錄）：
    .venv/bin/python Scripts/audio_qc.py                      # 所有 Transcripts/*/*.wav
    .venv/bin/python Scripts/audio_qc.py Transcripts/2026-10-05/2026-10-05.wav

怎麼看：
- 有聲 RMS 中位：App 已做自動增益，正常應落在 -23 dBFS 附近；舊錄音（增益前）多在 -22 到 -34。
- 頻寬：16 kHz 語音正常 6 kHz 以上占 0.3～1%。只剩 0.01～0.04% 表示會議軟體送來的就是窄頻音訊，
  さ行、し、つ 這類擦音的資訊已經不在，任何模型都救不回來；要從對方的麥克風與會議設定下手。
- 全零斷音：數十段以上、長度 20～200 ms 且週期出現，是系統負載（swap、會議中切模型重啟）造成的掉幀；
  那幾句一定會錯。對策：Apple 模式、關掉瀏覽器、不要在會議中切模型。
"""
import glob
import sys
import wave

import numpy as np

VOICED_DB = -45


def analyze(path):
    with wave.open(path) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(1)
    if len(samples) < rate:
        print(f"{path}\n  不到 1 秒，略過")
        return
    duration = len(samples) / rate
    rms = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-9)
    peak = 20 * np.log10(np.abs(samples).max() + 1e-9)
    clipped = np.mean(np.abs(samples) >= 0.999) * 100

    # 30 ms 框的 RMS；高於 VOICED_DB 的框視為有聲
    frame = int(rate * 0.03)
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, 1)) + 1e-9)
    voiced = frame_db[frame_db > VOICED_DB]
    if len(voiced) == 0:
        print(f"{path}\n  {duration / 60:.1f} 分鐘，整段無聲")
        return

    # 全零斷音：連續全零 >= 20 ms（排除開頭結尾各 1 秒）
    zero = (samples == 0).astype(np.int8)
    edges = np.diff(np.concatenate([[0], zero, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    gaps = [(start / rate, (end - start) / rate) for start, end in zip(starts, ends)
            if end - start >= rate * 0.02 and start > rate and end < len(samples) - rate]

    # 長期平均頻譜（有聲框）：能量累積到 95%、99% 的頻率，以及 4 kHz、6 kHz 以上的占比
    window = np.hanning(frame)
    spectrum = np.mean(np.abs(np.fft.rfft(frames[frame_db > VOICED_DB] * window, axis=1)) ** 2, 0)
    freqs = np.fft.rfftfreq(frame, 1 / rate)
    cumulative = np.cumsum(spectrum) / spectrum.sum()
    f95 = freqs[np.searchsorted(cumulative, 0.95)]
    f99 = freqs[np.searchsorted(cumulative, 0.99)]
    above4k = spectrum[freqs >= 4000].sum() / spectrum.sum() * 100
    above6k = spectrum[freqs >= 6000].sum() / spectrum.sum() * 100

    print(f"{path}")
    print(f"  {duration / 60:.1f} 分鐘  {rate} Hz  整體 RMS {rms:.1f} dBFS  峰值 {peak:.1f} dBFS  削波 {clipped:.3f}%")
    print(f"  有聲框 {len(voiced) / len(frame_db) * 100:.0f}%  有聲 RMS 中位 {np.median(voiced):.1f} dBFS"
          f"  p10 {np.percentile(voiced, 10):.1f}  p90 {np.percentile(voiced, 90):.1f}  底噪 {np.percentile(frame_db, 10):.1f} dBFS")
    print(f"  頻寬：95% 能量 < {f95:.0f} Hz  99% 能量 < {f99:.0f} Hz  >4 kHz 占 {above4k:.2f}%  >6 kHz 占 {above6k:.3f}%")
    preview = "、".join(f"{at:.1f}s/{length * 1000:.0f}ms" for at, length in gaps[:5])
    print(f"  全零斷音 >= 20 ms：{len(gaps)} 段" + (f"（前 5：{preview}）" if gaps else ""))


def main():
    paths = sys.argv[1:] or sorted(glob.glob("Transcripts/*/*.wav"))
    if not paths:
        print("找不到錄音：請給 .wav 路徑，或在專案目錄執行")
        return 1
    for path in paths:
        analyze(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

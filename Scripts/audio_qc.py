"""檢查錄下的會議音訊品質：音量、頻寬、削波、斷音。

用法（在專案目錄）：
    .venv/bin/python Scripts/audio_qc.py                      # 所有 Transcripts/*/*.wav
    .venv/bin/python Scripts/audio_qc.py Transcripts/2026-10-05/2026-10-05.wav

怎麼看：
- 有聲 RMS 中位：App 已做自動增益，正常應落在 -23 dBFS 附近；舊錄音（增益前）多在 -22 到 -34。
- 頻譜：各頻段相對 300～1000 Hz 的平均能量。正常寬頻語音每翻一倍頻率掉 6～10 dB，4～5 kHz 約 −20～−25、
  6～7 kHz 約 −28～−35；Zoom、Teams、Meet 的寬頻 codec 在 7 kHz 左右截止，之後掉到 −50 以下是正常的。
  辨識模型只吃 16 kHz 音訊、最多用到 8 kHz，所以截止在 7 kHz 以上就沒有頻寬損失。
  截止頻率落在 4 kHz 以下才是窄頻（電話等級，擦音 さ行、し、つ 會聽不清楚）；4～5 kHz 比上面低 10 dB 以上
  則多半是某個講者的麥克風收音悶。不要用「累積 99% 能量的頻率」判斷：語音能量本來就集中在 1 kHz 以下。
- 全零斷音：數十段以上、長度 20～200 ms 且週期出現，是系統負載（swap、會議中切模型重啟）造成的掉幀；
  那幾句一定會錯。對策：Apple 模式、關掉瀏覽器、不要在會議中切模型。
"""
import glob
import sys
import wave

import numpy as np

VOICED_DB = -45
BANDS = [(1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 7800)]
CUTOFF_DB = -45


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

    # 長期平均頻譜（有聲框）：各頻段相對 300～1000 Hz 的平均能量，以及第一個掉到 CUTOFF_DB 以下的頻段
    window = np.hanning(frame)
    spectrum = np.mean(np.abs(np.fft.rfft(frames[frame_db > VOICED_DB] * window, axis=1)) ** 2, 0)
    freqs = np.fft.rfftfreq(frame, 1 / rate)
    reference = spectrum[(freqs >= 300) & (freqs < 1000)].mean()
    levels = [10 * np.log10(spectrum[(freqs >= low) & (freqs < high)].mean() / reference + 1e-12) for low, high in BANDS]
    cutoff = next((low for (low, _), level in zip(BANDS, levels) if level < CUTOFF_DB), None)

    print(f"{path}")
    print(f"  {duration / 60:.1f} 分鐘  {rate} Hz  整體 RMS {rms:.1f} dBFS  峰值 {peak:.1f} dBFS  削波 {clipped:.3f}%")
    print(f"  有聲框 {len(voiced) / len(frame_db) * 100:.0f}%  有聲 RMS 中位 {np.median(voiced):.1f} dBFS"
          f"  p10 {np.percentile(voiced, 10):.1f}  p90 {np.percentile(voiced, 90):.1f}  底噪 {np.percentile(frame_db, 10):.1f} dBFS")
    print("  頻譜（相對 300～1000 Hz，dB）：" + "  ".join(f"{low / 1000:g}-{high / 1000:g}k {level:.0f}" for (low, high), level in zip(BANDS, levels)))
    if cutoff is None:
        verdict = "沒有截止，寬頻"
    elif cutoff >= 6000:
        verdict = f"約 {cutoff / 1000:g} kHz 截止，寬頻，模型沒有頻寬損失"
    elif cutoff >= 4000:
        verdict = f"約 {cutoff / 1000:g} kHz 截止，頻寬略窄"
    else:
        verdict = f"約 {cutoff / 1000:g} kHz 截止，窄頻（電話等級），擦音會聽不清楚"
    print(f"  頻寬判讀：{verdict}")
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

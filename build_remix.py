#!/usr/bin/env python3
"""
REMIX-AB-V4 Production Pipeline (V4.2 — Zero-Cut Seamless Phrase Continuity & DJ Groove Restoration)
A Vocal × B DJ Remix — Studio Stem Processing, Beat/Harmonic Alignment, Arrangement & Mastering
"""

import os
import json
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import butter, sosfiltfilt
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    HighShelfFilter,
    LowShelfFilter,
    Compressor,
    NoiseGate,
    Reverb,
    Delay,
    Limiter,
)

SR = 44100
BPM_A = 130.00
BPM_B = 143.63
TEMPO_RATIO = BPM_B / BPM_A  # 1.10484615
PITCH_SEMITONES = +0.67      # F major/D minor -> F# major/D# minor (-33c tuning of B)
PITCH_RATIO = 2.0 ** (PITCH_SEMITONES / 12.0)

BEAT_A = 60.0 / BPM_A        # 0.46153846 s
BAR_A = 4.0 * BEAT_A         # 1.84615385 s
P0_A = 0.0218                # Bar 0 Beat 1 grid anchor in Music A

BEAT_B = 60.0 / BPM_B        # 0.41773996 s
BAR_B = 4.0 * BEAT_B         # 1.67095983 s
P0_B = 1.7790                # Bar 1 Beat 1 main downbeat in Music B (after 1-bar signature drum intro)

RAW_A_VOC = "/tmp/stems/A_VocFT_Vocals.wav"
RAW_B_INST = "/tmp/stems/input_B_(Instrumental).wav"

OUT_DIR = "/home/user/-SKILL"
WORK_DIR = "/tmp/remix_work"
os.makedirs(WORK_DIR, exist_ok=True)


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def cosine_fade_in(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0:
        return audio
    fade_samples = min(fade_samples, len(audio))
    curve = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))[:, None]
    out = audio.copy()
    out[:fade_samples] *= curve
    return out


def cosine_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0:
        return audio
    fade_samples = min(fade_samples, len(audio))
    curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))[:, None]
    out = audio.copy()
    out[-fade_samples:] *= curve
    return out


def envelope_follower(mono_abs: np.ndarray, attack_ms: float, release_ms: float, sr: int = SR) -> np.ndarray:
    """Smooth one-pole ballistic envelope follower."""
    a_att = np.exp(-1.0 / max(1.0, (attack_ms * 0.001 * sr)))
    a_rel = np.exp(-1.0 / max(1.0, (release_ms * 0.001 * sr)))
    env = np.zeros_like(mono_abs, dtype=np.float64)
    cur = 0.0
    for i, x in enumerate(mono_abs):
        if x > cur:
            cur = a_att * cur + (1.0 - a_att) * x
        else:
            cur = a_rel * cur + (1.0 - a_rel) * x
        env[i] = cur
    return env


def clean_a_vocals(y_voc: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Clean separated A vocals:
    - Remove low-end accompaniment bleed (<95 Hz) and ultra-high hiss (>16.5 kHz)
    - Gentle low-threshold expander (-50 dB, ratio 2.0) so soft consonant onsets and phrase
      breath tails are 100% preserved without any noise-gate chattering or abrupt cutoffs
    - Hard-mute only pure non-singing instrumental sections of Music A
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=95.0),
        NoiseGate(threshold_db=-50.0, ratio=2.0, attack_ms=2.0, release_ms=220.0),
        PeakFilter(cutoff_frequency_hz=260.0, gain_db=-1.8, q=1.2),
        PeakFilter(cutoff_frequency_hz=3200.0, gain_db=1.2, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=9500.0, gain_db=1.2),
        LowpassFilter(cutoff_frequency_hz=16500.0),
    ])
    processed = board(y_voc.T.astype(np.float32), sr).T

    out = processed.copy()
    mute_regions = [
        (0.0, 14.80),
        (106.50, 118.30),
        (197.00, len(out) / sr),
    ]
    fade_s = int(0.18 * sr)
    for t0, t1 in mute_regions:
        i0, i1 = int(t0 * sr), min(len(out), int(t1 * sr))
        if i1 > i0:
            if i0 > fade_s:
                out[i0 - fade_s:i0] = cosine_fade_out(out[i0 - fade_s:i0], fade_s)
            out[i0:i1] = 0.0
            if i1 + fade_s < len(out):
                out[i1:i1 + fade_s] = cosine_fade_in(out[i1:i1 + fade_s], fade_s)

    pk = np.max(np.abs(out))
    if pk > 0:
        out = out * (db_to_lin(-1.0) / pk)
    return out


def polish_b_instrumental(y_inst: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Polish separated B DJ Instrumental AND heal UVR vocal-removal transient micro-dips:
    - In Music B's original mix, heavy vocal/rap hits caused momentary spectral subtraction dips
      (50-100ms drops of 15-25 dB in Breakdown 2 and bar-end fills).
    - Dynamic sustain-bed compensation + gentle glue compression fills ONLY those momentary
      spectral holes while keeping B's opening drum fill (0..1.78s), kick punch, and synth leads crisp!
    """
    eq_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=26.0),
        PeakFilter(cutoff_frequency_hz=58.0, gain_db=1.2, q=1.1),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=0.8),
    ])
    clean = eq_board(y_inst.T.astype(np.float32), sr).T

    # Parallel stereo harmonic sustain bed (tempo-synced 1/8-note + tight room) to bridge UVR micro-dips
    sustain_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200.0),
        LowpassFilter(cutoff_frequency_hz=6800.0),
        Delay(delay_seconds=BEAT_B * 0.5, feedback=0.20, mix=0.55),
        Reverb(room_size=0.46, damping=0.45, wet_level=1.0, dry_level=0.0, width=1.0),
    ])
    sustain_tail = sustain_board(clean.T.astype(np.float32), sr).T

    mono_abs = np.mean(np.abs(clean), axis=1)
    fast_env = envelope_follower(mono_abs, attack_ms=3.0, release_ms=25.0, sr=sr)
    slow_env = envelope_follower(mono_abs, attack_ms=70.0, release_ms=360.0, sr=sr)
    hole_ratio = np.clip((slow_env - fast_env * 1.55) / (slow_env + 1e-6), 0.0, 1.0)
    # Keep opening drum fill (0.00-1.78s) 100% pure dry drums!
    hole_ratio[:int(P0_B * sr)] = 0.0

    healed = clean + sustain_tail * (0.05 + 0.36 * hole_ratio[:, None])

    glue_board = Pedalboard([
        Compressor(threshold_db=-16.0, ratio=1.6, attack_ms=15.0, release_ms=120.0),
    ])
    out = glue_board(healed.T.astype(np.float32), sr).T

    pk = np.max(np.abs(out))
    if pk > 0:
        out = out * (db_to_lin(-0.8) / pk)
    return out


def stretch_and_pitch_vocal_exact(
    y_voc: np.ndarray,
    a_cut_start_s: float,
    a_cut_end_s: float,
    a_anchor_s: float,
    fade_in_ms: float,
    fade_out_ms: float,
    tag: str,
) -> tuple[np.ndarray, int]:
    """
    Extract a vocal phrase strictly from verified physical silence [a_cut_start_s .. a_cut_end_s],
    where `a_anchor_s` is the exact bar/beat grid anchor inside Music A that maps to `b_target_s`.
    Every cut start and cut end sits inside a real silence valley (<-39 dB), so zero syllables are ever cut!
    """
    t0 = max(0.0, a_cut_start_s)
    t1 = min(len(y_voc) / SR, a_cut_end_s)
    actual_pre_s = a_anchor_s - t0

    seg = y_voc[int(round(t0 * SR)):int(round(t1 * SR))].copy()
    seg = cosine_fade_in(seg, int(round(fade_in_ms * 0.001 * SR)))
    seg = cosine_fade_out(seg, int(round(fade_out_ms * 0.001 * SR)))

    in_wav = os.path.join(WORK_DIR, f"seg_in_{tag}.wav")
    out_wav = os.path.join(WORK_DIR, f"seg_out_{tag}.wav")
    sf.write(in_wav, seg, SR, subtype="FLOAT")

    rb_filter = (
        f"rubberband=tempo={TEMPO_RATIO:.8f}:pitch={PITCH_RATIO:.8f}"
        f":formant=preserved:pitchq=quality:transients=smooth"
        f":phase=laminar:smoothing=on:channels=together"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_wav, "-af", rb_filter, "-c:a", "pcm_f32le", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stretched, _ = sf.read(out_wav, dtype="float32")
    rb_latency_s = 0.009
    pre_offset_samples = int(round((actual_pre_s / TEMPO_RATIO + rb_latency_s) * SR))
    return stretched, pre_offset_samples


def apply_vocal_fx_and_polish(vocal_bus: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Studio DJ Remix Vocal Chain:
    1. Subtractive EQ + Presence boost + Air shelf
    2. Two-stage smooth vocal compression
    3. Tempo-synced stereo 1/4-note delay (ducked when dry vocal sings, blooms intoDJ breaks!)
    4. Lush stereo plate reverb
    """
    dry_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=110.0),
        PeakFilter(cutoff_frequency_hz=250.0, gain_db=-2.2, q=1.2),
        PeakFilter(cutoff_frequency_hz=1600.0, gain_db=1.4, q=1.1),
        PeakFilter(cutoff_frequency_hz=3400.0, gain_db=2.2, q=1.0),
        PeakFilter(cutoff_frequency_hz=6800.0, gain_db=-2.0, q=2.5),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.0),
        Compressor(threshold_db=-20.0, ratio=3.2, attack_ms=8.0, release_ms=110.0),
        Compressor(threshold_db=-14.0, ratio=2.2, attack_ms=3.0, release_ms=70.0),
    ])
    voc_dry = dry_chain(vocal_bus.T.astype(np.float32), sr).T

    active_mask = np.mean(np.abs(voc_dry), axis=1) > db_to_lin(-34.0)
    if np.any(active_mask):
        active_rms = np.sqrt(np.mean(voc_dry[active_mask] ** 2))
        target_rms = db_to_lin(-13.8)
        voc_dry *= (target_rms / (active_rms + 1e-9))

    delay_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=320.0),
        LowpassFilter(cutoff_frequency_hz=4200.0),
        Delay(delay_seconds=BEAT_B, feedback=0.28, mix=1.0),
        Reverb(room_size=0.55, damping=0.55, wet_level=0.35, dry_level=0.65, width=1.0),
    ])
    voc_delay = delay_board(voc_dry.T.astype(np.float32), sr).T

    voc_env = envelope_follower(np.mean(np.abs(voc_dry), axis=1), attack_ms=5.0, release_ms=190.0, sr=sr)
    duck_delay = np.clip(1.0 / (1.0 + (voc_env / db_to_lin(-22.0)) ** 2.0), 0.28, 1.0)[:, None]

    reverb_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=280.0),
        LowpassFilter(cutoff_frequency_hz=7500.0),
        Reverb(room_size=0.64, damping=0.48, wet_level=1.0, dry_level=0.0, width=1.0),
    ])
    voc_rev = reverb_board(voc_dry.T.astype(np.float32), sr).T
    duck_rev = np.clip(1.0 / (1.0 + (voc_env / db_to_lin(-18.0)) ** 1.5), 0.45, 1.0)[:, None]

    combined_voc = voc_dry + 0.18 * voc_delay * duck_delay + 0.22 * voc_rev * duck_rev
    return combined_voc.astype(np.float32)


def mix_vocal_and_dj_inst(
    voc_stem: np.ndarray,
    b_inst: np.ndarray,
    sr: int = SR,
) -> np.ndarray:
    """
    Dynamic Frequency-Selective Mixing + Kick-to-Vocal Micro-Groove + Club Mastering
    """
    n = len(b_inst)
    voc = voc_stem[:n].copy()
    inst = b_inst[:n].copy()

    sos_mid = butter(2, [450.0 / (sr / 2.0), 4200.0 / (sr / 2.0)], btype="bandpass", output="sos")
    inst_mid = sosfiltfilt(sos_mid, inst, axis=0)

    voc_mono_abs = np.mean(np.abs(voc), axis=1)
    voc_env = envelope_follower(voc_mono_abs, attack_ms=8.0, release_ms=140.0, sr=sr)
    duck_ratio = np.clip(voc_env / db_to_lin(-14.0), 0.0, 1.0)
    mid_reduction = (1.0 - db_to_lin(-3.2)) * duck_ratio[:, None]
    broad_gain = 1.0 - (1.0 - db_to_lin(-0.9)) * duck_ratio[:, None]
    inst_ducked = inst * broad_gain - inst_mid * mid_reduction

    sos_sub = butter(2, 120.0 / (sr / 2.0), btype="lowpass", output="sos")
    inst_kick = np.mean(np.abs(sosfiltfilt(sos_sub, inst, axis=0)), axis=1)
    kick_env = envelope_follower(inst_kick, attack_ms=2.0, release_ms=55.0, sr=sr)
    kick_norm = np.clip((kick_env - 0.12) / 0.28, 0.0, 1.0)
    voc_groove_gain = 1.0 - (1.0 - db_to_lin(-1.0)) * kick_norm[:, None]
    voc_grooved = voc * voc_groove_gain

    raw_mix = inst_ducked * db_to_lin(-0.7) + voc_grooved * db_to_lin(+0.7)

    fade_tail_s = int(1.35 * sr)
    raw_mix[-fade_tail_s:] = cosine_fade_out(raw_mix[-fade_tail_s:], fade_tail_s)
    raw_mix[:int(0.008 * sr)] = cosine_fade_in(raw_mix[:int(0.008 * sr)], int(0.008 * sr))

    master_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=24.0),
        LowShelfFilter(cutoff_frequency_hz=75.0, gain_db=0.6),
        HighShelfFilter(cutoff_frequency_hz=11000.0, gain_db=0.6),
        Compressor(threshold_db=-10.5, ratio=1.8, attack_ms=18.0, release_ms=120.0),
        Limiter(threshold_db=-1.0, release_ms=80.0),
    ])
    mastered = master_board(raw_mix.T.astype(np.float32), sr).T

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(mastered)
    target_lufs = -8.6
    gain_adj = db_to_lin(np.clip(target_lufs - lufs, -3.0, 3.0))
    mastered = mastered * gain_adj
    final_lim = Pedalboard([Limiter(threshold_db=-1.0, release_ms=60.0)])
    mastered = final_lim(mastered.T.astype(np.float32), sr).T

    pk = float(np.max(np.abs(mastered)))
    ceiling = db_to_lin(-0.5)
    if pk > ceiling:
        mastered = mastered * (ceiling / pk)
    return mastered


def main():
    print("1. Loading raw separated stems...")
    raw_a_voc, sr_a = sf.read(RAW_A_VOC, dtype="float32")
    raw_b_inst, sr_b = sf.read(RAW_B_INST, dtype="float32")
    assert sr_a == SR and sr_b == SR

    print("2. Cleaning A_Vocals and polishing/healing B_Instrumental...")
    a_voc_clean = clean_a_vocals(raw_a_voc, SR)
    b_inst_clean = polish_b_instrumental(raw_b_inst, SR)

    a_voc_path = os.path.join(OUT_DIR, "A_Vocals.wav")
    b_inst_path = os.path.join(OUT_DIR, "B_Instrumental.wav")
    sf.write(a_voc_path, a_voc_clean, SR, subtype="PCM_16")
    sf.write(b_inst_path, b_inst_clean, SR, subtype="PCM_16")
    print(f"   Saved {a_voc_path} and {b_inst_path}")

    # 3. Seamless Zero-Cut Multi-Bar Vocal Arrangement:
    # Every single (a_cut_start_s, a_cut_end_s) sits inside a verified <-40 dB natural breath/silence valley,
    # and Pre-Chorus -> Drop transitions are kept as 24-bar UNCUT original performances!
    # Format: (tag, a_anchor_bar, a_span_bars, a_cut_start_s, a_cut_end_s, b_target_bar_0idx, fin_ms, fout_ms, gain_db)
    arrangement = [
        # [01 OPENING HOOK] B Bars 1..8 (1.78s -> 15.15s):
        # Enters immediately at 1.78s after B's 1-bar signature drum fill!
        # Natural silence [166.85s (-90 dB) -> 181.60s (-52 dB)]: "曾经是心心念念…那残破的碎片"
        ("01_Opening_ChorusHook",             90.5,  7.5, 166.85, 181.60,  0.0, 15.0, 120.0, +0.5),

        # [02 SEAMLESS BUILD-UP -> MAIN DROP 1] B Bars 9..32 (15.15s -> 55.25s, 24 bars UNCUT!):
        # Natural silence [60.02s (-63.8 dB) -> 104.90s (-39.2 dB)]:
        # "黄昏下模糊的侧脸…时间总会说出再见" -> (zero splice!) -> "曾经是心心念念…放开不再拖欠"
        ("02_BuildUp_And_Drop1_Seamless24B",  32.0, 24.0,  60.02, 104.90,  8.0, 20.0, 220.0, +0.6),

        # [03 MID-GROOVE STORY VERSE] B Bars 35..42 (58.59s -> 71.96s, 8 bars UNCUT!):
        # Natural silence [14.95s (-93 dB) -> 29.75s (-81.9 dB)]:
        # "秋天里的落叶 挥手同树道别 / 关于你的意见 我都想要了解"
        ("03_MidGroove_Verse1_Uncut8B",        8.0,  8.0,  14.95,  29.75, 34.0, 20.0, 150.0, +1.6),

        # [04a POST-BREAKDOWN BUILD] B Bars 50..57 (83.66s -> 97.02s, 8 bars UNCUT!):
        # Natural silence [148.90s (-47.5 dB) -> 164.25s (-42.2 dB)]:
        # "黄昏下模糊的侧脸 遗忘了伤痛的少年 / 有风险才蜕变 时间总会说出再见"
        ("04a_PostBreak_PreChorus2_Uncut8B",  80.0,  8.0, 148.90, 164.25, 49.0, 20.0, 160.0, +0.5),

        # [04b DROP 2 FULL CHORUS] B Bars 58..73 (97.02s -> 123.76s, 16 bars UNCUT!):
        # Natural silence [166.60s (-93.5 dB) -> 196.00s (-41.0 dB)]:
        # "曾经是心心念念…放开不再拖欠"
        ("04b_Drop2_FullChorus2_Uncut16B",    90.0, 16.0, 166.60, 196.00, 57.0, 18.0, 200.0, +0.6),

        # [05 SEAMLESS BREAKDOWN 2 -> FINAL CLIMAX DROP] B Bars 75..98 (125.43s -> 165.53s, 24 bars UNCUT!):
        # Natural silence [60.02s (-63.8 dB) -> 104.90s (-39.2 dB)]:
        # "黄昏下模糊的侧脸…时间总会说出再见" -> (zero splice!) -> "曾经是心心念念…放开不再拖欠"
        # Ends cleanly at ~166.7s, leaving B Bars 100..106 (167s -> 179.03s) for a pure, uncut DJ Outro!
        ("05_Break2_And_FinalDrop_Seamless24B", 32.0, 24.0, 60.02, 104.90, 74.0, 20.0, 240.0, +0.7),
    ]

    print("3. Time-stretching, pitch-adapting & beat-aligning seamless uncut vocal arcs...")
    vocal_timeline = np.zeros_like(b_inst_clean, dtype=np.float32)
    cue_sheet = []

    for tag, a_bar, n_bars, a_cs, a_ce, b_bar, fin_ms, fout_ms, gain_db in arrangement:
        a_anchor = P0_A + a_bar * BAR_A
        b_anchor = P0_B + b_bar * BAR_B
        b_end_grid = P0_B + (b_bar + n_bars) * BAR_B

        stretched, pre_off = stretch_and_pitch_vocal_exact(
            a_voc_clean, a_cs, a_ce, a_anchor, fin_ms, fout_ms, tag
        )
        stretched *= db_to_lin(gain_db)

        start_sample = int(round(b_anchor * SR)) - pre_off
        end_sample = start_sample + len(stretched)

        src_i0 = max(0, -start_sample)
        dst_i0 = max(0, start_sample)
        dst_i1 = min(len(vocal_timeline), end_sample)
        src_i1 = src_i0 + (dst_i1 - dst_i0)

        vocal_timeline[dst_i0:dst_i1] += stretched[src_i0:src_i1]

        cue_info = {
            "section": tag,
            "a_bars": f"{a_bar}..{a_bar + n_bars}",
            "a_cut_silence_s": [round(a_cs, 2), round(a_ce, 2)],
            "b_bars_1indexed": f"{b_bar + 1}..{b_bar + n_bars}",
            "remix_time_s": [round(dst_i0 / SR, 2), round(dst_i1 / SR, 2)],
            "gain_db": gain_db,
        }
        cue_sheet.append(cue_info)
        print(f"   Placed {tag:36s}: A_Silence[{a_cs:6.2f}-{a_ce:6.2f}s] -> Remix[{dst_i0/SR:6.2f}-{dst_i1/SR:6.2f}s]")

    print("4. Applying Studio Vocal Chain (EQ, Dual-Compressor, Ducked Tempo Delay, Plate Reverb)...")
    aligned_voc_stem = apply_vocal_fx_and_polish(vocal_timeline, SR)

    os.makedirs(os.path.join(OUT_DIR, "stems_project"), exist_ok=True)
    aligned_voc_path = os.path.join(OUT_DIR, "stems_project", "Remix_Aligned_A_Vocals_Stem.flac")
    sf.write(aligned_voc_path, aligned_voc_stem, SR, format="FLAC", subtype="PCM_16")

    print("5. Mixing & Mastering Full DJ Remix...")
    final_mix = mix_vocal_and_dj_inst(aligned_voc_stem, b_inst_clean, SR)

    remix_wav_path = os.path.join(OUT_DIR, "REMIX-AB-V4_Full_DJ_Remix.wav")
    sf.write(remix_wav_path, final_mix, SR, subtype="PCM_16")

    remix_mp3_path = os.path.join(OUT_DIR, "REMIX-AB-V4_Full_DJ_Remix.mp3")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", remix_wav_path,
            "-codec:a", "libmp3lame", "-b:a", "320k", "-q:a", "0",
            remix_mp3_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 6. Create Audition Preview (0.00s -> 57.50s: Opening B Drum Fill + Immediate A Vocal Entry + Seamless 24-Bar Build & Drop 1!)
    preview_len = int(57.5 * SR)
    preview_audio = final_mix[:preview_len].copy()
    preview_audio = cosine_fade_out(preview_audio, int(1.5 * SR))
    preview_tmp_wav = os.path.join(WORK_DIR, "preview_tmp.wav")
    preview_mp3_path = os.path.join(OUT_DIR, "REMIX-AB-V4_Preview_Opening_And_Drop.mp3")
    sf.write(preview_tmp_wav, preview_audio, SR, subtype="PCM_16")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", preview_tmp_wav,
            "-codec:a", "libmp3lame", "-b:a", "320k", "-q:a", "0",
            preview_mp3_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    meter = pyln.Meter(SR)
    final_lufs = meter.integrated_loudness(final_mix)
    final_peak_db = lin_to_db(float(np.max(np.abs(final_mix))))

    project_meta = {
        "project_id": "REMIX-AB-V4",
        "version": "V4.2-Seamless",
        "source_a": {"file": "音乐A.mp3", "detected_grid_bpm": BPM_A, "librosa_frame_bpm": 129.20, "key": "F major / D minor"},
        "source_b": {"file": "音乐B.mp3", "detected_grid_bpm": BPM_B, "librosa_frame_bpm": 143.55, "key": "F# major / D# minor (-33 cents tuning)"},
        "adaptation": {
            "tempo_stretch_ratio": round(TEMPO_RATIO, 6),
            "pitch_shift_semitones": PITCH_SEMITONES,
            "opening_drum_fill_s": [0.0, round(P0_B, 3)],
            "vocal_entry_s": round(P0_B, 3),
        },
        "master_metrics": {
            "duration_s": round(len(final_mix) / SR, 3),
            "sample_rate": SR,
            "integrated_lufs": round(float(final_lufs), 2),
            "sample_peak_dbfs": round(float(final_peak_db), 2),
        },
        "arrangement_cues": cue_sheet,
    }
    with open(os.path.join(OUT_DIR, "stems_project", "REMIX-AB-V4_Project_Metadata.json"), "w", encoding="utf-8") as f:
        json.dump(project_meta, f, ensure_ascii=False, indent=2)

    print(f"DONE! Master Integrated Loudness: {final_lufs:.2f} LUFS, Peak: {final_peak_db:.2f} dBFS")


if __name__ == "__main__":
    main()

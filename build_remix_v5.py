#!/usr/bin/env python3
"""
REMIX-AB-V5 — 「碎片重燃 / Fragments Rekindled」
A Vocal × B DJ Remix — 全新编曲（凭感觉重编版）

编曲思路（与 V4 完全不同的段落映射）:
  0) 0.000–1.779   B 标志性开场鼓点(不动)
  1) B-bars 1–8    开场 Hook:   Chorus2 前半「曾经是心心念念…那残破的碎片」(7.5小节)
  2) B-bars 9–16   Build-up 1:  Verse 1a「秋天里的落叶…我都想要了解」(8小节, 情绪随 Build 上升)
  3) B-bars 17–32  Main Drop 1: 完整副歌「曾经是心心念念…放开不再拖欠」(16小节, 副歌炸在 Drop)
  4) B-bars 33–34  Breather:    纯器乐呼吸(人声延迟尾音)
  5) B-bars 35–51  Mid-Groove → Breakdown → Predrop 头部:
                   Verse 1b+Pre 1a 十六小节连唱「陪伴三个季节…没有勇气想得更远」
                   —「抱歉→奔跑」连唱交界恰好落在 Breakdown(鼓点撤出)正拍,
                    预副歌在无鼓氛围里推进,「更远」收在 Predrop 鼓点回归后 2.4 小节
  6) B-bars 52–57  纯器乐 Riser 蓄力(无任何人声, 10.9s 纯上升)
  7) B-bars 58–73  Main Drop 2: 完整副歌第二次爆发
  8) B-bars 74     Impact Hit
  9) B-bars 75–98  Breakdown 2 → Final Drop:
                   24 小节王牌连唱「黄昏下模糊的侧脸…时间总会说出再见 →(零气口)→
                   曾经是心心念念…放开不再拖欠」——「再见→曾经是」连唱交界精确压在
                   终极 Drop 重拍 (138.798s) 上
 10) B-bars 99–106 DJ Club Outro(人声混响/延迟尾音)

混音/母带: 人声 EQ+压缩+饱和 → 板混+1/8乒乓延迟(乐句尾 throw 加倍) →
           人声包络侧链避让 → 母带 EQ/立体声展宽/胶水压缩 → -7.8 LUFS → 砖墙限幅 -0.8 dBTP
"""

import json
import os

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

# ----------------------------------------------------------------------------
# 常量: 双曲网格 (来自 ARRANGEMENT_SPEC / README_HANDOFF_KIT)
# ----------------------------------------------------------------------------
SR = 44100
BPM_A = 130.00
BPM_B = 143.63
TEMPO_RATIO = BPM_B / BPM_A          # 1.10484615
BEAT_A = 60.0 / BPM_A                # 0.46153846 s
BAR_A = 4.0 * BEAT_A                 # 1.84615385 s
P0_A = 0.0218                        # A bar0 beat1 锚点
BEAT_B = 60.0 / BPM_B                # 0.41773996 s
BAR_B = 4.0 * BEAT_B                 # 1.67095983 s
P0_B = 1.7790                        # B bar1 beat1 (开场鼓点后)

VOC_PATH = "handoff_kit_for_other_models/A_Vocals_Full_Adapted_143.63BPM_Fsharp.flac"
B_PATH = "B_Instrumental.wav"

OUT_WAV = "REMIX-AB-V5_Full_Remix.wav"
OUT_MP3 = "REMIX-AB-V5_Full_Remix.mp3"
OUT_PREVIEW = "REMIX-AB-V5_Preview_Opening_And_Drop.mp3"
OUT_STEM = "stems_project/REMIX-AB-V5_Vocal_Stem.flac"
OUT_META = "stems_project/REMIX-AB-V5_Project_Metadata.json"


def a_bar_to_adapted(a_bar: float) -> float:
    """A 小节坐标 → 适配轨(143.63BPM)时间."""
    return (P0_A + a_bar * BAR_A) / TEMPO_RATIO + 0.009


def b_bar_to_time(b_bar: float) -> float:
    """B 小节坐标(0-indexed, bar 1 = 0) → 伴奏轨时间."""
    return P0_B + b_bar * BAR_B


# ----------------------------------------------------------------------------
# 编曲坐标表: 每段人声 = (标签, 适配轨安全切窗, A 小节锚点, 对齐到的 B 小节锚点)
# ----------------------------------------------------------------------------
# 切窗 = 静音谷内 < -45dB 物理切点(实测校验过), 锚点对齐保证整段内部节拍与 B 网格锁死
PLACEMENTS = [
    # 开场 Hook: Chorus2 前半 (7.5 小节), Bar 90.5 锚点 → B bar 1 重拍
    dict(tag="Hook_Open", cut=(151.03, 164.38), a_anchor_bar=90.5, b_anchor_bar=0,
         gain_db=0.6, rev=0.13, dly=0.10),
    # Build-up 1: Verse 1a (8 小节), Bar 8.0 锚点 → B bar 9 重拍
    dict(tag="Verse1a_Build", cut=(13.54, 26.94), a_anchor_bar=8.0, b_anchor_bar=8,
         gain_db=1.5, rev=0.15, dly=0.05),
    # Main Drop 1: 完整副歌, Bar 90.5 锚点 → B bar 17 重拍 (「曾经是」推在重拍前)
    dict(tag="Chorus_Drop1", cut=(150.80, 177.55), a_anchor_bar=90.5, b_anchor_bar=16,
         gain_db=0.5, rev=0.10, dly=0.08),
    # Verse1b+Pre1a 十六小节连唱, Bar 16.0 锚点 → B bar 35 重拍 (Mid-Groove 起点)
    dict(tag="Verse1b_Pre1a", cut=(26.94, 54.33), a_anchor_bar=16.0, b_anchor_bar=34,
         gain_db=1.15, rev=0.15, dly=0.07),
    # Main Drop 2: 完整副歌, Bar 90.5 锚点 → B bar 58 重拍
    dict(tag="Chorus_Drop2", cut=(150.80, 177.55), a_anchor_bar=90.5, b_anchor_bar=57,
         gain_db=0.5, rev=0.10, dly=0.08),
    # 24 小节王牌连唱 (Pre1b→Chorus1 零气口), Bar 32.0 锚点 → B bar 75 重拍
    # (内部「再见→曾经是」连唱交界 = A bar 40 → B bar 83 = Final Drop 重拍)
    dict(tag="Pre1b_Chorus1_Seamless", cut=(54.33, 95.07), a_anchor_bar=32.0, b_anchor_bar=74,
         gain_db=1.15, rev=0.17, dly=0.11),
]

# U2/U3 内部能量微调: (时间点, 增益dB, 混响send, 延迟send) — 平滑插值
# Verse1b+Pre1a: 连唱交界(71.97s) 前后 = Mid-Groove(热) → Breakdown(冷, 湿)
# Pre1b→Chorus1: 交界(138.798s) 前后 = Breakdown(湿) → Final Drop(干、亮)
FINE_RIDES = [
    # t,        gain_db, rev,  dly   (在 PLACEMENTS 基准上做插值微调)
    (58.70, 1.35, 0.13, 0.06),
    (71.60, 1.25, 0.14, 0.07),
    (72.40, 1.05, 0.21, 0.11),
    (84.60, 1.05, 0.21, 0.11),
    (86.20, 1.05, 0.21, 0.11),
    (126.20, 1.15, 0.22, 0.13),
    (138.40, 1.15, 0.22, 0.13),
    (139.20, 0.45, 0.11, 0.09),
    (166.90, 0.45, 0.12, 0.10),
]

# 乐句尾 throw: 人声结尾 1.2s 内 send 加倍, 让「碎片/更远/拖欠/再见」被回声接住
THROW_WINDOWS = [
    (13.7, 15.1), (27.3, 28.8), (53.4, 55.0), (84.9, 86.3),
    (122.1, 123.5), (137.4, 138.9), (165.7, 167.1),
]


# ----------------------------------------------------------------------------
# DSP 工具
# ----------------------------------------------------------------------------
def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def lin_to_db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def biquad_peaking(f0, gain_db, Q):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2.0 * Q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return signal.tf2sos(np.array(b) / a[0], np.array(a) / a[0])


def biquad_shelf(f0, gain_db, high=False, S=1.0):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / SR
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)
    cosw = np.cos(w0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(A) * alpha
    if not high:
        b0 = A * ((A + 1) - (A - 1) * cosw + two_sqrt_a_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - two_sqrt_a_alpha)
        a0 = (A + 1) + (A - 1) * cosw + two_sqrt_a_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - two_sqrt_a_alpha
    else:
        b0 = A * ((A + 1) + (A - 1) * cosw + two_sqrt_a_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - two_sqrt_a_alpha)
        a0 = (A + 1) - (A - 1) * cosw + two_sqrt_a_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - two_sqrt_a_alpha
    return signal.tf2sos(np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0)


def biquad_hpf(f0, Q=0.707):
    w0 = 2.0 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2.0 * Q)
    cosw = np.cos(w0)
    b = np.array([(1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2])
    a = np.array([1 + alpha, -2 * cosw, 1 - alpha])
    return signal.tf2sos(b / a[0], a / a[0])


def butter_hpf(f0, order=2):
    return signal.butter(order, f0, btype="high", fs=SR, output="sos")


def butter_lp(f0, order=2):
    return signal.butter(order, f0, btype="low", fs=SR, output="sos")


def cosine_fade(n, fade_in=0, fade_out=0):
    env = np.ones(n)
    if fade_in > 0:
        k = min(fade_in, n)
        env[:k] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    if fade_out > 0:
        k = min(fade_out, n)
        env[n - k:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
    return env


def compress_block(x, thr_db, ratio, att_ms, rel_ms, knee_db=6.0, makeup_db=0.0, block=256):
    """前馈压缩器(块级电平检测+软拐点+起释平滑)."""
    n = x.shape[0]
    pad = (-n) % block
    if pad:
        x = np.pad(x, ((0, pad), (0, 0)))
    nb = x.shape[0] // block
    blk = x.reshape(nb, block, -1)
    det_db = lin_to_db(np.max(np.abs(blk), axis=(1, 2)))
    # 软拐点增益计算
    over = det_db - thr_db
    gain_db = np.zeros(nb)
    m1 = over > knee_db / 2
    m2 = (~m1) & (over > -knee_db / 2)
    gain_db[m1] = -(1 - 1 / ratio) * over[m1]
    gain_db[m2] = -(1 - 1 / ratio) * (over[m2] + knee_db / 2) ** 2 / (2 * knee_db)
    # 块间 attack/release 平滑
    a_att = np.exp(-block / max(1.0, att_ms * 0.001 * SR))
    a_rel = np.exp(-block / max(1.0, rel_ms * 0.001 * SR))
    state = 0.0
    smoothed = np.empty(nb)
    for i in range(nb):
        g = gain_db[i]
        coef = a_att if g < state else a_rel
        state = coef * state + (1 - coef) * g
        smoothed[i] = state
    # 块增益线性插值回采样
    centers = (np.arange(nb) + 0.5) * block
    t = np.arange(x.shape[0])
    g_db = np.interp(t, centers, smoothed)
    y = x * db_to_lin(g_db + makeup_db)[:, None]
    return y[:n]


def make_reverb_ir(decay=1.55, predelay=0.022, seed=7):
    rng = np.random.default_rng(seed)
    n = int(decay * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2))
    # 指数衰减 + 尾部逐渐变暗
    ir *= np.exp(-6.9 * t / decay)[:, None]
    lp = butter_lp(5200, 2)
    ir = signal.sosfilt(lp, ir, axis=0)
    # 早期反射
    for d_ms, g in [(11, 0.5), (19, -0.38), (27, 0.3), (37, -0.24), (53, 0.18)]:
        d = int(d_ms * 0.001 * SR)
        ir[d:d + 200, :] += g * ir[:200, :]
    pre = np.zeros((int(predelay * SR), 2))
    ir = np.concatenate([pre, ir], axis=0)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0)).max()
    return ir.astype(np.float64)


def pingpong_delay(send_mono, dt=0.20887, fb=0.40, taps=8):
    """1/8 音符乒乓延迟: 偶数拍左、奇数拍右, 每次回声低通变暗."""
    n = len(send_mono)
    out = np.zeros((n, 2))
    lp = butter_lp(3800, 2)
    cur = send_mono.copy()
    for k in range(1, taps + 1):
        d = int(k * dt * SR)
        if d >= n:
            break
        amp = fb ** k
        tap = np.zeros((n, 2))
        tap[d:, k % 2] = cur[:-d]
        tap = signal.sosfilt(lp, tap, axis=0)
        out += amp * tap
        cur = signal.sosfilt(lp, cur.reshape(-1, 1), axis=0).ravel()  # 逐拍变暗
    return out


def limiter(x, ceil_db=-0.8, look_ms=5.0, rel_ms=90.0, block=128):
    ceil = float(db_to_lin(ceil_db))
    n = x.shape[0]
    pad = (-n) % block
    if pad:
        x = np.pad(x, ((0, pad), (0, 0)))
    need = np.minimum(1.0, ceil / np.maximum(np.max(np.abs(x), axis=1), 1e-9))
    look = max(1, int(look_ms * 0.001 * SR))
    need = minimum_filter1d(need, size=look + 1)
    nb = x.shape[0] // block
    gblk = need.reshape(nb, block).min(axis=1)
    a_rel = np.exp(-block / max(1.0, rel_ms * 0.001 * SR))
    state = 1.0
    out_g = np.empty(nb)
    for i in range(nb):
        g = gblk[i]
        if g < state:
            state = g          # 瞬时下压
        else:
            state = a_rel * state + (1 - a_rel) * g
        out_g[i] = state
    centers = (np.arange(nb) + 0.5) * block
    g = np.interp(np.arange(x.shape[0]), centers, out_g)
    y = x * g[:, None]
    y = np.clip(y, -ceil, ceil)
    return y[:n]


def ramp_envelope(n_pts, breakpoints):
    """断点插值包络: breakpoints = [(t, value), ...]"""
    ts = np.array([p[0] for p in breakpoints])
    vs = np.array([p[1] for p in breakpoints])
    return ts, vs


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    os.makedirs("stems_project", exist_ok=True)

    print("== 加载素材 ==")
    voc, _ = sf.read(VOC_PATH, dtype="float64")      # 适配干声 143.63BPM / F#
    b, _ = sf.read(B_PATH, dtype="float64")          # 纯净 DJ 伴奏
    n_total = b.shape[0]
    dur = n_total / SR
    print(f"   适配干声 {len(voc)/SR:.3f}s | 伴奏 {dur:.3f}s")

    # ------------------------------------------------------------------
    # 1) 按编曲表拼接人声时间线 (纯数字, 边界 15ms 余弦淡入出)
    # ------------------------------------------------------------------
    print("== 拼接人声时间线 (V5 编曲) ==")
    voc_tl = np.zeros((n_total, 2))
    seg_meta = []
    for p in PLACEMENTS:
        a_anchor = a_bar_to_adapted(p["a_anchor_bar"])
        b_anchor = b_bar_to_time(p["b_anchor_bar"])
        shift = b_anchor - a_anchor
        i0 = int(round(p["cut"][0] * SR))
        i1 = int(round(p["cut"][1] * SR))
        clip = voc[i0:i1].copy()
        # 节拍网格校验: S 必须 ≡ 相位差 (mod BEAT_B)
        phase = (shift - 0.079308) / BEAT_B
        grid_err_ms = abs(phase - round(phase)) * BEAT_B * 1000
        fade = int(0.015 * SR)
        clip *= cosine_fade(len(clip), fade, fade)[:, None]
        t0 = int(round((p["cut"][0] + shift) * SR))
        t1 = t0 + len(clip)
        assert t1 <= n_total, f"{p['tag']} 超出总长!"
        voc_tl[t0:t1] += clip
        seg_meta.append(dict(tag=p["tag"], shift=shift,
                             remix_start=(p["cut"][0] + shift),
                             remix_end=(p["cut"][1] + shift),
                             grid_err_ms=grid_err_ms))
        print(f"   {p['tag']:24s} 切窗{p['cut']} → {p['cut'][0]+shift:7.3f}–{p['cut'][1]+shift:7.3f}s"
              f" | 锚点 A-bar {p['a_anchor_bar']}→B-bar {p['b_anchor_bar']}"
              f" | 网格误差 {grid_err_ms:.2f}ms")

    # ------------------------------------------------------------------
    # 2) 人声链: HPF → EQ → 压缩 → 饱和 → 分段增益
    # ------------------------------------------------------------------
    print("== 人声链 (EQ/压缩/饱和/段落骑推) ==")
    v = voc_tl
    v = signal.sosfilt(butter_hpf(85, 2), v, axis=0)
    v = signal.sosfilt(biquad_peaking(250, -2.5, 1.0), v, axis=0)      # 去浑
    v = signal.sosfilt(biquad_peaking(3300, 3.0, 1.1), v, axis=0)      # 唇齿清晰度
    v = signal.sosfilt(biquad_shelf(10500, 2.0, high=True), v, axis=0)  # 空气感
    v = compress_block(v, thr_db=-24, ratio=2.8, att_ms=6, rel_ms=140,
                       knee_db=6, makeup_db=3.0)
    v = np.tanh(1.25 * v) / np.tanh(1.25)                                # 轻饱和

    # 段落增益/效果 send 包络 (PLACEMENTS 基准 + FINE_RIDES 微调, 全部 ≥200ms 平滑)
    tt = np.arange(n_total) / SR
    gain_pts, rev_pts, dly_pts = [], [], []
    for p, m in zip(PLACEMENTS, seg_meta):
        s, e = m["remix_start"], m["remix_end"]
        gain_pts += [(s, p["gain_db"]), (e, p["gain_db"])]
        rev_pts += [(s, p["rev"]), (e, p["rev"])]
        dly_pts += [(s, p["dly"]), (e, p["dly"])]
    # 段间空档保持末值 (无插值跳变风险, 人声静音)
    for fine in FINE_RIDES:
        gain_pts.append((fine[0], fine[1]))
        rev_pts.append((fine[0], fine[2]))
        dly_pts.append((fine[0], fine[3]))
    gain_pts = sorted(set(gain_pts))
    rev_pts = sorted(set(rev_pts))
    dly_pts = sorted(set(dly_pts))
    trim_db = np.interp(tt, *ramp_envelope(0, gain_pts))
    rev_send = np.interp(tt, *ramp_envelope(0, rev_pts))
    dly_send = np.interp(tt, *ramp_envelope(0, dly_pts))
    # 乐句尾 throw (send 抬升, 700ms 余弦过渡)
    for w0, w1 in THROW_WINDOWS:
        m = (tt >= w0) & (tt <= w1)
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m.sum()))) if m.sum() else np.array([])
        rev_send[m] += 0.10 * ramp
        dly_send[m] += 0.07 * ramp

    v *= db_to_lin(trim_db)[:, None]

    # ------------------------------------------------------------------
    # 3) 效果: 板混响 + 乒乓延迟
    # ------------------------------------------------------------------
    print("== 效果 (板混 + 1/8 乒乓延迟) ==")
    ir = make_reverb_ir()
    send_rev = v * rev_send[:, None]
    wet_rev = signal.fftconvolve(send_rev, ir, axes=0)[:n_total]
    send_dly_mono = np.mean(v, axis=1) * dly_send
    wet_dly = pingpong_delay(send_dly_mono)

    vocal_bus = v + 0.62 * wet_rev + 0.55 * wet_dly
    # 高频立体声微展宽 (人声干声近单声, 宽度放在空气层)
    sos_air = biquad_hpf(2200, 0.707)
    air = signal.sosfilt(sos_air, vocal_bus, axis=0)
    mid = air.mean(axis=1, keepdims=True)
    side = air - mid
    vocal_bus = vocal_bus + 0.22 * side   # 逐声道偏离量即侧信号, 抬 22% = 高频展宽

    # ------------------------------------------------------------------
    # 4) 人声包络侧链避让 (伴奏让 1.6dB) + 伴奏保护性 HPF
    # ------------------------------------------------------------------
    print("== 混合 (人声侧链避让) ==")
    b = signal.sosfilt(butter_hpf(25, 2), b, axis=0)
    env = np.max(np.abs(vocal_bus), axis=1)
    # 块级包络: 20ms 攻击 / 300ms 释放
    blk = 256
    pad = (-len(env)) % blk
    e = np.pad(env, (0, pad)).reshape(-1, blk).max(axis=1)
    a_att = np.exp(-blk / (0.020 * SR))
    a_rel = np.exp(-blk / (0.300 * SR))
    state = 0.0
    sm = np.empty(len(e))
    for i in range(len(e)):
        x = e[i]
        coef = a_att if x > state else a_rel
        state = coef * state + (1 - coef) * x
        sm[i] = state
    centers = (np.arange(len(e)) + 0.5) * blk
    env_s = np.interp(np.arange(len(env)), centers, sm)
    duck = 1.0 - 0.17 * np.clip(env_s / 0.10, 0, 1)     # 最多 -1.6dB
    b_ducked = b * duck[:, None]

    mix = b_ducked + vocal_bus
    peak_pre = np.max(np.abs(mix))
    print(f"   混合后峰值 {lin_to_db(peak_pre):.2f} dBFS")

    # ------------------------------------------------------------------
    # 5) 母带: EQ → 展宽 → 胶水 → 响度 → 限幅
    # ------------------------------------------------------------------
    print("== 母带 ==")
    m = signal.sosfilt(biquad_shelf(78, 0.8), mix, axis=0)      # 低频配重
    m = signal.sosfilt(biquad_peaking(280, -1.0, 0.9), m, axis=0)
    m = signal.sosfilt(biquad_shelf(9000, 0.8, high=True), m, axis=0)
    # 高频 M/S 展宽
    sos_airm = biquad_hpf(6000, 0.707)
    hi = signal.sosfilt(sos_airm, m, axis=0)
    mid = hi.mean(axis=1, keepdims=True)
    side = hi - mid
    m = m + 0.18 * side   # 高频侧信号抬 18%
    # 胶水压缩
    m = compress_block(m, thr_db=-15, ratio=1.6, att_ms=25, rel_ms=300,
                       knee_db=8, makeup_db=0.5)
    # 响度归一 + 限幅迭代收敛 (限幅会吃掉约 1dB 响度, 迭代逼近目标)
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    target_lufs = -7.8
    lufs_pre = meter.integrated_loudness(m)
    gain_total_db = 0.0
    for it in range(4):
        lufs_cur = meter.integrated_loudness(m)
        gain_db = target_lufs - lufs_cur
        if abs(gain_db) < 0.15:
            break
        gain_total_db += gain_db
        m = limiter(m * db_to_lin(gain_db), ceil_db=-0.8)
        print(f"   迭代{it+1}: LUFS {lufs_cur:.2f} → 增益 {gain_db:+.2f} dB")
    lufs_post = meter.integrated_loudness(m)
    peak_post = np.max(np.abs(m))
    print(f"   成品: LUFS {lufs_post:.2f} | 峰值 {lin_to_db(peak_post):.2f} dBFS | 时长 {len(m)/SR:.3f}s")

    # ------------------------------------------------------------------
    # 6) 导出
    # ------------------------------------------------------------------
    print("== 导出 ==")
    sf.write(OUT_WAV, m, SR, subtype="PCM_16")
    print(f"   {OUT_WAV}")

    stem = vocal_bus * db_to_lin(gain_total_db) * 0.85
    sf.write(OUT_STEM, stem, SR, subtype="PCM_24")
    print(f"   {OUT_STEM}")

    import lameenc
    def write_mp3(path, audio, sr=SR):
        pcm = (np.clip(audio, -1, 1) * 32767.0).astype("<i2").tobytes()
        enc = lameenc.Encoder()
        enc.set_bit_rate(320)
        enc.set_in_sample_rate(sr)
        enc.set_channels(2)
        enc.set_quality(2)
        data = enc.encode(pcm)
        data += enc.flush()
        with open(path, "wb") as f:
            f.write(bytes(data))
        print(f"   {path}")

    write_mp3(OUT_MP3, m)
    write_mp3(OUT_PREVIEW, m[: int(46.0 * SR)])

    # ------------------------------------------------------------------
    # 7) 工程元数据
    # ------------------------------------------------------------------
    meta = {
        "project_id": "REMIX-AB-V5",
        "version": "V5-Fragments-Rekindled",
        "concept": "开场Hook→主歌Build→副歌Drop1→连唱Verse/Pre跨Breakdown→副歌Drop2→24小节零气口连唱压终极Drop→DJ Outro",
        "source_a": {"file": "音乐A.mp3", "grid_bpm": 130.0, "key": "F major / D minor"},
        "source_b": {"file": "音乐B.mp3", "grid_bpm": 143.63, "key": "F# major / D# minor (-33 cents)"},
        "master_metrics": {
            "duration_s": round(len(m) / SR, 3),
            "sample_rate": SR,
            "integrated_lufs": round(float(lufs_post), 2),
            "sample_peak_dbfs": round(float(lin_to_db(peak_post)), 2),
        },
        "arrangement_cues": [
            {
                "section": s["tag"],
                "a_bars_anchor": p["a_anchor_bar"],
                "b_bars_anchor_0idx": p["b_anchor_bar"],
                "a_cut_silence_s": list(p["cut"]),
                "remix_time_s": [round(s["remix_start"], 3), round(s["remix_end"], 3)],
                "gain_db": p["gain_db"],
                "beat_grid_error_ms": round(s["grid_err_ms"], 3),
            }
            for p, s in zip(PLACEMENTS, seg_meta)
        ],
        "b_section_map": {
            "Intro_Signature_Drum_Fill": [0.0, 1.779],
            "Opening_Groove_hook": [1.779, 15.147],
            "BuildUp_1_verse": [15.147, 28.514],
            "Main_Drop_1_chorus": [28.514, 55.248],
            "PostDrop_Breather": [55.248, 58.592],
            "MidGroove_to_Predrop_seamless16B": [58.592, 86.14],
            "Pure_Riser": [86.14, 97.024],
            "Main_Drop_2_chorus": [97.024, 123.759],
            "Impact_Hit": [123.759, 125.430],
            "Breakdown2_FinalDrop_seamless24B": [125.430, 166.998],
            "DJ_Club_Outro_tails": [166.998, 179.026],
        },
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"   {OUT_META}")
    print("完成 ✓")


if __name__ == "__main__":
    main()

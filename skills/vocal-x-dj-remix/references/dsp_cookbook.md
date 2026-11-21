# DSP Cookbook（可直接复制的配方）

> SKILL.md 第 4、5 节的代码化。全部配方在 44.1kHz float64 下实测通过。
> 通用性说明：时间常数与频率是"好听的起点"，换素材微调 ±20% 即可。

```python
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

SR = 44100

def db_to_lin(db): return 10.0 ** (np.asarray(db, float) / 20.0)
def lin_to_db(x):  return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))
```

## 1. 滤波器（RBJ biquad + Butterworth）

```python
def biquad_peaking(f0, gain_db, Q):
    A = 10 ** (gain_db / 40); w0 = 2*np.pi*f0/SR; alpha = np.sin(w0)/(2*Q)
    b = [1+alpha*A, -2*np.cos(w0), 1-alpha*A]
    a = [1+alpha/A, -2*np.cos(w0), 1-alpha/A]
    return signal.tf2sos(np.array(b)/a[0], np.array(a)/a[0])

def biquad_shelf(f0, gain_db, high=False, S=1.0):
    A = 10 ** (gain_db / 40); w0 = 2*np.pi*f0/SR
    alpha = np.sin(w0)/2*np.sqrt((A+1/A)*(1/S-1)+2); c = np.cos(w0)
    t = 2*np.sqrt(A)*alpha
    if not high:
        b = [A*((A+1)-(A-1)*c+t), 2*A*((A-1)-(A+1)*c), A*((A+1)-(A-1)*c-t)]
        a = [(A+1)+(A-1)*c+t, -2*((A-1)+(A+1)*c), (A+1)+(A-1)*c-t]
    else:
        b = [A*((A+1)+(A-1)*c+t), -2*A*((A-1)+(A+1)*c), A*((A+1)+(A-1)*c-t)]
        a = [(A+1)-(A-1)*c+t, 2*((A-1)-(A+1)*c), (A+1)-(A-1)*c-t]
    return signal.tf2sos(np.array(b)/a[0], np.array(a)/a[0])

butter_hpf = lambda f, o=2: signal.butter(o, f, 'high', fs=SR, output='sos')
butter_lp  = lambda f, o=2: signal.butter(o, f, 'low',  fs=SR, output='sos')
```

## 2. 包络与淡入出

```python
def cosine_fade(n, fade_in=0, fade_out=0):
    env = np.ones(n)
    if fade_in:  k = min(fade_in, n);  env[:k] = 0.5*(1-np.cos(np.linspace(0, np.pi, k)))
    if fade_out: k = min(fade_out, n); env[n-k:] = 0.5*(1+np.cos(np.linspace(0, np.pi, k)))
    return env
```

## 3. 块级压缩器（软拐点 + 起释平滑）

```python
def compress_block(x, thr_db, ratio, att_ms, rel_ms, knee_db=6.0, makeup_db=0.0, block=256):
    n = x.shape[0]; pad = (-n) % block
    if pad: x = np.pad(x, ((0, pad), (0, 0)))
    nb = x.shape[0] // block
    det = lin_to_db(np.abs(x.reshape(nb, block, -1)).max(axis=(1, 2)))
    over = det - thr_db
    g = np.zeros(nb)
    m1 = over > knee_db/2; m2 = (~m1) & (over > -knee_db/2)
    g[m1] = -(1-1/ratio)*over[m1]
    g[m2] = -(1-1/ratio)*(over[m2]+knee_db/2)**2/(2*knee_db)
    aa = np.exp(-block/max(1, att_ms*1e-3*SR)); ar = np.exp(-block/max(1, rel_ms*1e-3*SR))
    state, sm = 0.0, np.empty(nb)
    for i in range(nb):
        coef = aa if g[i] < state else ar
        state = coef*state + (1-coef)*g[i]; sm[i] = state
    g_db = np.interp(np.arange(x.shape[0]), (np.arange(nb)+0.5)*block, sm)
    return (x * db_to_lin(g_db + makeup_db)[:, None])[:n]
```

## 4. 板混响 IR + 1/8 乒乓延迟

```python
def make_reverb_ir(decay=1.55, predelay=0.022, seed=7):
    rng = np.random.default_rng(seed)
    n = int(decay*SR); t = np.arange(n)/SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9*t/decay)[:, None]
    ir = signal.sosfilt(butter_lp(5200), ir, axis=0)
    for d_ms, g in [(11,.5), (19,-.38), (27,.3), (37,-.24), (53,.18)]:
        d = int(d_ms*1e-3*SR); ir[d:d+200] += g*ir[:200]
    ir = np.concatenate([np.zeros((int(predelay*SR), 2)), ir], 0)
    return ir / np.sqrt((ir**2).sum(0)).max()

def pingpong_delay(send_mono, dt=0.20887, fb=0.40, taps=8):   # dt = 1/8 音符
    n = len(send_mono); out = np.zeros((n, 2)); lp = butter_lp(3800)
    cur = send_mono.copy()
    for k in range(1, taps+1):
        d = int(k*dt*SR)
        if d >= n: break
        tap = np.zeros((n, 2)); tap[d:, k % 2] = cur[:-d]
        out += (fb**k) * signal.sosfilt(lp, tap, axis=0)
        cur = signal.sosfilt(lp, cur.reshape(-1, 1), axis=0).ravel()
    return out
# 用法: wet = signal.fftconvolve(v*rev_send[:,None], ir, axes=0)[:n]
```

## 5. 网格回声（dub throw）

```python
def add_ghost(v, src_t0, src_t1, at, gain_db, lp_hz, pan, sr=SR):
    """把 [src_t0,src_t1] 的声音变暗、摆位、放到 at 时刻(全部秒, 混音时间轴)."""
    seg = v[int(src_t0*sr):int(src_t1*sr)].copy()
    seg = signal.sosfilt(butter_lp(lp_hz), seg, axis=0)
    seg *= cosine_fade(len(seg), int(0.008*sr), int(0.05*sr))[:, None]
    seg *= db_to_lin(gain_db) * np.array([1-pan, 1+pan])
    d = int(round(at*sr)); e = min(d+len(seg), len(v))
    v[d:e] += seg[:e-d]
```
用法要点：落点必须是 B 的拍/小节网格点；增益 -8→-21dB 递减；
低通 4k→1k 递暗；pan ±0.2 交替。源窗取句尾 0.8–1.0s。

## 6. Drop 前退场 / Sub 冲击 / 开场 Riser

```python
def predrop_dip(b, drop_t, depth_db=-4.5, w=0.5):        # 原地修改伴奏
    j0, j1 = int((drop_t-w)*SR), int(drop_t*SR)
    ramp = 0.5*(1-np.cos(np.linspace(0, np.pi, j1-j0)))
    b[j0:j1] *= db_to_lin(depth_db*ramp)[:, None]

def sub_impact(at, peak_db=-11.0, dur=1.1):
    t = np.arange(int(dur*SR))/SR
    ph = 2*np.pi*np.cumsum(32 + 28*np.exp(-t/0.09))/SR
    x = np.sin(ph)*np.exp(-t/0.32)
    return x * cosine_fade(len(x), int(0.004*SR), int(0.25*SR)) * db_to_lin(peak_db)

def intro_riser(dur, lo_lp=900, hi_lp=8000, db0=-25, db1=-15, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur*SR); a = np.arange(n)/n
    nz = rng.standard_normal((n, 2))
    x = (1-a)[:,None]*signal.sosfilt(butter_lp(lo_lp), nz, axis=0) \
        + a[:,None]*signal.sosfilt(butter_lp(hi_lp), nz, axis=0)
    x *= db_to_lin(db0 + (db1-db0)*a**1.5)[:, None]
    return x * cosine_fade(n, int(0.05*SR), int(0.025*SR))[:, None]
```

## 7. 软削波（⚠️ 单位小信号增益）+ 限幅器

```python
def softclip(x, thr=0.65):
    """单位小信号增益。千万别用 tanh(d*x)/tanh(d) —— 它在小信号处有 d/tanh(d) 倍增益!"""
    ax = np.abs(x); over = ax > thr; y = x.copy()
    y[over] = np.sign(x[over])*(thr + (1-thr)*np.tanh((ax[over]-thr)/(1-thr)))
    return y

def limiter(x, ceil_db=-0.8, look_ms=5.0, rel_ms=90.0, block=128):
    ceil = db_to_lin(ceil_db); n = x.shape[0]; pad = (-n) % block
    if pad: x = np.pad(x, ((0, pad), (0, 0)))
    need = np.minimum(1.0, ceil/np.maximum(np.abs(x).max(1), 1e-9))
    need = minimum_filter1d(need, max(1, int(look_ms*1e-3*SR))+1)
    nb = x.shape[0]//block; g = need.reshape(nb, block).min(1)
    ar = np.exp(-block/max(1, rel_ms*1e-3*SR)); state, out = 1.0, np.empty(nb)
    for i in range(nb):
        state = g[i] if g[i] < state else ar*state + (1-ar)*g[i]
        out[i] = state
    gain = np.interp(np.arange(x.shape[0]), (np.arange(nb)+0.5)*block, out)
    return np.clip(x*gain[:, None], -ceil, ceil)[:n]
```

## 8. 响度迭代收敛（母带收尾）

```python
import pyloudnorm as pyln
meter = pyln.Meter(SR)
target = -7.9
for it in range(6):
    cur = meter.integrated_loudness(m)
    g = target - cur
    if abs(g) < 0.15: break
    m = limiter(softclip(m * db_to_lin(g)))
# 期望: 2–3 轮收敛。若第 3 轮增益仍 > 1dB, 检查软削波是否用错(见上)。
```

## 9. MP3 导出（无 ffmpeg 环境）

```python
import lameenc
def write_mp3(path, audio, sr=SR, kbps=320):
    pcm = (np.clip(audio, -1, 1)*32767).astype('<i2').tobytes()
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps); enc.set_in_sample_rate(sr)
    enc.set_channels(2); enc.set_quality(2)
    data = enc.encode(pcm) + enc.flush()
    open(path, 'wb').write(bytes(data))
```

## 10. 人声侧链避让（永远安全的保底）

```python
def vocal_duck(b, vocal, max_db=-1.6, att=0.020, rel=0.300, ref=0.10, blk=256):
    env = np.abs(vocal).max(1)
    pad = (-len(env)) % blk
    e = np.pad(env, (0, pad)).reshape(-1, blk).max(1)
    aa, ar = np.exp(-blk/(att*SR)), np.exp(-blk/(rel*SR))
    state, sm = 0.0, np.empty(len(e))
    for i in range(len(e)):
        coef = aa if e[i] > state else ar
        state = coef*state + (1-coef)*e[i]; sm[i] = state
    env_s = np.interp(np.arange(len(env)), (np.arange(len(e))+0.5)*blk, sm)
    duck = 1 + db_to_lin(max_db)*np.clip(env_s/ref, 0, 1) - np.clip(env_s/ref, 0, 1)
    # 等价写法: duck = 1 - (1-db_to_lin(max_db))*clip(env_s/ref,0,1)
    return b * duck[:, None]
```

# 无耳质检清单（QC Without Ears）

> SKILL.md 第 6 节的可运行版本。导出前逐项跑，全绿才许交货。
> 每一项都给出**判据 + 测法 + 实测案例数值**（案例来自 REMIX-AB-V5.1）。

## 总控脚本骨架

```python
import numpy as np, soundfile as sf
from scipy import signal as sg
import pyloudnorm as pyln

mix,  sr = sf.read('REMIX-AB-V5_Full_Remix.wav', dtype='float64')
stem, _  = sf.read('stems_project/REMIX-AB-V5_Vocal_Stem.flac', dtype='float64')
mm, vm = np.mean(mix, 1), np.mean(stem, 1)
rms = lambda x: 20*np.log10(max(np.sqrt(np.mean(x**2)), 1e-9))
```

## 1. 网格误差（< 2ms / 每切片）

构建脚本对每个切片打印 `grid_err_ms`。公式见 `grid_and_placement.md` §3。
案例：7 个切片全部 0.00ms。**任何一片 > 2ms = 锚点算错，返工。**

## 2. 接缝爆音（切点二阶差分 < 0.05）

```python
d2 = np.abs(vm[1:-1] - 0.5*(vm[:-2] + vm[2:]))
for t in cut_points:                      # 编曲表里的每个落点边界
    i = int(t*sr)
    print(t, d2[max(0,i-220):i+220].max())   # ±5ms 窗
```
判据：
- **切点处 < 0.05** → 无爆音（案例最大 0.037）。
- 乐句中部出现 0.3–0.5 的尖峰是**辅音瞬态**（塞擦音/爆破音），是正常甚至有益的
  （唇齿清晰度），不要误判。判断依据：位置在词首而非切点。
- 若切点处 > 0.1：淡入出没做够，或切进了非静音区。

## 3. 落点复测（与设计差 < 0.15s）

```python
def find_onset(t_lo, t_hi, thr=-42):
    for t in np.arange(t_lo, t_hi, 0.01):
        if rms(vm[int(t*sr):int(t*sr)+int(0.03*sr)]) > thr:
            return round(t, 2)
```
- 对每段设计"期望区间"再测。塞擦音起音的字（曾/秋/陪）实测比感知起音早 0.1–0.15s，
  期望区间要相应放宽。
- **连唱交界没有 onset**（零气口），改测"交界两侧帧能量 > -20dB"确认连唱完整保住了。

## 4. 空洞与回声区

```python
# 计划内的纯器乐区: 分轨应接近死寂(混响尾 <-45dB 亦可)
print(rms(vm[int(93.5*sr):int(96.3*sr)]))    # 案例 -95.9 dB ✓
# 计划内的回声区: 应清晰可闻但不抢戏
print(rms(vm[int(87.0*sr):int(92.5*sr)]))    # 案例 -28.9 dB ✓ (回声, 对照 -95.9)
```
经验区间：回声/尾音区 **-25 ~ -31 dB**；若是 < -60dB = 效果没发出来，查 send 与增益。

## 5. 段落电平平衡表

```python
for name, t0, t1 in sections:      # sections 来自编曲表
    print(f'{name:10s} mix {rms(mm[t0*sr:t1*sr]):6.1f} | vocal {rms(vm[t0*sr:t1*sr]):6.1f}')
```
判据（注意静音稀释 RMS，用乐句段近似）：
- Drop 段人声低于全混 **3–6dB**（案例 -4.7~-5.5）。
- Breakdown 段可到 **2–4dB**（床安静，人声是主角）。
- 主歌段看起来会低 8–11dB（乐句间隙稀释），换算乐句段后应在 4–8dB 内。
- 全段 > -1dB（人声盖过一切）或 < -15dB（人声消失）都是事故。

## 6. 频谱合理性

```python
f, P = sg.welch(vm, sr, nperseg=8192)
for lo, hi in [(0,120),(120,500),(500,2000),(2000,6000),(6000,16000)]:
    m = (f>=lo)&(f<hi); print(lo, hi, round(100*P[m].sum()/P.sum(),1))
```
人声分轨健康值：120–500 ≈ 20–25%、500–2000 ≈ 55–65%、2–6k ≈ 8–12%、>6k ≈ 2–3%。
特殊处理段（滤波告别）用**谱重心**验收：亮版 1060Hz → 暗版 525Hz、高频段 -9.3dB = 合格。

## 7. 动态设计元素抽查

- **Drop 前退场**：`rms(mm[drop-0.35s : drop-0.05s])` 应比 `rms(mm[drop-1.35s : drop-1.05s])`
  低 2–4dB（注意用无人声窗口测，人声会干扰读数）。
- **Sub 冲击**：25–70Hz 带通能量，冲击后 1s vs 前 1s 应高 **>8dB**（案例 +10.7）。
- **开场 riser**：开场区 mix RMS 应明显高于纯伴奏同期（有能量爬升）。

## 8. 母带指标

```python
meter = pyln.Meter(sr)
print(meter.integrated_loudness(mix), 20*np.log10(np.max(np.abs(mix))))
```
- LUFS 在目标 ±0.3 内（俱乐部向 -7.5~-8.5）。
- 真峰 ≤ -0.8dBTP（导出 16-bit 后再测一次，防止转换溢出）。
- 时长与伴奏严格一致（1:1 网格，不多不少）。

## 9. 文件完整性

- WAV 16-bit / MP3 320kbps 头合法（`fffb` 或 `ID3`）；试听段 30–50s；
- 元数据 JSON 的 cue 表与最终脚本打印一致；
- 构建脚本从头跑一遍能原样复现全部输出（幂等）。

## 10. 交付前最后一次"用眼睛听"

把编曲说明里的段落表按时间轴重读一遍，问自己三个问题：
1. 这条能量曲线如果画出来，有没有突然的断崖（人声/密度/动态）？
2. 每一句歌词的结尾有没有着落（下一段、回声、或明确的留白设计）？
3. 如果我是个只能看频谱图的人，这张图讲的故事和我的编曲概念一致吗？
三个都答"是"，交货。

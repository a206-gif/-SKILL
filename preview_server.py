#!/usr/bin/env python3
"""Live Preview Web Server for REMIX-AB-V4 Deliverables (binds 0.0.0.0:8080)"""
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = "/home/user/-SKILL"
os.chdir(ROOT)


class ThreadingHTTPServer(
    __import__("socketserver").ThreadingMixIn, HTTPServer
):
    daemon_threads = True


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Supports HTTP/1.1 Range requests so HTML5 <audio> scrubbing works seamlessly."""

    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Connection", "close")
        super().end_headers()

    def do_HEAD(self):
        self._handle_request(send_body=False)

    def do_GET(self):
        self._handle_request(send_body=True)

    def _handle_request(self, send_body: bool = True):
        if self.path in ("/", "/index.html"):
            self.send_Index(send_body=send_body)
            return
        range_header = self.headers.get("Range")
        if not range_header:
            return super().do_GET() if send_body else super().do_HEAD()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        file_size = os.path.getsize(path)
        byte_range = range_header.strip().split("=")[1].split(",")[0]
        start_s, end_s = byte_range.split("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)

        if start >= file_size or start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if send_body:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def send_Index(self, send_body: bool = True):
        meta_path = os.path.join(ROOT, "stems_project", "REMIX-AB-V4_Project_Metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        lufs_val = meta.get("master_metrics", {}).get("integrated_lufs", -7.81)
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>REMIX-AB-V4｜A Vocal × B DJ Remix 交付试听台</title>
<style>
  :root {{
    --bg: #0b0d14;
    --card: #141826;
    --border: #262d45;
    --accent: #6c5ce7;
    --accent2: #00cec9;
    --text: #eef1f8;
    --muted: #9aa3ba;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", sans-serif;
    background: radial-gradient(circle at top right, #1b1f38 0%, var(--bg) 60%);
    color: var(--text);
    line-height: 1.6;
  }}
  .container {{ max-width: 920px; margin: 0 auto; }}
  .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    background: rgba(108,92,231,0.2); border: 1px solid var(--accent);
    color: #b8afff; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
  }}
  h1 {{ margin: 12px 0 6px; font-size: 26px; }}
  .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .stat {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }}
  .stat .k {{ font-size: 12px; color: var(--muted); }}
  .stat .v {{ font-size: 17px; font-weight: 700; margin-top: 4px; color: var(--accent2); }}
  .track-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; margin-bottom: 14px;
  }}
  .track-card.primary {{
    border-color: var(--accent);
    box-shadow: 0 0 24px rgba(108,92,231,0.18);
  }}
  .track-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
  .track-title {{ font-weight: 700; font-size: 16px; }}
  .track-desc {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; }}
  audio {{ width: 100%; height: 40px; outline: none; }}
  .dl-btn {{
    text-decoration: none; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600;
    background: var(--accent); color: white; transition: opacity 0.15s;
  }}
  .dl-btn:hover {{ opacity: 0.88; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
  <span class="badge">PROJECT: REMIX-AB-V4</span>
  <h1>A Vocal × B DJ Remix 制作成品与分轨验收台</h1>
  <div class="sub">原唱人声分离移植（UVR-MDX-NET-Voc_FT） + DJ 伴奏提纯（UVR-MDX-NET-Inst_HQ_5） + Rubberband 保共振峰变速变调与混音母带</div>

  <div class="grid">
    <div class="stat"><div class="k">音乐 A 实测速度 / 调性</div><div class="v">130.00 BPM · F大调/D小调</div></div>
    <div class="stat"><div class="k">音乐 B 实测速度 / 调性</div><div class="v">143.63 BPM · F#大调/D#小调</div></div>
    <div class="stat"><div class="k">人声拉伸比 / 音高微调</div><div class="v">×1.1048 · +0.67 半音</div></div>
    <div class="stat"><div class="k">成品响度 / 峰值限制</div><div class="v">{lufs_val} LUFS · -0.50 dBFS</div></div>
  </div>

  <div class="track-card primary">
    <div class="track-header">
      <div class="track-title">1. 完整成品：REMIX-AB-V4_Full_DJ_Remix (02:59)</div>
      <div>
        <a class="dl-btn" href="/REMIX-AB-V4_Full_DJ_Remix.mp3" download>下载 MP3 (320k)</a>
        <a class="dl-btn" style="background:#0097a7;margin-left:6px;" href="/REMIX-AB-V4_Full_DJ_Remix.wav" download>下载无损 WAV</a>
      </div>
    </div>
    <div class="track-desc">开场 0.00–1.78s 采用 B 原始标志性鼓点，1.78s 起 A 精选副歌「曾经是心心念念…」零延迟切入 B 的 DJ 主旋律，含三段高能量 Drop 与 DJ 式收尾。</div>
    <audio controls preload="metadata" src="/REMIX-AB-V4_Full_DJ_Remix.mp3"></audio>
  </div>

  <div class="track-card">
    <div class="track-header">
      <div class="track-title">2. 试听预览版：REMIX-AB-V4_Preview_Opening_And_Drop (00:56)</div>
      <a class="dl-btn" href="/REMIX-AB-V4_Preview_Opening_And_Drop.mp3" download>下载预览 MP3</a>
    </div>
    <div class="track-desc">快速验收开场设计（B 开场鼓点 → A 副歌立即切入 → 情绪铺垫 → 第一段完整高潮 Drop）。</div>
    <audio controls preload="metadata" src="/REMIX-AB-V4_Preview_Opening_And_Drop.mp3"></audio>
  </div>

  <div class="track-card">
    <div class="track-header">
      <div class="track-title">3. 分离后的原唱人声：A_Vocals.wav</div>
      <a class="dl-btn" href="/A_Vocals.wav" download>下载 A_Vocals.wav</a>
    </div>
    <div class="track-desc">由音乐 A 经 UVR-MDX-NET-Voc_FT 深度分离并清除原曲伴奏低频与间奏底噪后的纯净原唱人声。</div>
    <audio controls preload="none" src="/A_Vocals.wav"></audio>
  </div>

  <div class="track-card">
    <div class="track-header">
      <div class="track-title">4. 处理后的 DJ 伴奏：B_Instrumental.wav</div>
      <a class="dl-btn" href="/B_Instrumental.wav" download>下载 B_Instrumental.wav</a>
    </div>
    <div class="track-desc">由音乐 B 经 UVR-MDX-NET-Inst_HQ_5 移除 B 原有男声/Rap、完整保留标志性开场鼓点、Bass 与合成器主旋律的纯净 DJ 伴奏。</div>
    <audio controls preload="none" src="/B_Instrumental.wav"></audio>
  </div>

  <div class="track-card">
    <div class="track-header">
      <div class="track-title">5. 工程对齐人声分轨：stems_project/Remix_Aligned_A_Vocals_Stem.flac</div>
      <a class="dl-btn" href="/stems_project/Remix_Aligned_A_Vocals_Stem.flac" download>下载对齐无损分轨 FLAC</a>
    </div>
    <div class="track-desc">已完成 143.63 BPM 节拍拉伸、+0.67 半音和声适配与混响延迟处理的 Remix 人声独立无损工程轨，可与 B_Instrumental.wav 1:1 同步导入任意 DAW 二次微调。</div>
    <audio controls preload="none" src="/stems_project/Remix_Aligned_A_Vocals_Stem.flac"></audio>
  </div>

  <div class="track-card primary">
    <div class="track-header">
      <div class="track-title">6. 供其他模型编曲测试：全曲预适配纯干声 & 工程坐标包</div>
      <div>
        <a class="dl-btn" href="/handoff_kit_for_other_models/A_Vocals_Full_Adapted_143.63BPM_Fsharp.flac" download>下载 143.63BPM 预适配全曲干声 (FLAC)</a>
        <a class="dl-btn" style="background:#0097a7;margin-left:6px;" href="/handoff_kit_for_other_models/ARRANGEMENT_SPEC.json" download>下载编曲坐标 JSON</a>
        <a class="dl-btn" style="background:#2d3436;margin-left:6px;" href="/handoff_kit_for_other_models/README_HANDOFF_KIT.md" download>下载交接说明 MD</a>
      </div>
    </div>
    <div class="track-desc">已将整首 A_Vocals.wav 预先保共振峰拉伸至 143.63 BPM 并对齐 F#大调（纯干声无混响），附带精确小节网格、歌词时间轴与 &lt;-40dB 安全静音切点表，可直接喂给其他模型自由测试编曲能力！</div>
    <audio controls preload="none" src="/handoff_kit_for_other_models/A_Vocals_Full_Adapted_143.63BPM_Fsharp.flac"></audio>
  </div>
</div>
</body>
</html>"""
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8080), RangeHTTPRequestHandler)
    print("Serving REMIX-AB-V4 Preview Studio on http://0.0.0.0:8080", flush=True)
    server.serve_forever()

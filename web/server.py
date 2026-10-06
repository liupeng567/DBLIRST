"""DSLD 检视台本地后端（可选，零第三方依赖，仅用标准库 http.server）。

用途
----
web/index.html 在 file:// 下也能完整交互（图像走浏览器目录选择、检测走前端
Top-Hat 引擎）。本脚本提供两条额外能力，二者都需要它：

  ① 按**磁盘路径**读图：浏览器沙箱不允许读任意本地路径，检测检视常常需要
     直接指向 D:\\DBLIRST\\data\\cache\\ittd\\seq_XXXX 下的缓存帧；
  ② 服务端 **torch 权重推理**：前端只上传图像字节 + 权重 ID，真正的前向在
     本机 dsld.models（MSHNet / T-MSD3D）里跑，与 scripts/eval_baseline.py
     同口径（归一化走 dsld.data.preprocess.normalize，框走 mask_to_boxes）。

启动：
    python web/server.py                 # 默认 127.0.0.1:8765
    python web/server.py --port 9000 --root D:/some/dir

安全边界：默认只监听回环；路径参数一律限定在 --root 之内（路径逃逸防护）；
权重按内容哈希缓存，不落盘。torch 缺失时 /api/infer 返回 error，前端自动
回退本地引擎，功能不中断。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import posixpath
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMG_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
GT_EXT = {".xml", ".txt", ".json"}

# 权重缓存：weight_id → (model, cfg, device, meta)
_WEIGHTS: dict[str, dict] = {}
_LOCK = threading.Lock()
_OPTIONS: dict = {"root": REPO, "device": "auto", "max_upload": 96 << 20}


# ───────────────────────────── 路径工具 ─────────────────────────────


def safe_join(root: Path, rel: str) -> Path:
    """把外部传入的路径限制在 root 之内；越界直接拒绝。"""
    rel = (rel or "").strip().strip('"').replace("\\", "/")
    if not rel:
        raise ValueError("空路径")
    p = Path(rel)
    cand = (p if p.is_absolute() else root / rel).resolve()
    try:
        cand.relative_to(root.resolve())
    except ValueError:
        # 允许用户直接输入盘符绝对路径（仅限本机回环服务），但仍做存在性校验
        if not cand.exists():
            raise ValueError(f"路径不存在或越界: {rel}")
    return cand


def list_dir(root: Path, d: str) -> dict:
    target = safe_join(root, d)
    dirs, files = [], []
    if not target.is_dir():
        return {"dir": str(target), "dirs": dirs, "files": files, "parents": parents_of(target)}
    for e in sorted(target.iterdir(), key=lambda p: p.name):
        if e.is_dir():
            dirs.append(e.name + ("/" if os.name != "nt" else "\\"))
        elif e.suffix.lower() in IMG_EXT:
            files.append(e.name)
    return {"dir": str(target), "dirs": dirs, "files": files, "parents": parents_of(target)}


def parents_of(p: Path, limit: int = 6) -> list[str]:
    out, cur = [], p.parent
    for _ in range(limit):
        s = str(cur)
        if cur.parent == cur:
            break
        out.append(s)
        cur = cur.parent
    return out


def resolve_paths(root: Path, paths: list[str], dir_: str = "") -> dict:
    """路径 → 前端可直接 <img src> 的 data URL（base64 内联，绕开沙箱限制）。"""
    items = []
    for raw in paths:
        try:
            if raw == "*":
                base = safe_join(root, dir_)
                cand_list = [p for p in sorted(base.iterdir()) if p.suffix.lower() in IMG_EXT]
            else:
                cand_list = [safe_join(root, raw)]
            for p in cand_list:
                if p.suffix.lower() not in IMG_EXT:
                    items.append({"path": str(p), "name": p.name, "ok": False, "error": "非图像"})
                    continue
                data = p.read_bytes()
                items.append({
                    "path": str(p),
                    "name": p.name,
                    "ok": True,
                    "size": len(data),
                    "url": "data:image/%s;base64,%s"
                           % (("jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else
                               ("bmp" if p.suffix.lower() == ".bmp" else "png")),
                              base64.b64encode(data).decode("ascii")),
                })
        except Exception as ex:  # noqa: BLE001
            items.append({"path": str(raw), "name": Path(str(raw)).name, "ok": False,
                          "error": str(ex)})
    return {"items": items}


def read_gt(root: Path, ann_dir: str, img_dir: str, limit: int = 400) -> dict:
    """按 ITTD 官方布局读 GT：Annotation/{v}/*.xml  或 Evaluation/cm_GT/*.txt。"""
    a_dir = safe_join(root, ann_dir)
    i_dir = safe_join(root, img_dir)
    frames: dict[int, list] = {}
    seq = int(a_dir.name) if a_dir.name.isdigit() else -1

    xmls = sorted(a_dir.glob("*.xml"))
    if xmls:
        import xml.etree.ElementTree as ET

        for x in xmls[:limit]:
            stem = x.stem
            if not stem.isdigit():
                continue
            root_el = ET.parse(x).getroot()
            boxes = []
            for obj in root_el.findall("object"):
                b = obj.find("bndbox")
                if b is None:
                    continue
                boxes.append({
                    "x1": int(float(b.findtext("xmin"))), "y1": int(float(b.findtext("ymin"))),
                    "x2": int(float(b.findtext("xmax"))), "y2": int(float(b.findtext("ymax"))),
                    "track_id": int(obj.findtext("name") or -1),
                })
            frames[int(stem)] = boxes
        return {"seq": seq, "frames": frames, "source": "voc_xml", "n": len(frames)}

    txts = sorted(a_dir.glob("*.txt"))
    for t in txts[:1]:
        lines = t.read_text(encoding="utf-8").splitlines()
        if not lines or not re.match(r"^targetnum:\s*\d+$", lines[0].strip()):
            continue
        seq_m = re.search(r"(\d+)", t.stem)
        if seq_m:
            seq = int(seq_m.group(1))
        for line in lines[1:]:
            m = re.match(r"^frame:(\d+)\s+(\d+)(.*)$", line.strip())
            if not m:
                continue
            fno, n_obj, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", rest)]
            boxes = []
            if n_obj and len(nums) == 5 * n_obj:
                for k in range(0, len(nums), 5):
                    boxes.append({"x1": nums[k + 1], "y1": nums[k + 2], "x2": nums[k + 3],
                                  "y2": nums[k + 4], "track_id": int(nums[k])})
            elif n_obj and len(nums) == 3 * n_obj:
                for k in range(0, len(nums), 3):
                    x, y = nums[k + 1], nums[k + 2]
                    boxes.append({"x1": x, "y1": y, "x2": x, "y2": y, "track_id": int(nums[k]),
                                  "point": True})
            frames[fno] = boxes
        return {"seq": seq, "frames": frames, "source": "official_txt", "n": len(frames)}

    return {"seq": seq, "frames": frames, "source": "none", "n": 0,
            "hint": f"{a_dir} 下未找到 *.xml 或 cm_GT txt"}


# ───────────────────────────── 推理（可选 torch） ─────────────────────────────


def pick_device() -> str:
    if _OPTIONS["device"] != "auto":
        return _OPTIONS["device"]
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def load_weight(weight_id: str) -> dict:
    """weight_id = 'path:<绝对路径>' 或 'blob:<sha256>'（前者直读，后者用上传缓存）。"""
    with _LOCK:
        if weight_id in _WEIGHTS:
            return _WEIGHTS[weight_id]

    kind, _, key = weight_id.partition(":")
    if kind == "path":
        ckpt_path = safe_join(_OPTIONS["root"], key)
    elif kind == "blob":
        ckpt_path = _BLOBS.get(key)
        if ckpt_path is None:
            raise RuntimeError("权重会话已失效，请重新加载")
    else:
        raise RuntimeError(f"未知 weight id: {weight_id}")

    import torch  # noqa: PLC0415

    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf  # noqa: PLC0415

    from dsld.train.trainer import build_model  # noqa: PLC0415

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = state.get("cfg")
    cfg = OmegaConf.create(cfg) if cfg is not None else None
    sd = state.get("model", state)
    if cfg is None:
        raise RuntimeError("ckpt 缺 cfg 字段，无法重建模型")
    model = build_model(cfg)
    model.load_state_dict(sd, strict=True)
    device = pick_device()
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    rec = {
        "model": model,
        "cfg": cfg,
        "device": device,
        "meta": {
            "file": Path(ckpt_path).name,
            "size": ckpt_path.stat().st_size,
            "model_type": cfg.model.type,
            "params_M": round(n_params / 1e6, 3),
            "epoch": state.get("epoch", -1),
            "device": device,
        },
    }
    with _LOCK:
        _WEIGHTS[weight_id] = rec
    return rec


_BLOBS: dict[str, Path] = {}


def decode_image(data_url: str):
    import numpy as np  # noqa: PLC0415

    from dsld.data.preprocess.normalize import frame_stats, normalize_frame  # noqa: PLC0415

    head, _, payload = data_url.partition(",")
    ext = "png" if "png" in head else "jpeg"
    buf = io.BytesIO(base64.b64decode(payload))
    try:
        import cv2  # noqa: PLC0415

        raw = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:  # noqa: BLE001 —— 无 cv2 时退回 PIL
        from PIL import Image  # noqa: PLC0415

        raw = np.asarray(Image.open(buf).convert("L"))
    if raw is None:
        raise RuntimeError("图像解码失败")

    med, sigma = frame_stats(raw)
    x = normalize_frame(raw.astype(np.float32), med, sigma)
    return x, raw.shape[:2]


def infer(weight_id: str, data_url: str, thr: float, min_area: int, max_boxes: int,
          model_type: str = "auto") -> dict:
    """单帧推理入口。

    注意 T-MSD3D 的多尺度帧差通道在单帧下恒为 0（无历史帧可差），
    此处仅供检视定位用；正式的时序口径走 dsld/eval/infer_seq.infer_temporal
    （T=32、步距 24、前 8 帧预热、α 图平均）。
    """
    rec = load_weight(weight_id)
    model, cfg, device = rec["model"], rec["cfg"], rec["device"]

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from dsld.eval.mask_to_boxes import mask_to_boxes  # noqa: PLC0415

    x, (h, w) = decode_image(data_url)
    t0 = time.perf_counter()
    with torch.no_grad():
        xt = torch.from_numpy(x)[None, None].to(device)  # [1,1,1,H,W]（T=1 逐帧入口）
        logits = model(xt)
        prob = torch.sigmoid(logits.float())
        if prob.dim() == 5:
            prob = prob[0, -1, 0]
        else:
            prob = prob[0, 0]
        prob = prob.cpu().numpy()
    boxes = mask_to_boxes(prob, frame=1, thr=float(thr), min_area=int(min_area))
    boxes.sort(key=lambda b: -b.score)
    if max_boxes and len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]
    return {
        "boxes": [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "score": b.score,
                   "track_id": b.track_id} for b in boxes],
        "ms": (time.perf_counter() - t0) * 1000.0,
        "w": int(w), "h": int(h),
        "model_type": str(cfg.model.type) if model_type == "auto" else model_type,
        "device": device,
    }


# ───────────────────────────── HTTP ─────────────────────────────


class Handler(BaseHTTPRequestHandler):
    server_version = "DSLDViewer/1.0"
    protocol_version = "HTTP/1.1"

    # -- 工具 --
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _static(self, rel: str) -> None:
        web = Path(__file__).resolve().parent
        rel = rel.lstrip("/") or "index.html"
        p = (web / rel).resolve()
        try:
            p.relative_to(web)
        except ValueError:
            return self._json({"error": "越界"}, 403)
        if p.is_dir():
            p = p / "index.html"
        if not p.is_file():
            return self._json({"error": "not found: " + rel}, 404)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in {"application/javascript", "application/json"}:
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > _OPTIONS["max_upload"]:
            raise ValueError(f"请求体过大 {n} B")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        sys.stderr.write("[viewer] %s - %s\n" % (self.address_string(), fmt % args))

    # -- GET --
    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/api/health":
            info = {"ok": True, "root": str(_OPTIONS["root"]), "repo": str(REPO),
                    "weights": len(_WEIGHTS)}
            try:
                import torch  # noqa: PLC0415

                info["torch"] = torch.__version__
                info["cuda"] = bool(torch.cuda.is_available())
                info["device"] = pick_device()
                info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else None
            except Exception as ex:  # noqa: BLE001
                info["torch"] = None
                info["torch_error"] = str(ex)
            return self._json(info)
        if u.path == "/api/gt":
            try:
                return self._json(read_gt(_OPTIONS["root"],
                                          (q.get("ann") or [""])[0],
                                          (q.get("img") or [""])[0]))
            except Exception as ex:  # noqa: BLE001
                return self._json({"error": str(ex), "frames": {}}, 200)
        return self._static(u.path)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    # -- POST --
    def do_POST(self) -> None:  # noqa: BLE001
        u = urllib.parse.urlparse(self.path)
        try:
            body = self._body()
        except Exception as ex:  # noqa: BLE001
            return self._json({"error": "请求体解析失败: " + str(ex)}, 400)

        try:
            if u.path == "/api/list":
                return self._json(list_dir(_OPTIONS["root"], body.get("dir", "")))
            if u.path == "/api/resolve":
                return self._json(resolve_paths(_OPTIONS["root"], body.get("paths") or [],
                                                body.get("dir", "")))
            if u.path == "/api/weight":
                return self._json(self._weight(body))
            if u.path == "/api/infer":
                wid = body.get("weight") or ""
                if not wid:
                    return self._json({"error": "未指定权重 id"}, 200)
                res = infer(wid, body.get("image", ""), float(body.get("thr", 0.5)),
                            int(body.get("min_area", 4)), int(body.get("max_boxes", 0)),
                            body.get("model_type", "auto"))
                return self._json(res)
        except Exception as ex:  # noqa: BLE001
            return self._json({"error": str(ex)}, 200)

        return self._json({"error": "unknown endpoint " + u.path}, 404)

    @staticmethod
    def _weight(body: dict) -> dict:
        import tempfile  # noqa: PLC0415

        if body.get("path"):
            p = safe_join(_OPTIONS["root"], body["path"])
            if not p.is_file():
                raise FileNotFoundError(str(p))
            rec = load_weight("path:" + str(p))
            return {"ok": True, "id": "path:" + str(p), "info": rec["meta"]}

        name = body.get("name") or "weight.pt"
        data = base64.b64decode(body.get("data") or "")
        if not data:
            raise ValueError("空权重数据")
        digest = hashlib.sha256(data).hexdigest()[:32]
        with _LOCK:
            hit = _BLOBS.get(digest)
            if hit is None or not hit.exists():
                fd, tmp = tempfile.mkstemp(prefix="dsld_", suffix=Path(name).suffix or ".pt")
                with os.fdopen(fd, "wb") as fp:
                    fp.write(data)
                hit = Path(tmp)
                _BLOBS[digest] = hit
        rec = load_weight("blob:" + digest)
        return {"ok": True, "id": "blob:" + digest, "info": rec["meta"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="DSLD 检视台本地后端（可选）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=str(REPO), help="路径参数允许的根目录")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    _OPTIONS["root"] = Path(args.root).resolve()
    _OPTIONS["device"] = args.device
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    mimetypes.add_type("image/bmp", ".bmp")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    dev = pick_device()
    print("=" * 72)
    print("DSLD 检视台后端")
    print(f"  地址      http://{args.host}:{args.port}/")
    print(f"  根目录    {_OPTIONS['root']}")
    print(f"  推理设备  {dev}")
    print("  浏览器打开上面地址即可；Ctrl+C 退出")
    print("=" * 72)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
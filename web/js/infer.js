/* DSLD 检视台 —— 推理后端。
 *
 * 三种引擎，按可用性自动降级（顶栏「引擎」chip 始终显示当前实际使用者）：
 *   1. local : 纯前端图像处理 —— 局部对比度 Top-Hat + 显著性阈值 + 8 连通域，
 *              产出与 dsld/eval/mask_to_boxes.py 同构的框（x1,y1,x2,y2,score）。
 *              无需后端、无需权重，用于离线检视与交互自检。
 *   2. weight: 用户选择的权重文件经 /api/weight 上传（由 web/server.py 缓存），
 *              之后每帧图像只送字节、按权重 id 走 /api/infer。
 *   3. server: 用户给出权重磁盘路径，服务端 torch.load + dsld.models 真正前向。
 *
 * 权重文件本体始终由用户显式选择（菜单「模型 → 加载权重」），前端不猜测路径。
 */
window.DSLD = window.DSLD || {};
(function (ns) {
  'use strict';

  const IMG_RE = /\.(bmp|png|jpg|jpeg|tif|tiff|webp)$/i;
  const IMG_EXT = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'];

  function isImageName(name) { return IMG_RE.test(name); }

  /* 从任意路径串里取帧号：优先尾部纯数字文件名，其次路径中的数字段。 */
  function frameNoOf(path) {
    const base = String(path).split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
    if (/^\d+$/.test(base)) return parseInt(base, 10);
    const m = String(path).match(/(\d+)(?=[^0-9]*$)/);
    return m ? parseInt(m[1], 10) : -1;
  }

  /* 序列标识 = 路径里最像序列目录的那一段（原样保留，如 seq_0031 / "12"）；
     GT 键比较由调用方用 seqNumOf 抽数字段完成，故此处不做归一。 */
  function seqOf(path) {
    const parts = String(path).split(/[\\/]/).filter(Boolean);
    for (let i = parts.length - 1; i >= 0; i -= 1) {
      if (/^(seq[_-]?\d{1,4}|\d{1,4})$/i.test(parts[i])) return parts[i];
    }
    return '—';
  }

  /* ───────────────────────── 本地（无后端）推理 ───────────────────────── */

const R_TH = 3;      // 背景估计半窗（Top-Hat）
const MIN_AREA = 3;  // 连通域最小面积默认值 px²（与 mask_to_boxes.min_area 同量级）

  function toGray(img) {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth || img.width;
    c.height = img.naturalHeight || img.height;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const n = c.width * c.height;
    const out = new Float32Array(n);
    for (let i = 0, j = 0; i < n; i += 1, j += 4) {
      out[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
    }
    return { data: out, w: c.width, h: c.height };
  }

  /* 可分离盒式均值（边界用 clamp 复制） */
  function boxMean(src, w, h, r) {
    const tmp = new Float32Array(w * h);
    const out = new Float32Array(w * h);
    const inv = 1 / (2 * r + 1);
    for (let y = 0; y < h; y += 1) {
      const row = y * w;
      let acc = 0;
      for (let x = -r; x <= r; x += 1) acc += src[row + Math.min(w - 1, Math.max(0, x))];
      for (let x = 0; x < w; x += 1) {
        tmp[row + x] = acc * inv;
        acc += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.min(w - 1, Math.max(0, x - r))];
      }
    }
    for (let x = 0; x < w; x += 1) {
      let acc = 0;
      for (let y = -r; y <= r; y += 1) acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
      for (let y = 0; y < h; y += 1) {
        out[y * w + x] = acc * inv;
        acc += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.min(h - 1, Math.max(0, y - r)) * w + x];
      }
    }
    return out;
  }

  /* 8 连通域标记（显式栈，避免递归爆栈） */
  function components(mask, w, h, minArea) {
    const lab = new Int32Array(w * h);
    const stack = new Int32Array(w * h);
    const boxes = [];
    let cur = 0;
    for (let s = 0; s < mask.length; s += 1) {
      if (!mask[s] || lab[s]) continue;
      cur += 1;
      let sp = 0;
      stack[sp++] = s;
      lab[s] = cur;
      let x1 = w, y1 = h, x2 = -1, y2 = -1, area = 0, sum = 0;
      while (sp > 0) {
        const p = stack[--sp];
        const py = (p / w) | 0;
        const px = p - py * w;
        area += 1;
        sum += mask[p] > 1 ? mask[p] : 0;
        if (px < x1) x1 = px;
        if (py < y1) y1 = py;
        if (px > x2) x2 = px;
        if (py > y2) y2 = py;
        for (let dy = -1; dy <= 1; dy += 1) {
          const ny = py + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx += 1) {
            if (!dx && !dy) continue;
            const nx = px + dx;
            if (nx < 0 || nx >= w) continue;
            const q = ny * w + nx;
            if (mask[q] && !lab[q]) { lab[q] = cur; stack[sp++] = q; }
          }
        }
      }
      if (area >= minArea) boxes.push({ x1, y1, x2, y2, area, score: sum / area });
    }
    return boxes;
  }

  /* 局部对比度显著性：Top-Hat = 原图 − 邻域均值；score 归一到 [0,1]。 */
  function detectLocal(img, opt) {
    const t0 = performance.now();
    const g = toGray(img);
    const { data, w, h } = g;
    const bg = boxMean(data, w, h, R_TH);
    const res = new Float32Array(w * h);
    let mx = 0;
    for (let i = 0; i < res.length; i += 1) {
      const v = data[i] - bg[i];
      res[i] = v > 0 ? v : 0;
      if (res[i] > mx) mx = res[i];
    }
    const sigma = Math.max(0.35 * mx, 6);
    const thr = Math.max(opt.thr * sigma, 8);
    const mask = new Float32Array(w * h);
    for (let i = 0; i < res.length; i += 1) mask[i] = res[i] >= thr ? res[i] : 0;

    const minArea = Math.max(1, opt.minArea || MIN_AREA);
    let boxes = components(mask, w, h, minArea);
    boxes = boxes.map((b) => ({
      x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
      score: Math.min(0.999, 0.35 + 0.65 * ((b.score / (sigma * 2.2)) || 0)),
      track_id: -1,
    }));
    boxes.sort((a, b) => b.score - a.score);
    if (opt.maxBoxes > 0 && boxes.length > opt.maxBoxes) boxes = boxes.slice(0, opt.maxBoxes);
    return { boxes, ms: performance.now() - t0, w, h };
  }

  /* ───────────────────────── 服务端引擎 ───────────────────────── */

  async function detectServer(img, opt) {
    const t0 = performance.now();
    const dataUrl = imgToDataUrl(img);
    const r = await fetch('api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        weight: opt.weight || '',
        thr: opt.thr,
        min_area: opt.minArea,
        max_boxes: opt.maxBoxes,
        model_type: opt.modelType || 'auto',
      }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    const boxes = (j.boxes || []).map((b) => ({
      x1: b.x1 | 0, y1: b.y1 | 0, x2: b.x2 | 0, y2: b.y2 | 0,
      score: +b.score || 0, track_id: b.track_id === undefined ? -1 : b.track_id | 0,
    }));
    return { boxes, ms: j.ms || (performance.now() - t0), w: j.w, h: j.h, backend: true };
  }

  function imgToDataUrl(img) {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth || img.width;
    c.height = img.naturalHeight || img.height;
    c.getContext('2d').drawImage(img, 0, 0);
    return c.toDataURL('image/png');
  }

  ns.Infer = {
    IMG_EXT, isImageName, frameNoOf, seqOf,
    toGray, boxMean, components, detectLocal, imgToDataUrl,
    detectServer,
    ENGINE_LABEL: {
      server: '服务端模型',
      demo: '演示 Top-Hat（非模型）',
    },
  };
})(window.DSLD);
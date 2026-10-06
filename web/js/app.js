/* DSLD 检视台 —— 主程序。
 *
 * 布局契约见 css/app.css：--side-w 夹在 [132px, 20vw]；
 *   菜单栏 → 左侧图像路径栏（≤20%）→ 右侧画布 stage（原图 / 检测 / GT / 三联）→ 底部播放条
 *   右下角 .readout 固定尺寸显示鼠标所在面板的像素坐标与命中目标框。
 *
 * 检测状态跨帧保持：state.det.cache 以 (帧, 权重, 参数) 为键缓存框，
 * 切帧不重置权重与阈值；已算过的帧回看零延迟。
 */
(function (ns) {
  'use strict';

  const { $, $$, el, clamp, toast, modal, confirmBox, download, copyText, fmtBytes, fmtMs,
    fmtTime, stamp, natCmp, debounce, savePref, imgLoad, canvasToBlob, Infer } = ns;
  const { isImageName, frameNoOf, seqOf, detectLocal, detectServer, imgToDataUrl } = Infer;

  const GAP = 10;            // 面板间距（与 app.css .panel + .panel 一致）
  const CAP_H = 22;          // 导出对比图标题条高度
  const GRAY_CACHE = 6;      // 像素读数用的灰度图缓存条数

  const state = {
    frames: [],              // {id, path, name, url, group, frameNo, img, w, h}
    filter: '',
    index: -1,
    playList: [],
    playing: false,
    timer: 0,
    fps: savePref('fps') || 8,
    loop: true,

    view: savePref('view') || 'trio',
    zoom: { scale: 1, tx: 0, ty: 0, fit: true },
    show: Object.assign({ det: true, gt: true, center: true, label: true, track: true },
      savePref('show') || {}),

    weights: null,           // {id, name, size, source:'file'|'server-path', bytes}
    /* localFallback=false：未加载权重时**不做任何检测**，检测面板显式留空，
       避免把前端 Top-Hat 结果误读成模型输出；需要离线自检时手动开启。 */
    params: Object.assign({ thr: 0.35, minArea: 3, maxBoxes: 60, localFallback: false },
      savePref('params') || {}),
    det: { cache: new Map(), order: [], busy: 0, lastErr: '' },

    gt: new Map(),           // `${seq}|${frameNo}` → [{x1,y1,x2,y2,track_id}]
    gtByName: new Map(),     // basename → boxes
    grayCache: new Map(),   // frameId → {data,w,h}

    api: { ok: false, info: null, tried: false },
    inflight: new Set(),
  };

  /* ───────────────────────────── 菜单 ───────────────────────────── */

  function menuDefs() {
    return [
      {
        label: '文件',
        items: [
          { label: '加载图像文件…', kbd: 'Ctrl+O', act: () => pick('#pick-images', onPickFiles) },
          { label: '加载图像文件夹…', kbd: 'Ctrl+Shift+O', act: () => pick('#pick-folder', onPickFolder) },
          { label: '加载图像路径清单…', kbd: 'Ctrl+L', act: () => pick('#pick-list', onPickList) },
          { sep: true },
          { label: '加载 GT 标注文件夹…', act: () => pick('#pick-gt-folder', onPickGtFolder) },
          { label: '加载 GT 标注文件…', act: () => pick('#pick-gt-files', onPickGtFiles) },
          { sep: true },
          { label: '保存三联对比图…', kbd: 'S', act: () => saveCompare(true) },
          { label: '保存当前视图…', kbd: 'Shift+S', act: () => saveCompare(false) },
          { label: '导出当前帧检测结果…', act: exportFrameBoxes },
          { sep: true },
          { label: '从后端目录读取图像…', act: browseServer },
          { label: '清空图像列表', danger: true, act: clearAll },
        ],
      },
      {
        label: '模型',
        items: [
          { label: '加载模型权重…', kbd: 'Ctrl+W', act: () => pick('#pick-weight', onPickWeight) },
          { label: '指定权重路径（服务端）…', act: weightByPath },
          { label: '权重信息', act: showWeightInfo },
          {
            label: '卸载权重', danger: true, disabled: () => !state.weights, act: unloadWeight,
          },
          { sep: true },
          {
            label: '无权重时用演示算法',
            check: () => !!state.params.localFallback, style: 'check',
            act: toggleLocalFallback,
          },
          { label: '推理参数…', act: paramsDialog },
          { label: '清空检测缓存', act: clearDetCache },
        ],
      },
      {
        label: '视图',
        items: [
          { label: '原图', check: () => state.view === 'raw', style: 'radio', act: setView },
          { label: '模型检测结果', check: () => state.view === 'det', style: 'radio', act: setView },
          { label: 'GT 标注', check: () => state.view === 'gt', style: 'radio', act: setView },
          { label: '检测 × GT 叠加', check: () => state.view === 'overlay', style: 'radio', act: setView },
          { label: '三联对比', check: () => state.view === 'trio', style: 'radio', act: setView },
          { sep: true },
          { label: '检测框', check: () => state.show.det, style: 'check', act: () => toggleShow('det') },
          { label: 'GT 框', check: () => state.show.gt, style: 'check', act: () => toggleShow('gt') },
          { label: '中心判决点', check: () => state.show.center, style: 'check', act: () => toggleShow('center') },
          { label: '框标签', check: () => state.show.label, style: 'check', act: () => toggleShow('label') },
          { label: '跟踪 ID', check: () => state.show.track, style: 'check', act: () => toggleShow('track') },
          { sep: true },
          { label: '适应窗口', kbd: '0', act: () => { state.zoom.fit = true; applyZoom(); } },
          { label: '1:1 原始像素', act: () => { state.zoom.fit = false; state.zoom.scale = 1; centerZoom(); } },
          { label: '放大', kbd: '+', act: () => zoomBy(1.25) },
          { label: '缩小', kbd: '-', act: () => zoomBy(1 / 1.25) },
        ],
      },
      {
        label: '播放',
        items: [
          { label: '播放 / 暂停', kbd: 'Space', check: () => state.playing, act: togglePlay },
          { label: '上一帧', kbd: '←', act: () => step(-1) },
          { label: '下一帧', kbd: '→', act: () => step(1) },
          { sep: true },
          { label: '倍速 ×0.5', act: () => setFps(Math.max(1, state.fps / 2)) },
          { label: '倍速 ×2', act: () => setFps(Math.min(30, state.fps * 2)) },
          { label: '循环播放', check: () => state.loop, act: () => { state.loop = !state.loop; syncControls(); } },
          { sep: true },
          { label: '按当前筛选列表播放', act: () => { setPlayList(currentFiltered()); } },
          { label: '按全部列表播放', act: () => setPlayList(state.frames.slice()) },
        ],
      },
      {
        label: '帮助',
        items: [
          { label: '后端状态…', act: apiDialog },
          { label: '快捷键…', kbd: 'F1', act: shortcutsDialog },
          { label: '关于', act: aboutDialog },
        ],
      },
    ];
  }

  function buildMenus() {
    const host = $('#menus');
    host.innerHTML = '';
    menuDefs().forEach((m) => {
      const pop = el('div', { class: 'menu-pop' });
      m.items.forEach((it) => {
        if (it.sep) { pop.appendChild(el('div', { class: 'menu-sep' })); return; }
        if (it.label === undefined) return;
        const b = el('button', {
          class: (it.danger ? 'danger ' : '') + (it.style === 'radio' ? 'radio' : (it.check ? 'check' : '')),
          onclick: () => { closeMenus(); if (!it.disabled || !it.disabled()) it.act(); syncMenus(); },
        }, [
          el('span', { text: it.label }),
          it.kbd ? el('kbd', { text: it.kbd }) : null,
        ]);
        if (it.disabled && it.disabled()) b.disabled = true;
        pop.appendChild(b);
      });
      const wrap = el('div', { class: 'menu' }, [
        el('button', { class: 'menu-btn', text: m.label }),
        pop,
      ]);
      wrap.querySelector('.menu-btn').onclick = (e) => {
        e.stopPropagation();
        const on = wrap.classList.contains('open');
        closeMenus();
        if (!on) { wrap.classList.add('open'); syncMenus(); }
      };
      host.appendChild(wrap);
    });
    document.addEventListener('click', closeMenus);
  }

  function closeMenus() { $$('#menus .menu.open').forEach((m) => m.classList.remove('open')); }

  function syncMenus() {
    $$('#menus .menu').forEach((wrap, i) => {
      const def = menuDefs()[i];
      const btns = wrap.querySelectorAll('.menu-pop > button');
      let k = 0;
      def.items.forEach((it) => {
        if (it.sep) return;
        const b = btns[k++];
        if (!b) return;
        b.classList.toggle('on', !!(it.check && it.check()));
        b.disabled = !!(it.disabled && it.disabled());
      });
    });
  }

  function pick(sel, cb) {
    const inp = $(sel);
    inp.value = '';
    inp.onchange = () => { if (inp.files && inp.files.length) cb(inp.files); };
    inp.click();
  }

  /* ───────────────────────── 图像载入 ───────────────────────── */

  function makeFrame(rec) {
    return Object.assign({ id: 'f' + (state.frames.length + 1) + '_' + Math.random().toString(36).slice(2, 7) }, rec);
  }

  function pathDir(p) {
    const parts = String(p).split(/[\\/]/);
    parts.pop();
    return parts.join('/') || '.';
  }

  async function addFrames(list, opts) {
    const o = opts || {};
    const before = state.frames.length;
    const added = [];
    let bad = 0;
    for (const rec of list) {
      try {
        const img = await imgLoad(rec.url);
        const frame = makeFrame({
          path: rec.path,
          name: rec.name,
          url: rec.url,
          group: rec.group || pathDir(rec.path),
          frameNo: rec.frameNo === undefined ? frameNoOf(rec.path) : rec.frameNo,
          seq: rec.seq || seqOf(rec.path),
          img,
          w: img.naturalWidth,
          h: img.naturalHeight,
        });
        state.frames.push(frame);
        added.push(frame);
      } catch (e) {
        bad += 1;
      }
    }
    if (added.length) {
      sortFrames();
      renderSidebar();
      if (state.index < 0) {
        const first = o.select === undefined ? 0 : o.select;
        select(first >= 0 ? first : indexOfFrame(added[0]));
      } else {
        syncControls();
      }
      if (o.fitOnAdd !== false) fitZoom();
      toast(`已加载 ${added.length} 张图像${bad ? `，${bad} 张失败` : ''}`, bad ? 'warn' : 'ok');
    } else if (bad) {
      toast(`图像加载失败 ${bad} 张（浏览器安全策略或格式不支持）`, 'err');
    }
    return added;
  }

  function sortFrames() {
    state.frames.sort((a, b) => {
      const g = natCmp(a.group, b.group);
      if (g !== 0) return g;
      const n = a.frameNo - b.frameNo;
      if (n !== 0 && isFinite(n) && isFinite(a.frameNo) && isFinite(b.frameNo)) return n;
      return natCmp(a.name, b.name);
    });
    state.frames.forEach((f, i) => { f.idx = i; });
  }

  function onPickFiles(fileList) {
    const arr = Array.from(fileList).filter((f) => isImageName(f.name));
    if (!arr.length) return toast('未选择图像文件', 'warn');
    const recs = arr.map((f) => ({
      path: (f.webkitRelativePath || f.name),
      name: f.name,
      url: URL.createObjectURL(f),
      file: f,
    }));
    addFrames(recs);
  }

  function onPickFolder(fileList) {
    const arr = Array.from(fileList).filter((f) => isImageName(f.name));
    if (!arr.length) return toast('该文件夹内未找到图像', 'warn');
    arr.sort(natCmp);
    const recs = arr.map((f) => ({
      path: (f.webkitRelativePath || f.name).replace(/\\/g, '/'),
      name: f.name,
      url: URL.createObjectURL(f),
      file: f,
    }));
    addFrames(recs, { fitOnAdd: false });
  }

  /* 路径清单：一行一个路径；先按已加载列表匹配，否则走后端解析 */
  async function onPickList(fileList) {
    const f = fileList[0];
    const text = await f.text();
    const lines = text.split(/\r?\n/).map((s) => s.trim())
      .filter((s) => s && !/^#/.test(s));
    if (!lines.length) return toast('清单为空', 'warn');

    const loaded = [];
    const miss = [];
    for (const p of lines) {
      const norm = p.replace(/\\/g, '/');
      const hit = state.frames.find((fr) => fr.path === norm || fr.path.endsWith('/' + norm) || fr.name === norm);
      if (hit) loaded.push(hit);
      else miss.push(norm);
    }

    let resolved = [];
    if (miss.length) {
      if (state.api.ok) {
        const r = await apiPost('api/resolve', { paths: miss });
        resolved = (r.items || []).filter((it) => it.ok)
          .map((it) => ({ path: it.path.replace(/\\/g, '/'), name: it.name, url: it.url }));
      } else {
        resolved = matchByLooseName(miss);
      }
    }
    const got = loaded.concat(await addFrames(resolved));
    if (got.length) {
      const first = indexOfFrame(got[0]);
      select(first);
      if (state.playing) togglePlay();
    }
    if (miss.length && got.length < miss.length) {
      toast(`清单 ${lines.length} 条：命中 ${loaded.length}，解析 ${resolved.length}，未找到 ${miss.length - resolved.length}`, 'warn');
    }
  }

  function matchByLooseName(paths) {
    const out = [];
    for (const p of paths) {
      const base = p.split(/[\\/]/).pop();
      const fr = state.frames.find((f) => f.name === base);
      if (fr) out.push({ path: p, name: fr.name, url: fr.url, file: fr.file, group: pathDir(p), frameNo: frameNoOf(p) });
    }
    return out;
  }

  async function onPickGtFolder(fileList) {
    await loadGtFiles(Array.from(fileList).filter((f) => /\.(xml|txt|json)$/i.test(f.name)));
  }

  async function onPickGtFiles(fileList) {
    await loadGtFiles(Array.from(fileList));
  }

  /* GT 键统一成 `${seqNum}|${frameNo}`；seqNum 取路径里的数字段（1..87），
     取不到用 -1 → '*'，保证任意来源（XML 目录名 / cm_GT 文件名 / 图像路径）都能对上。 */
  function seqNumOf(tag) {
    if (tag === undefined || tag === null) return -1;
    const s = String(tag);
    if (s === '—') return -1;
    const m = s.match(/(\d{1,4})/);
    return m ? parseInt(m[1], 10) : -1;
  }

  function gtKey(seq, frameNo) { return `${seqNumOf(seq)}|${frameNo}`; }

  function pathDirName(dir) {
    const parts = String(dir || '').split('/').filter(Boolean);
    return parts.length ? parts[parts.length - 1] : '—';
  }

  async function loadGtFiles(files) {
    let xml = 0; let txt = 0; let json = 0; let bad = 0;
    for (const f of files) {
      const rel = (f.webkitRelativePath || f.name).replace(/\\/g, '/');
      try {
        if (/\.xml$/i.test(f.name)) { parseVocXml(await f.text(), pathDir(rel), f.name); xml += 1; }
        else if (/\.txt$/i.test(f.name)) { parseOfficialTxt(await f.text(), f.name); txt += 1; }
        else if (/\.json$/i.test(f.name)) { parseGtJson(await f.text()); json += 1; }
        else bad += 1;
      } catch (e) {
        bad += 1;
        if (bad <= 3) toast(`GT 解析失败 ${f.name}：${e.message || e}`, 'warn');
      }
    }
    if (xml + txt + json) {
      renderSidebar();
      redraw();
      syncControls();
      toast(`GT 标注：VOC ${xml} · 官方 txt ${txt} · JSON ${json}${bad ? ` · 跳过 ${bad}` : ''}`, 'ok');
    } else {
      toast('未解析到 GT 标注', 'warn');
    }
  }

/* VOC XML（ITTD 官方 Annotation/{v}/{f:03d}.xml）：
     <object><name> = 跟踪 ID</name><bndbox>xmin,ymin,xmax,ymax</bndbox></object> */
  function parseVocXml(text, dir, fileName) {
    const doc = new DOMParser().parseFromString(text, 'application/xml');
    if (doc.querySelector('parsererror')) throw new Error('XML 格式错误');
    const seq = pathDirName(dir);
    const boxes = [];
    Array.from(doc.querySelector('object') ? doc.querySelectorAll('object') : []).forEach((o) => {
      const b = o.querySelector('bndbox');
      if (!b) return;
      const q = (t) => Math.round(parseFloat(b.querySelector(t).textContent));
      const nameEl = o.querySelector('name');
      const tid = nameEl ? parseInt(nameEl.textContent, 10) : -1;
      boxes.push({
        x1: q('xmin'), y1: q('ymin'), x2: q('xmax'), y2: q('ymax'),
        track_id: isFinite(tid) ? tid : -1,
      });
    });
    const fnameEl = doc.querySelector('filename');
    const key = (fnameEl ? fnameEl.textContent.trim() : '') || fileName;
    putGt(seq, frameNoOf(key), boxes, key);
    return boxes;
  }

  /* 官方 cm_GT/{v}.txt：首行 targetnum: N，之后 frame:001 n object:1 x1 y1 x2 y2
     （每目标 3 数的中心点版本亦支持，与 official_score.det_center 同口径） */
  function parseOfficialTxt(text, fileName) {
    const rows = text.split(/\r?\n/).filter((s) => s.trim());
    if (!rows.length || !/^targetnum:\s*\d+\s*$/i.test(rows[0].trim())) {
      throw new Error('非官方 GT 格式（首行须为 targetnum: N）');
    }
    const sid = seqNumOf(fileName);
    const seq = sid >= 0 ? String(sid) : '—';
    let hit = 0;
    rows.slice(1).forEach((line) => {
      const m = line.trim().match(/^frame:(\d+)\s+(\d+)(.*)$/);
      if (!m) return;
      const fno = parseInt(m[1], 10);
      const nObj = parseInt(m[2], 10);
      if (!nObj) { putGt(seq, fno, [], null); hit += 1; return; }
      const nums = (m[3].match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
      const boxes = [];
      if (nums.length === 5 * nObj) {
        for (let k = 0; k < nums.length; k += 5) {
          boxes.push({
            x1: nums[k + 1], y1: nums[k + 2], x2: nums[k + 3], y2: nums[k + 4],
            track_id: nums[k],
          });
        }
      } else if (nums.length === 3 * nObj) {
        for (let k = 0; k < nums.length; k += 3) {
          const x = nums[k + 1]; const y = nums[k + 2];
          boxes.push({ x1: x, y1: y, x2: x, y2: y, track_id: nums[k], point: true });
        }
      } else {
        return;
      }
      putGt(seq, fno, boxes, null);
      hit += 1;
    });
    if (!hit) throw new Error('官方 GT 无有效帧');
  }

  function parseGtJson(text) {
    const j = JSON.parse(text);
    const push = (key, boxes) => putGt(seqOf(key || ''), frameNoOf(key || ''), boxes, key || null);
    if (Array.isArray(j)) j.forEach((e) => push(e.file || e.path || e.name, e.boxes || []));
    else if (j.frames) Object.keys(j.frames).forEach((k) => push(k, j.frames[k].boxes || j.frames[k]));
    else throw new Error('无法识别的 JSON 结构');
  }

  function putGt(seq, frameNo, boxes, name) {
    const sid = seqNumOf(seq);
    state.gt.set(gtKey(sid, frameNo), boxes);
    if (name) state.gtByName.set(String(name), boxes);
  }

  function gtOf(frame) {
    if (!frame) return [];
    const sid = seqNumOf(frame.seq);
    for (const k of [gtKey(sid, frame.frameNo), gtKey('—', frame.frameNo), gtKey(sid, -1)]) {
      if (state.gt.has(k)) return state.gt.get(k);
    }
    if (state.gtByName.has(frame.name)) return state.gtByName.get(frame.name);
    return [];
  }

  /* ───────────────────────── 检测（含跨帧缓存） ───────────────────────── */

  /* 检测计划：没有可用权重就**不跑任何检测**，避免把前端算法结果当成模型输出。
     server : 权重已在服务端就绪 → 真模型前向
     demo   : 用户显式开启「无权重演示」→ 前端 Top-Hat（面板会标注「演示」）
     off    : 不检测，检测面板留空并给出原因 */
  function detPlan() {
    const w = state.weights;
    if (w && w.serverId) return { mode: 'server', label: ns.Infer.ENGINE_LABEL.server };
    if (w) {
      const why = state.api.ok ? '服务端未能加载该权重' : '本地服务未连接，无法加载权重';
      if (state.params.localFallback) {
        return { mode: 'demo', label: ns.Infer.ENGINE_LABEL.demo, warn: why + '（已回退演示算法）' };
      }
      return { mode: 'off', label: '检测未启用', note: ['权重已载入，但 ' + why, '启动 python web/server.py 后自动启用'] };
    }
    if (state.params.localFallback) {
      return { mode: 'demo', label: ns.Infer.ENGINE_LABEL.demo, warn: '演示算法结果，非模型输出' };
    }
    return {
      mode: 'off', label: '未加载权重',
      note: ['未加载模型权重 · 无检测结果', '菜单「模型 → 加载模型权重…」启用真模型推理',
        '或「模型 → 无权重时用演示算法」查看离线效果'],
    };
  }

  function detKey(frame) {
    const plan = detPlan();
    const w = state.weights ? `${state.weights.id}@${state.weights.size}` : 'none';
    return `${frame.id}|${w}|${plan.mode}|${state.params.thr}|${state.params.minArea}|${state.params.maxBoxes}`;
  }

  function cachedDet(frame) {
    return state.det.cache.get(detKey(frame));
  }

  async function ensureDet(frame) {
    if (!frame) return null;
    const plan = detPlan();
    if (plan.mode === 'off') { syncChips(); return null; }
    const key = detKey(frame);
    const hit = state.det.cache.get(key);
    if (hit) return hit;
    if (state.inflight.has(key)) return null;
    state.inflight.add(key);
    state.det.busy += 1;
    syncBusy();
    try {
      const res = plan.mode === 'server'
        ? await detectServer(frame.img, {
          thr: state.params.thr, minArea: state.params.minArea, maxBoxes: state.params.maxBoxes,
          weight: state.weights.serverId, modelType: state.weights.modelType || 'auto',
        })
        : detectLocal(frame.img, {
          thr: state.params.thr, minArea: state.params.minArea, maxBoxes: state.params.maxBoxes,
        });
      const rec = { boxes: res.boxes || [], ms: res.ms || 0, engine: plan.mode, t: Date.now() };
      state.det.cache.set(key, rec);
      state.det.order.push(key);
      while (state.det.order.length > 600) state.det.cache.delete(state.det.order.shift());
      if (current() === frame) redraw();
      syncChips();
      return rec;
    } catch (e) {
      const msg = String((e && e.message) || e);
      state.det.cache.set(key, { boxes: [], ms: 0, engine: 'error', err: msg, t: Date.now() });
      state.det.order.push(key);
      if (state.det.lastErr !== msg) {
        state.det.lastErr = msg;
        toast('服务端推理失败：' + msg, 'err', 6000);
      }
      if (current() === frame) redraw();
      return null;
    } finally {
      state.inflight.delete(key);
      state.det.busy -= 1;
      syncBusy();
    }
  }

  function clearDetCache() {
    state.det.cache.clear();
    state.det.order = [];
    state.det.lastErr = '';
    redraw();
    syncChips();
    toast('检测缓存已清空', 'ok');
  }

  /* ───────────────────────── 权重 ───────────────────────── */

  async function onPickWeight(fileList) {
    const f = fileList[0];
    state.weights = {
      id: 'w_' + f.name.replace(/\W+/g, '_') + '_' + f.size + '_' + Date.now().toString(36),
      name: f.name, size: f.size, source: 'file', file: f, serverId: null,
    };
    state.det.cache.clear();
    state.det.order = [];
    // 上传一次，服务端缓存为 weight id，后续按 id 引用
    if (state.api.ok) {
      busy('上传权重到本地服务…');
      try {
        const buf = await f.arrayBuffer();
        let bin = '';
        const bytes = new Uint8Array(buf);
        const CH = 0x8000;
        for (let i = 0; i < bytes.length; i += CH) {
          bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
        }
        const r = await apiPost('api/weight', { name: f.name, data: btoa(bin) });
        if (r.ok) { state.weights.serverId = r.id; state.weights.info = r.info || {}; }
        else toast('服务端未接受权重：' + (r.error || '未知原因'), 'err');
      } catch (e) {
        toast('权重上传失败：' + (e.message || e), 'err');
      } finally {
        unbusy();
      }
    } else {
      toast('本地服务未连接，权重未参与推理（检测面板保持空白）', 'warn', 5000);
    }
    state.det.lastErr = '';
    syncChips();
    if (current()) ensureDet(current());
  }

  function weightByPath() {
    const inp = el('input', { type: 'text', placeholder: 'D:\\DBLIRST\\experiments\\mshnet_B\\ckpt\\B_0_25.pt', style: { width: '100%' } });
    const info = el('div', { class: 'hint', html: '由 <code>web/server.py</code> 在本机读取并 <code>torch.load</code>；权重不会上传到任何外部服务。' });
    const m = modal({
      title: '指定权重路径（服务端加载）',
      body: el('div', {}, [
        el('div', { class: 'field' }, [el('label', { text: '权重文件' }), inp]),
        info,
      ]),
      actions: [
        { label: '取消' },
        {
          label: '加载',
          primary: true,
          onClick: async (close) => {
            const p = inp.value.trim();
            if (!p) return toast('路径为空', 'warn');
            if (!state.api.ok) { close(); return toast('本地服务未启动，无法按路径加载', 'err'); }
            busy('加载权重…');
            try {
              const r = await apiPost('api/weight', { path: p });
              if (!r.ok) throw new Error(r.error || '未知错误');
              state.weights = {
                id: 'w_path_' + Date.now().toString(36), name: p.split(/[\\/]/).pop(),
                size: r.info ? r.info.size : 0, source: 'server-path', serverId: r.id, info: r.info || {},
              };
              state.det.cache.clear();
              state.det.order = [];
              close();
              syncChips();
              if (current()) ensureDet(current());
              toast('权重已加载：' + state.weights.name, 'ok');
            } catch (e) {
              toast('加载失败：' + (e.message || e), 'err');
            } finally { unbusy(); }
          },
        },
      ],
    });
    return m;
  }

  function unloadWeight() {
    if (!state.weights) return;
    state.weights = null;
    state.det.cache.clear();
    state.det.order = [];
    state.det.lastErr = '';
    syncChips();
    redraw();
    toast('权重已卸载，检测回到本地引擎', 'ok');
  }

  function showWeightInfo() {
    if (!state.weights) return toast('未加载权重', 'warn');
    const w = state.weights;
    const rows = [
      ['名称', w.name],
      ['大小', fmtBytes(w.size)],
      ['来源', w.source === 'file' ? '本地文件选择' : '服务端路径'],
      ['服务端 ID', w.serverId || '未上传（本地引擎）'],
      ['检测引擎', detPlan().label],
    ];
    if (w.info) Object.keys(w.info).forEach((k) => rows.push([k, String(w.info[k])]));
    modal({
      title: '权重信息',
      body: el('table', { class: 'info-table' }, rows.map((r) =>
        el('tr', {}, [el('td', { text: r[0] }), el('td', { text: r[1] })]))),
      actions: [{ label: '关闭', primary: true }],
    });
  }

  /* 演示算法开关：默认关闭。开启后未加载权重也能看 Top-Hat 效果，
   但面板/导出图会明确标注「演示」，不冒充模型输出。 */
  function toggleLocalFallback() {
    state.params.localFallback = !state.params.localFallback;
    savePref('params', state.params);
    state.det.cache.clear();
    state.det.order = [];
    state.det.lastErr = '';
    toast(state.params.localFallback
      ? '已开启演示算法：结果来自前端 Top-Hat，非模型输出'
      : '已关闭演示算法：未加载权重时不再产生检测结果',
    state.params.localFallback ? 'warn' : 'ok');
    if (current()) ensureDet(current());
    redraw();
    syncChips();
  }

  function paramsDialog() {
    const mk = (key, label, step, min, max) => {
      const i = el('input', { type: 'number', value: state.params[key], step, min, max });
      return { key, node: el('div', { class: 'field' }, [el('label', { text: label }), i]) };
    };
    const thr = mk('thr', '显著性阈值 thr', 0.05, 0, 1);
    const area = mk('minArea', '最小面积 px²', 1, 1, 64);
    const maxb = mk('maxBoxes', '最多显示框数', 5, 0, 500);
    const fb = el('input', { type: 'checkbox' });
    fb.checked = !!state.params.localFallback;
    modal({
      title: '推理参数',
      width: 'min(560px, 92vw)',
      body: el('div', {}, [
        thr.node, area.node, maxb.node,
        el('div', { class: 'field' }, [
          el('label', { text: '演示算法' }),
          el('label', { class: 'picks' }, [fb, el('span', { text: '无权重时用前端 Top-Hat 出图（非模型输出）' })]),
        ]),
        el('div', { class: 'hint', html:
          '默认<b>不做检测</b>：未加载权重时检测面板留空，避免把算法结果误读成模型输出。<br>'
          + '检测来源优先级：已加载权重（服务端前向） → 演示算法（需手动勾选）。<br>'
          + '参数变更会清空检测缓存；缓存键含帧号 + 权重 + 参数，因此切帧不会重置状态。' }),
      ]),
      actions: [
        { label: '取消' },
        {
          label: '应用',
          primary: true,
          onClick: (close) => {
            state.params.thr = clamp(parseFloat(thr.node.querySelector('input').value) || 0, 0, 1);
            state.params.minArea = clamp(parseInt(area.node.querySelector('input').value, 10) || 1, 1, 64);
            state.params.maxBoxes = clamp(parseInt(maxb.node.querySelector('input').value, 10) || 0, 0, 500);
            state.params.localFallback = fb.checked;
            savePref('params', state.params);
            state.det.cache.clear();
            state.det.order = [];
            state.det.lastErr = '';
            close();
            if (current()) ensureDet(current());
            redraw();
            syncChips();
          },
        },
      ],
    });
  }

  /* ───────────────────────── 后端 ───────────────────────── */

  async function apiGet(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }

  async function apiPost(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }

  async function probeApi() {
    if (!/^https?:/.test(location.protocol)) {
      // file:// 下 fetch 本地相对路径不可用，只能提示用户起服务
      state.api.tried = true;
      syncChips();
      return false;
    }
    try {
      const j = await apiGet('api/health');
      state.api.ok = !!(j && j.ok);
      state.api.info = j;
    } catch (e) {
      state.api.ok = false;
      state.api.info = { error: String(e.message || e) };
    }
    state.api.tried = true;
    syncChips();
    return state.api.ok;
  }

  function apiDialog() {
    const rows = [
      ['地址', location.origin + location.pathname.replace(/[^/]*$/, '') + 'api'],
      ['状态', state.api.ok ? '已连接' : (state.api.tried ? '不可用' : '未探测')],
    ];
    if (state.api.info) {
      Object.keys(state.api.info).forEach((k) => {
        const v = state.api.info[k];
        if (typeof v === 'object') return;
        rows.push([k, String(v)]);
      });
    }
    rows.push(['提示', '启动：python web/server.py  （可选，用于服务端 torch 推理）']);
    modal({
      title: '后端状态',
      body: el('div', {}, [
        el('table', { class: 'info-table' }, rows.map((r) =>
          el('tr', {}, [el('td', { text: r[0] }), el('td', { text: r[1] })]))),
      ]),
      actions: [
        { label: '重新探测', onClick: (close) => { close(); probeApi().then((ok) => toast(ok ? '后端已连接' : '后端不可用', ok ? 'ok' : 'warn')); } },
        { label: '关闭', primary: true },
      ],
    });
  }

  async function browseServer() {
    if (!state.api.ok) {
      return modal({
        title: '从后端目录读取',
        body: el('div', { class: 'hint', html:
          '需要本地服务：<br><code>python web/server.py</code><br><br>'
          + '服务未启动时，可直接用「加载图像文件夹…」，浏览器会把目录读入内存（无需服务）。' }),
        actions: [{ label: '知道了', primary: true }],
      });
    }
    const input = el('input', { type: 'text', value: 'D:/DBLIRST/data/cache/ittd/seq_0001' });
    const list = el('div', { class: 'browse-list' });
    const quick = el('div', { class: 'quick' });
    const go = async (dir) => {
      list.innerHTML = '<div class="side-empty">读取中…</div>';
      try {
        const r = await apiPost('api/list', { dir });
        quick.innerHTML = '';
        (r.parents || []).forEach((p) => quick.appendChild(
          el('button', { text: p, onclick: () => { input.value = p; go(p); } })));
        list.innerHTML = '';
        (r.dirs || []).forEach((d) => list.appendChild(
          el('button', { class: 'dir', text: d, onclick: () => { input.value = dir + '/' + d; go(dir + '/' + d); } })));
        (r.files || []).forEach((f) => list.appendChild(el('button', { text: f, disabled: true })));
        if (!(r.dirs || []).length && !(r.files || []).length) {
          list.innerHTML = '<div class="side-empty">该目录无图像</div>';
        }
      } catch (e) {
        list.innerHTML = '<div class="side-empty">读取失败：' + ns.el('span', { text: String(e.message || e) }).textContent + '</div>';
      }
    };
    quick.appendChild(el('button', { text: 'seq_0001', onclick: () => { input.value = 'D:/DBLIRST/data/cache/ittd/seq_0001'; go(input.value); } }));
    quick.appendChild(el('button', { text: 'cache/ittd', onclick: () => { input.value = 'D:/DBLIRST/data/cache/ittd'; go(input.value); } }));
    modal({
      title: '从后端目录读取图像',
      width: 'min(720px, 92vw)',
      body: el('div', {}, [
        el('div', { class: 'browse' }, [
          input,
          el('button', { class: 'btn', text: '进入', onclick: () => go(input.value.trim()) }),
        ]),
        quick,
        list,
        el('div', { class: 'hint', text: '进入目录后点「加载本目录图像」，按磁盘路径直接取图（不走浏览器沙箱）。' }),
      ]),
      actions: [
        { label: '关闭' },
        {
          label: '加载本目录图像',
          primary: true,
          onClick: async (close) => {
            const dir = input.value.trim();
            close();
            busy('按路径解析图像…');
            try {
              const r = await apiPost('api/resolve', { paths: ['*'], dir });
              const recs = (r.items || []).filter((it) => it.ok)
                .map((it) => ({ path: it.path.replace(/\\/g, '/'), name: it.name, url: it.url }));
              await addFrames(recs, { fitOnAdd: false });
            } catch (e) {
              toast('解析失败：' + (e.message || e), 'err');
            } finally { unbusy(); }
          },
        },
      ],
    });
  }

  /* ───────────────────────── 侧栏 ───────────────────────── */

  function currentFiltered() {
    const q = state.filter.trim().toLowerCase();
    if (!q) return state.frames;
    return state.frames.filter((f) => {
      if (f.path.toLowerCase().includes(q)) return true;
      if (String(f.frameNo) === q) return true;
      if (String(f.seq).toLowerCase() === q) return true;
      const g = gtOf(f);
      if (g.length && ('gt' + g.length) === q) return true;
      const d = cachedDet(f);
      return !!(d && ('det' + d.boxes.length) === q);
    });
  }

  function renderSidebar() {
    const host = $('#side-list');
    const list = currentFiltered();
    host.innerHTML = '';
    $('#side-count').textContent = `${state.index >= 0 ? state.index + 1 : 0} / ${state.frames.length}`;
    $('#side-sub').textContent = state.frames.length
      ? `${state.frames.length} 帧 · ${new Set(state.frames.map((f) => f.seq)).size} 段 · ${list.length} 显示`
      : '未加载图像';

    if (!state.frames.length) {
      host.appendChild(el('div', { class: 'side-empty', html:
        '<b>图像列表为空</b>菜单「文件」→<br>加载图像文件…<br>加载图像文件夹…<br>加载图像路径清单…' }));
      return;
    }
    if (!list.length) {
      host.appendChild(el('div', { class: 'side-empty', html:
        '<b>无匹配项</b>换个关键词，或清空过滤框' }));
      return;
    }

    let group = null;
    for (const f of list) {
      /* 分组头只在真正分目录时出现：散选单文件时 group 为 "."，不出头 */
      const showGroup = f.group && f.group !== '.';
      if (showGroup && f.group !== group) {
        group = f.group;
        host.appendChild(el('div', { class: 'side-group', text: group }));
      }
      const gts = gtOf(f);
      const det = cachedDet(f);
      const node = el('div', {
        class: 'side-item' + (f.idx === state.index ? ' on' : '') + (state.playing && playHas(f) ? ' playing' : ''),
        title: f.path,
        onclick: () => select(f.idx),
        oncontextmenu: (e) => { e.preventDefault(); copyText(f.path).then((ok) => toast(ok ? '路径已复制' : '复制失败', ok ? 'ok' : 'err')); },
      }, [
        el('div', { class: 'side-name', text: f.name }),
        gts.length ? el('div', { class: 'side-badge gt', text: 'GT' + gts.length }) : null,
        det ? el('div', { class: 'side-badge det', text: 'D' + det.boxes.length }) : null,
        el('div', { class: 'side-path', text: f.path }),
      ]);
      host.appendChild(node);
    }
  }

  function playHas(f) { return state.playList.some((x) => x.id === f.id); }

  function indexOfFrame(f) { return state.frames.indexOf(f); }

  function current() { return state.index >= 0 ? state.frames[state.index] : null; }

  /* ───────────────────────── 视图渲染 ───────────────────────── */

  const VIEW_PANELS = {
    raw: [{ key: 'raw', cap: '原始图像' }],
    det: [{ key: 'det', cap: '模型检测结果' }],
    gt: [{ key: 'gt', cap: 'GT 标注' }],
    overlay: [{ key: 'overlay', cap: '检测 × GT 叠加' }],
    trio: [{ key: 'raw', cap: '原始图像' }, { key: 'det', cap: '模型检测结果' }, { key: 'gt', cap: 'GT 标注' }],
  };

  const PAL = {
    det: '#ffeb00',   /* = render.DET_COLOR */
    gt: '#ffd700',    /* = render.GT_COLOR */
    hit: '#35d07f',
    miss: '#ff6b6b',
    fa: '#35c2ff',
    center: '#ff9d2e',
  };

  function panelsFor(view) { return VIEW_PANELS[view] || VIEW_PANELS.trio; }

  function buildStage(frame) {
    const world = $('#world');
    world.innerHTML = '';
    if (!frame) { $('#stage-empty').hidden = false; return; }
    $('#stage-empty').hidden = true;
    panelsFor(state.view).forEach((p) => {
      const img = el('img', { src: frame.url, draggable: 'false', alt: p.cap });
      const cv = el('canvas');
      const panel = el('div', { class: 'panel', dataset: { key: p.key } }, [
        img, cv, el('div', { class: 'panel-cap', text: p.cap }),
      ]);
      world.appendChild(panel);
      img.addEventListener('load', () => { fitCanvas(panel); redraw(); }, { once: true });
      if (img.complete && img.naturalWidth) { fitCanvas(panel); }
    });
    requestAnimationFrame(() => { if (state.zoom.fit) fitZoom(); else applyZoom(); redraw(); });
  }

  function fitCanvas(panel) {
    const img = panel.querySelector('img');
    const cv = panel.querySelector('canvas');
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    cv.style.width = w + 'px';
    cv.style.height = h + 'px';
  }

  function contentSize() {
    const panels = $$('#world .panel');
    if (!panels.length) return { w: 0, h: 0 };
    let w = 0; let h = 0;
    panels.forEach((p, i) => {
      const img = p.querySelector('img');
      const pw = img.naturalWidth || img.width;
      const ph = img.naturalHeight || img.height;
      w += pw + (i ? GAP : 0);
      h = Math.max(h, ph);
    });
    return { w, h };
  }

  function fitZoom() {
    state.zoom.fit = true;
    const size = contentSize();
    if (!size.w) return;
    const stage = $('#stage').getBoundingClientRect();
    const s = Math.min((stage.width - 24) / size.w, (stage.height - 24) / size.h);
    state.zoom.scale = clamp(s, 0.05, 8);
    centerZoom();
  }

  function centerZoom() {
    const size = contentSize();
    const stage = $('#stage').getBoundingClientRect();
    state.zoom.tx = Math.round((stage.width - size.w * state.zoom.scale) / 2);
    state.zoom.ty = Math.round((stage.height - size.h * state.zoom.scale) / 2);
    applyZoom();
  }

  function zoomBy(k) {
    state.zoom.fit = false;
    const stage = $('#stage').getBoundingClientRect();
    const cx = stage.width / 2;
    const cy = stage.height / 2;
    const s0 = state.zoom.scale;
    const s1 = clamp(s0 * k, 0.05, 16);
    state.zoom.tx = cx - (cx - state.zoom.tx) * (s1 / s0);
    state.zoom.ty = cy - (cy - state.zoom.ty) * (s1 / s0);
    state.zoom.scale = s1;
    applyZoom();
  }

  function applyZoom() {
    const { scale, tx, ty } = state.zoom;
    $('#world').style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    $('#zoom-label').textContent = Math.round(scale * 100) + '%';
    $('#st-zoom').textContent = '缩放 ' + Math.round(scale * 100) + '%';
    $('#ro-zoom').textContent = 'zoom ' + Math.round(scale * 100) + '%';
    $$('#world .panel').forEach(fitCanvas);
    redraw();
  }

  function boxRect(b, w, h) {
    return {
      x: Math.min(b.x1, b.x2), y: Math.min(b.y1, b.y2),
      w: Math.max(1, Math.abs(b.x2 - b.x1) + (b.point ? 0 : 1)),
      h: Math.max(1, Math.abs(b.y2 - b.y1) + (b.point ? 0 : 1)),
      w0: w, h0: h,
    };
  }

  function drawBoxes(cv, boxes, color, opts) {
    const dashed = !!(opts && opts.dashed);
    const g = cv.getContext('2d');
    const w = cv.width; const h = cv.height;
    const s = state.zoom.scale;
    const lw = Math.max(1, Math.round(1.5 / s));
    const fs = Math.max(9, Math.round(11 / s));
    g.lineWidth = lw;
    g.font = `${fs}px "Cascadia Mono", Consolas, monospace`;
    g.textBaseline = 'bottom';
    boxes.forEach((b) => {
      const r = boxRect(b, w, h);
      g.strokeStyle = color;
      g.setLineDash((dashed || b.dashed) ? [4 / s, 3 / s] : []);
      g.strokeRect(Math.round(r.x) + 0.5, Math.round(r.y) + 0.5, Math.max(1, r.w), Math.max(1, r.h));
      g.setLineDash([]);
      if (b.point) {
        g.beginPath();
        g.moveTo(r.x - 3 / s, r.y); g.lineTo(r.x + 3 / s, r.y);
        g.moveTo(r.x, r.y - 3 / s); g.lineTo(r.x, r.y + 3 / s);
        g.stroke();
      }
      if (state.show.center && !b.point) {
        const cx = (r.x + r.w / 2); const cy = (r.y + r.h / 2);
        g.strokeStyle = PAL.center;
        g.beginPath();
        g.moveTo(cx - 2.5 / s, cy); g.lineTo(cx + 2.5 / s, cy);
        g.moveTo(cx, cy - 2.5 / s); g.lineTo(cx, cy + 2.5 / s);
        g.stroke();
      }
      if (state.show.label) {
        const txt = (state.show.track && b.track_id >= 0 ? '#' + b.track_id + ' ' : '')
          + (b.score === undefined ? '' : b.score.toFixed(2));
        if (txt) {
          const tw = g.measureText(txt).width;
          g.fillStyle = 'rgba(0,0,0,.62)';
          g.fillRect(r.x, Math.max(0, r.y - fs - 2 / s), tw + 4 / s, fs + 2 / s);
          g.fillStyle = color;
          g.fillText(txt, r.x + 2 / s, Math.max(fs, r.y - 2 / s));
        }
      }
    });
    g.setLineDash([]);
  }

  /* 面板占位说明：无检测结果时写清原因，而不是留一片空白让人误判为「0 目标」 */
  function drawPanelNote(cv, lines, tone) {
    const g = cv.getContext('2d');
    const s = state.zoom.scale;
    const fs = Math.max(11, Math.round(13 / s));
    const lh = fs * 1.85;
    g.save();
    g.font = `${fs}px "Microsoft YaHei UI", "Segoe UI", sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    const y0 = cv.height / 2 - ((lines.length - 1) * lh) / 2;
    lines.forEach((t, i) => {
      g.fillStyle = tone === 'err' ? 'rgba(255,140,140,.92)' : 'rgba(163,178,196,.9)';
      g.fillText(t, cv.width / 2, y0 + i * lh);
    });
    g.restore();
  }

  function detPanelLines(det) {
    const plan = detPlan();
    if (det && det.err) return ['服务端推理失败', det.err];
    if (plan.mode === 'off') return plan.note || ['检测未启用'];
    if (det && !det.boxes.length) return ['本帧未检出目标（0 框）'];
    return null;
  }

  function redraw() {
    const frame = current();
    if (!frame) return;
    const det = cachedDet(frame);
    const gts = gtOf(frame);
    const plan = detPlan();
    $$('#world .panel').forEach((panel) => {
      const key = panel.dataset.key;
      const cv = panel.querySelector('canvas');
      const g = cv.getContext('2d');
      g.clearRect(0, 0, cv.width, cv.height);
      const cap = panel.querySelector('.panel-cap');
      if (key === 'raw') return;
      if (key === 'det') {
        if (state.show.det && det) drawBoxes(cv, det.boxes, PAL.det, {});
        const note = detPanelLines(det);
        if (note) drawPanelNote(cv, note, det && det.err ? 'err' : 'dim');
        cap.textContent = plan.mode === 'off'
          ? '模型检测结果 · 未启用'
          : `模型检测结果 · ${det ? det.boxes.length : 0} 框${plan.mode === 'demo' ? '（演示）' : ''}`;
      } else if (key === 'gt') {
        if (state.show.gt) drawBoxes(cv, gts.map((b) => Object.assign({ score: 1 }, b)), PAL.gt, { dashed: true });
        cap.textContent = `GT 标注 · ${gts.length} 目标`;
      } else if (key === 'overlay') {
        if (state.show.gt) drawBoxes(cv, gts.map((b) => Object.assign({ score: 1 }, b)), PAL.gt, { dashed: true });
        if (state.show.det && det) drawBoxes(cv, det.boxes, PAL.det, {});
        cap.textContent = plan.mode === 'off'
          ? `叠加 · GT ${gts.length} · 检测未启用`
          : `叠加 · 检 ${det ? det.boxes.length : 0} / GT ${gts.length}`;
        if (plan.mode === 'off' && !state.show.gt) {
          drawPanelNote(cv, detPanelLines(det), 'dim');
        }
      }
    });
    syncChips();
    syncStatus(frame, det, gts);
  }

  function setView(v) {
    state.view = v;
    savePref('view', v);
    readoutReset();
    $$('#view-seg button').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
    syncMenus();
    buildStage(current());
    if (current()) ensureDet(current());
  }

  function toggleShow(k) {
    state.show[k] = !state.show[k];
    savePref('show', state.show);
    syncMenus();
    redraw();
  }

  /* ───────────────────────── 帧切换 / 播放 ───────────────────────── */

  function select(idx, opts) {
    if (!state.frames.length) return;
    readoutReset();
    state.index = clamp(idx, 0, state.frames.length - 1);
    const frame = current();
    const keepScroll = $('.side-item.on');
    buildStage(frame);
    ensureDet(frame);
    renderSidebar();
    const active = $('#side-list .side-item.on');
    if (active) active.scrollIntoView({ block: 'nearest' });
    else if (keepScroll) keepScroll.scrollIntoView({ block: 'nearest' });
    syncControls();
    if (!state.zoom.fit) applyZoom();
    prefetchNeighbours();
  }

  function prefetchNeighbours() {
    for (let d = 1; d <= 2; d += 1) {
      const a = state.frames[state.index + d];
      const b = state.frames[state.index - d];
      if (a && !cachedDet(a)) ensureDet(a);
      if (b && !cachedDet(b)) ensureDet(b);
    }
  }

  function step(d) {
    if (!state.frames.length) return;
    let i = state.index + d;
    if (i < 0) i = state.loop ? state.frames.length - 1 : 0;
    if (i >= state.frames.length) i = state.loop ? 0 : state.frames.length - 1;
    if (i === state.index) return;
    select(i);
  }

  function setPlayList(list) {
    state.playList = list;
    toast(`播放列表 ${list.length} 帧`, 'ok');
    if (list.length && state.playing) { togglePlay(); togglePlay(); }
  }

  function togglePlay() {
    if (state.playing) {
      clearInterval(state.timer);
      state.timer = 0;
      state.playing = false;
      toast('暂停', null, 900);
    } else {
      if (!state.frames.length) return toast('未加载图像', 'warn');
      const list = currentFiltered();
      state.playList = list;
      if (!state.playList.length) return toast('播放列表为空', 'warn');
      if (state.index < 0) select(indexOfFrame(state.playList[0]));
      state.timer = setInterval(playTick, Math.max(16, Math.round(1000 / state.fps)));
      state.playing = true;
    }
    syncControls();
    renderSidebar();
  }

  function playTick() {
    const list = state.playList;
    if (!list.length) return togglePlay();
    let pos = list.findIndex((f) => f.id === (current() || {}).id);
    if (pos < 0) pos = -1;
    pos += 1;
    if (pos >= list.length) {
      if (!state.loop) return togglePlay();
      pos = 0;
    }
    select(indexOfFrame(list[pos]));
  }

  function setFps(v) {
    state.fps = clamp(Math.round(v), 1, 30);
    savePref('fps', state.fps);
    if (state.playing) { clearInterval(state.timer); state.timer = setInterval(playTick, Math.max(16, Math.round(1000 / state.fps))); }
    syncControls();
  }

  /* ───────────────────────── 保存对比图 ───────────────────────── */

  async function saveCompare(forceTrio) {
    const frame = current();
    if (!frame) return toast('未加载图像', 'warn');
    const view = forceTrio ? 'trio' : state.view;
    const keys = panelsFor(view).map((p) => p.key);
    const det = cachedDet(frame) || await ensureDet(frame);
    const gts = gtOf(frame);

    const gap = GAP;
    const capH = CAP_H;
    const footH = 26;
    const W = frame.w * keys.length + gap * (keys.length - 1);
    const H = frame.h + capH + footH;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const g = cv.getContext('2d');
    g.fillStyle = '#0b0f14';
    g.fillRect(0, 0, W, H);
    g.fillStyle = '#ffb020';
    g.font = '16px "Microsoft YaHei UI", sans-serif';
    g.textBaseline = 'middle';
    g.fillText(`DSLD 检测检视 · ${frame.path}`, 10, capH / 2);
    g.fillStyle = '#8fa1b5';
    g.font = '13px "Cascadia Mono", Consolas, monospace';
    g.textAlign = 'right';
    g.fillText(`${fmtTime(new Date())}  ${frame.w}×${frame.h}  thr=${state.params.thr}  引擎=${detPlan().label}`, W - 10, capH / 2);
    g.textAlign = 'left';

    keys.forEach((key, i) => {
      const x = i * (frame.w + gap);
      const y = capH;
      g.drawImage(frame.img, x, y);
      const sub = document.createElement('canvas');
      sub.width = frame.w; sub.height = frame.h;
      const sg = sub.getContext('2d');
      if (key === 'det' && state.show.det) {
        drawBoxesOn(sg, det ? det.boxes : [], PAL.det);
      } else if (key === 'gt' && state.show.gt) {
        drawBoxesOn(sg, gts.map((b) => Object.assign({ score: 1 }, b)), PAL.gt, true);
      } else if (key === 'overlay') {
        if (state.show.gt) drawBoxesOn(sg, gts.map((b) => Object.assign({ score: 1 }, b)), PAL.gt, true);
        if (state.show.det) drawBoxesOn(sg, det ? det.boxes : [], PAL.det);
      }
      g.drawImage(sub, x, y);
      const cap = { raw: '原始图像', det: `模型检测结果 ${det ? det.boxes.length : 0} 框`, gt: `GT 标注 ${gts.length} 目标`, overlay: '检测 × GT 叠加' }[key];
      g.fillStyle = '#dbe4ee';
      g.font = '13px "Microsoft YaHei UI", sans-serif';
      g.fillText(cap, x + 4, H - footH / 2);
      g.strokeStyle = '#334154';
      g.strokeRect(x + 0.5, capH + 0.5, frame.w - 1, frame.h - 1);
    });

    const name = `DSLD_${view}_${(frame.seq || 'x').replace(/\W+/g, '')}_${String(frame.frameNo).padStart(4, '0')}_${stamp()}.png`;
    const blob = await ns.canvasToBlob(cv);
    download(blob, name);
    toast(`已保存 ${name}`, 'ok');
  }

  /* 导出用：1:1 像素、线宽 2px、标签 12px */
  function drawBoxesOn(g, boxes, color, dashed) {
    g.lineWidth = 2;
    g.font = '12px "Cascadia Mono", Consolas, monospace';
    g.textBaseline = 'bottom';
    boxes.forEach((b) => {
      const r = boxRect(b, g.canvas.width, g.canvas.height);
      g.strokeStyle = color;
      g.setLineDash(dashed ? [5, 4] : []);
      g.strokeRect(r.x + 1, r.y + 1, Math.max(1, r.w - 1), Math.max(1, r.h - 1));
      g.setLineDash([]);
      if (b.point) {
        g.beginPath();
        g.moveTo(r.x - 5, r.y); g.lineTo(r.x + 5, r.y);
        g.moveTo(r.x, r.y - 5); g.lineTo(r.x, r.y + 5);
        g.stroke();
      }
      if (state.show.label && !b.point) {
        const txt = (state.show.track && b.track_id >= 0 ? '#' + b.track_id + ' ' : '') + (b.score || 0).toFixed(2);
        const tw = g.measureText(txt).width;
        g.fillStyle = 'rgba(0,0,0,.65)';
        g.fillRect(r.x, Math.max(0, r.y - 15), tw + 5, 14);
        g.fillStyle = color;
        g.fillText(txt, r.x + 2, Math.max(12, r.y - 2));
      }
    });
  }

  function exportFrameBoxes() {
    const frame = current();
    if (!frame) return toast('未加载图像', 'warn');
    const det = cachedDet(frame);
    const rows = det ? det.boxes.map((b, i) => [
      frame.frameNo, b.track_id >= 0 ? b.track_id : i + 1, b.score.toFixed(4), b.x1, b.y1, b.x2, b.y2,
    ].join(' ')) : [];
    const head = '# frame track_id score x1 y1 x2 y2  (DSLD 检视台导出)';
    const txt = [head].concat(rows).join('\n') + '\n';
    download(new Blob([txt], { type: 'text/plain' }), `det_${frame.seq}_${frame.frameNo}_${stamp()}.txt`);
    toast('检测结果已导出', 'ok');
  }

  function clearAll() {
    confirmBox('清空', '清空图像列表（权重与 GT 保留）？', () => {
      state.frames.forEach((f) => { if (f.url.startsWith('blob:')) URL.revokeObjectURL(f.url); });
      state.frames = [];
      state.index = -1;
      state.playList = [];
      if (state.playing) togglePlay();
      state.det.cache.clear();
      state.det.order = [];
      buildStage(null);
      renderSidebar();
      syncControls();
      syncChips();
      toast('已清空', 'ok');
    });
  }

  /* ───────────────────────── 像素读数 ───────────────────────── */

  function grayOf(frame) {
    let g = state.grayCache.get(frame.id);
    if (g) return g;
    g = Infer.toGray(frame.img);
    state.grayCache.set(frame.id, g);
    while (state.grayCache.size > GRAY_CACHE) {
      state.grayCache.delete(state.grayCache.keys().next().value);
    }
    return g;
  }

  function pixelAt(frame, x, y) {
    try {
      const g = grayOf(frame);
      if (x < 0 || y < 0 || x >= g.w || y >= g.h) return '—';
      return Math.round(g.data[y * g.w + x]);
    } catch (e) {
      return '—';
    }
  }

  function initPixelReadout() {
    const stage = $('#stage');
    stage.addEventListener('mousemove', (e) => {
      const panel = e.target.closest ? e.target.closest('.panel') : null;
      const frame = current();
      if (!panel || !frame) return readoutReset();
      /* 用 canvas 覆盖层取参考矩形：img 带 1px 边框，用 img 的 border-box 会带来
         约 1~2px 的坐标偏移（4×4 小目标上足以读错位置）。 */
      const ref = panel.querySelector('canvas') || panel.querySelector('img');
      const r = ref.getBoundingClientRect();
      if (!r.width || !r.height) return readoutReset();
      const x = Math.floor(((e.clientX - r.left) / r.width) * frame.w);
      const y = Math.floor(((e.clientY - r.top) / r.height) * frame.h);
      if (x < 0 || y < 0 || x >= frame.w || y >= frame.h) return readoutReset();
      readout(panelCapOf(panel.dataset.key), x, y, frame, panel.dataset.key);
    });
    stage.addEventListener('mouseleave', readoutReset);
  }

  function panelCapOf(key) {
    return { raw: '原图', det: '检测', gt: 'GT', overlay: '叠加' }[key] || key;
  }

  function readout(cap, x, y, frame, key) {
    $('#ro-panel').textContent = cap;
    $('#ro-xy').textContent = `${x}, ${y}`;
    $('#ro-gray').textContent = pixelAt(frame, x, y);
    let hit = null;
    if (key !== 'raw') {
      const pools = [];
      if ((key === 'det' || key === 'overlay') && state.show.det && cachedDet(frame)) {
        cachedDet(frame).boxes.forEach((b) => pools.push(Object.assign({ _k: 'det' }, b)));
      }
      if ((key === 'gt' || key === 'overlay') && state.show.gt) {
        gtOf(frame).forEach((b) => pools.push(Object.assign({ _k: 'gt', score: 1 }, b)));
      }
      hit = pools.find((b) => {
        const r = boxRect(b, frame.w, frame.h);
        return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
      });
    }
    const bo = $('#ro-box');
    if (hit) {
      const r = boxRect(hit, frame.w, frame.h);
      bo.textContent = `${hit._k === 'gt' ? 'GT' : 'det'} ${r.w}×${r.h} @${r.x},${r.y}`
        + (hit._k === 'det' ? ` s=${(hit.score || 0).toFixed(2)}` : '')
        + (hit.track_id >= 0 ? ` #${hit.track_id}` : '');
      bo.style.color = hit._k === 'gt' ? PAL.gt : PAL.det;
    } else {
      bo.textContent = '—';
      bo.style.color = '';
    }
  }

  function readoutReset() {
    $('#ro-panel').textContent = '—';
    $('#ro-xy').textContent = '—';
    $('#ro-gray').textContent = '—';
    $('#ro-box').textContent = '—';
    $('#ro-box').style.color = '';
  }

  /* ───────────────────────── 控件同步 ───────────────────────── */

  function syncControls() {
    const n = state.frames.length;
    $('#frame-total').textContent = n;
    $('#frame-cur').textContent = n ? state.index + 1 : 0;
    const sl = $('#frame-slider');
    sl.max = String(Math.max(0, n - 1));
    sl.value = String(Math.max(0, state.index));
    const f = current();
    $('#frame-seq').textContent = f ? `段 ${f.seq} · ${f.name}` : '—';
    $('#jump-input').max = String(Math.max(1, n));
    $('#jump-input').value = String(state.index + 1 || 1);
    $('#btn-play').classList.toggle('on', state.playing);
    $('#btn-play').textContent = state.playing ? '❚❚ 暂停' : '▶ 播放';
    $('#btn-loop').classList.toggle('on', state.loop);
    $('#speed-range').value = String(state.fps);
    $('#speed-label').textContent = state.fps + ' fps';
    $('#side-count').textContent = `${state.index >= 0 ? state.index + 1 : 0} / ${n}`;
    $$('#view-seg button').forEach((b) => b.classList.toggle('on', b.dataset.view === state.view));
    syncMenus();
  }

  function syncChips() {
    const f = current();
    const det = f ? cachedDet(f) : null;
    const gts = f ? gtOf(f) : [];
    const wchip = $('#chip-weight');
    if (state.weights) {
      wchip.textContent = `权重 ${state.weights.name}`;
      wchip.className = 'chip ' + (state.weights.serverId ? 'chip-ok' : 'chip-warn');
      wchip.title = state.weights.serverId ? '服务端已加载，参与推理'
        : '本地服务不可用，权重未参与推理';
    } else {
      wchip.textContent = '权重 未加载';
      wchip.className = 'chip chip-idle';
      wchip.title = '未加载权重 → 不执行检测，检测面板留空';
    }
    const plan = detPlan();
    const echip = $('#chip-engine');
    echip.textContent = '引擎 ' + plan.label;
    echip.className = 'chip-sm ' + (plan.mode === 'server' ? '' : 'warn');
    echip.title = plan.warn || (plan.note ? plan.note[0] : '');

    const bchip = $('#chip-backend');
    if (state.api.ok) {
      const d = state.api.info || {};
      bchip.textContent = `后端 已连接${d.torch ? ' · torch' + (d.device ? ' ' + d.device : '') : ' · 无 torch'}`;
      bchip.className = 'chip ' + (d.torch ? 'chip-ok' : 'chip-warn');
    } else {
      bchip.textContent = '后端 未连接';
      bchip.className = 'chip chip-idle';
      bchip.title = 'python web/server.py 可启用服务端权重推理与按路径读图';
    }

    const nBox = det && !det.err ? det.boxes.length : null;
    $('#chip-detect').textContent = state.det.busy ? '检测 进行中'
      : (plan.mode === 'off' ? '检测 未启用' : (nBox === null ? '检测 无结果' : `检测 ${nBox} 框`));
    $('#chip-detect').className = 'chip ' + (state.det.busy ? 'chip-busy'
      : (plan.mode === 'server' && nBox !== null ? 'chip-ok' : 'chip-idle'));

    $('#chip-det').textContent = plan.mode === 'off' ? '检测 未启用'
      : (nBox === null ? '检测 —' : `检测 ${nBox} 框 ${fmtMs(det.ms)}`);
    $('#chip-det').className = 'chip-sm ' + (plan.mode === 'off' ? 'warn' : 'det');
    $('#chip-gt').textContent = `GT ${gts.length} 目标`;
    const sc = nBox === null ? null : frameScore(det.boxes, gts);
    $('#chip-score').textContent = sc ? `命中 ${sc.hit} 虚警 ${sc.fa} 漏检 ${sc.miss}` : '评分 —';
    $('#chip-score').className = 'chip-sm ' + (sc && sc.fa === 0 && sc.miss === 0 ? '' : 'warn');
  }

  /* 官方口径（official_score.py ①）：GT 框内有检测点 = 正确，框外检测点 = 虚警 */
  function frameScore(dets, gts) {
    if (!dets.length && !gts.length) return null;
    let hit = 0; let miss = 0; let fa = 0;
    gts.forEach((g) => {
      const r = boxRect(g, 1e9, 1e9);
      const cx = (Math.min(g.x1, g.x2) + Math.max(g.x1, g.x2)) / 2;
      const cy = (Math.min(g.y1, g.y2) + Math.max(g.y1, g.y2)) / 2;
      const inside = dets.some((d) => cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h);
      if (inside) hit += 1; else miss += 1;
    });
    dets.forEach((d) => {
      const cx = (Math.min(d.x1, d.x2) + Math.max(d.x1, d.x2)) / 2;
      const cy = (Math.min(d.y1, d.y2) + Math.max(d.y1, d.y2)) / 2;
      const inGt = gts.some((g) => {
        const r = boxRect(g, 1e9, 1e9);
        return cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h;
      });
      if (!inGt) fa += 1;
    });
    return { hit, miss, fa };
  }

  function syncStatus(frame, det, gts) {
    const plan = detPlan();
    $('#st-path').textContent = frame.path;
    $('#st-frame').textContent = `${frame.seq} · 帧 ${frame.frameNo} · ${frame.w}×${frame.h}`;
    $('#st-det').textContent = plan.mode === 'off' ? `检测 未启用 · ${plan.label}`
      : state.det.busy ? '检测 计算中…'
        : det ? `检测 ${det.boxes.length} 框 / ${fmtMs(det.ms)} / ${plan.label}` : '检测 无结果';
    const sc = det && !det.err ? frameScore(det.boxes, gts) : null;
    $('#st-score').textContent = sc ? `命中 ${sc.hit} · 虚警 ${sc.fa} · 漏检 ${sc.miss}` : '评分 —';
    $('#st-api').textContent = state.api.ok
      ? `后端 ${location.host}${state.api.info && state.api.info.torch ? ' · torch' : ''}`
      : '后端 未连接';
  }

  function busy(text) {
    $('#busy-text').textContent = text;
    $('#busy').hidden = false;
  }

  function unbusy() {
    if (!state.det.busy) $('#busy').hidden = true;
    syncChips();
  }

  function syncBusy() {
    $('#busy-text').textContent = '推理中…';
    if (state.det.busy) $('#busy').hidden = false;
    else $('#busy').hidden = true;
    syncChips();
  }

  /* ───────────────────────── 帮助 ───────────────────────── */

  const KEYS = [
    ['← / →', '上一帧 / 下一帧'],
    ['空格', '播放 / 暂停'],
    ['Home / End', '首帧 / 末帧'],
    ['0', '适应窗口'],
    ['+ / -', '放大 / 缩小'],
    ['1…5', '原图 / 检测 / GT / 叠加 / 三联'],
    ['Ctrl+O', '加载图像文件'],
    ['Ctrl+Shift+O', '加载图像文件夹'],
    ['Ctrl+L', '加载图像路径清单'],
    ['Ctrl+W', '加载模型权重'],
    ['S / Shift+S', '保存三联对比图 / 当前视图'],
    ['鼠标滚轮', '以光标为中心缩放'],
    ['鼠标拖拽', '平移画布'],
  ];

  function shortcutsDialog() {
    modal({
      title: '快捷键',
      width: 'min(560px, 92vw)',
      body: el('table', { class: 'kbd-table' }, KEYS.map((k) =>
        el('tr', {}, [el('td', { text: k[0] }), el('td', { text: k[1] })]))),
      actions: [{ label: '关闭', primary: true }],
    });
  }

  function aboutDialog() {
    const rows = [
      ['名称', 'DSLD 检测效果检视台'],
      ['对应方案', '双状态液态动力学（MSHNet / T-MSD3D 基线）'],
      ['数据', 'ITTD 87 段 × 250 帧，原生 640×480'],
      ['检测口径', 'IoU≥0.5 框级；中心命中（official_score.py ①）'],
      ['检测引擎', detPlan().label],
      ['图像列表', `${state.frames.length} 帧 / GT ${state.gt.size} 组 / 缓存 ${state.det.cache.size} 帧`],
    ];
    modal({
      title: '关于',
      body: el('div', {}, [
        el('table', { class: 'info-table' }, rows.map((r) =>
          el('tr', {}, [el('td', { text: r[0] }), el('td', { text: r[1] })]))),
        el('div', { class: 'hint', html:
          '左侧栏宽度夹在 132px–20%；权重加载后切帧不清空检测状态，'
          + '已算帧结果按 (帧,权重,参数) 缓存。' }),
      ]),
      actions: [{ label: '关闭', primary: true }],
    });
  }

  /* ───────────────────────── 交互绑定 ───────────────────────── */

  function bindStage() {
    const stage = $('#stage');
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      state.zoom.fit = false;
      const rect = stage.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const s0 = state.zoom.scale;
      const s1 = clamp(s0 * (e.deltaY < 0 ? 1.15 : 1 / 1.15), 0.05, 16);
      state.zoom.tx = mx - (mx - state.zoom.tx) * (s1 / s0);
      state.zoom.ty = my - (my - state.zoom.ty) * (s1 / s0);
      state.zoom.scale = s1;
      applyZoom();
    }, { passive: false });

    let drag = null;
    stage.addEventListener('mousedown', (e) => {
      if (e.button !== 0 && e.button !== 1) return;
      drag = { x: e.clientX, y: e.clientY, tx: state.zoom.tx, ty: state.zoom.ty };
      stage.style.cursor = 'grabbing';
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!drag) return;
      state.zoom.fit = false;
      state.zoom.tx = drag.tx + (e.clientX - drag.x);
      state.zoom.ty = drag.ty + (e.clientY - drag.y);
      applyZoom();
    });
    window.addEventListener('mouseup', () => { drag = null; stage.style.cursor = ''; });
    stage.addEventListener('dblclick', () => { state.zoom.fit = true; fitZoom(); });
  }

  function bindSidebar() {
    $('#side-filter').addEventListener('input', debounce((e) => {
      state.filter = e.target.value;
      renderSidebar();
    }, 140));
    $('#btn-scroll').onclick = () => {
      const a = $('#side-list .side-item.on');
      if (a) a.scrollIntoView({ block: 'center' });
    };
    $('#btn-copy').onclick = () => {
      if (!state.frames.length) return toast('列表为空', 'warn');
      copyText(state.frames.map((f) => f.path).join('\n'))
        .then((ok) => toast(ok ? `已复制 ${state.frames.length} 条路径` : '复制失败', ok ? 'ok' : 'err'));
    };

    /* 宽度拖拽：硬性夹在 [132px, 20vw]（app.css 顶部约束） */
    const handle = $('#side-resize');
    let resizing = false;
    handle.addEventListener('mousedown', (e) => {
      resizing = true;
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!resizing) return;
      const w = clamp(e.clientX, 132, Math.round(window.innerWidth * 0.2));
      document.documentElement.style.setProperty('--side-w', w + 'px');
      if (state.zoom.fit) fitZoom();
    });
    window.addEventListener('mouseup', () => {
      if (!resizing) return;
      resizing = false;
      const w = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--side-w'), 10);
      savePref('sideW', w);
    });
    const sw = savePref('sideW');
    if (sw) document.documentElement.style.setProperty('--side-w', clamp(sw, 132, Math.round(window.innerWidth * 0.2)) + 'px');
  }

  function bindTransport() {
    $('#btn-first').onclick = () => select(0);
    $('#btn-last').onclick = () => select(state.frames.length - 1);
    $('#btn-prev').onclick = () => step(-1);
    $('#btn-next').onclick = () => step(1);
    $('#btn-play').onclick = togglePlay;
    $('#btn-loop').onclick = () => { state.loop = !state.loop; syncControls(); };
    $('#frame-slider').addEventListener('input', (e) => {
      if (state.playing) togglePlay();
      select(parseInt(e.target.value, 10));
    });
    $('#jump-input').addEventListener('change', (e) => {
      const v = parseInt(e.target.value, 10);
      if (isFinite(v)) select(v - 1);
    });
    $('#speed-range').addEventListener('input', (e) => setFps(parseInt(e.target.value, 10)));
  }

  function bindViewbar() {
    $$('#view-seg button').forEach((b) => { b.onclick = () => setView(b.dataset.view); });
    $('#zoom-in').onclick = () => zoomBy(1.25);
    $('#zoom-out').onclick = () => zoomBy(1 / 1.25);
    $('#zoom-fit').onclick = () => { state.zoom.fit = true; fitZoom(); };
    $('#zoom-actual').onclick = () => { state.zoom.fit = false; state.zoom.scale = 1; centerZoom(); };
  }

  function bindKeys() {
    window.addEventListener('keydown', (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if (e.ctrlKey || e.metaKey) {
        const k = e.key.toLowerCase();
        if (k === 'o' && e.shiftKey) { e.preventDefault(); pick('#pick-folder', onPickFolder); }
        else if (k === 'o') { e.preventDefault(); pick('#pick-images', onPickFiles); }
        else if (k === 'l') { e.preventDefault(); pick('#pick-list', onPickList); }
        else if (k === 'w') { e.preventDefault(); pick('#pick-weight', onPickWeight); }
        return;
      }
      switch (e.key) {
        case 'ArrowLeft': e.preventDefault(); step(-1); break;
        case 'ArrowRight': e.preventDefault(); step(1); break;
        case 'Home': e.preventDefault(); select(0); break;
        case 'End': e.preventDefault(); select(state.frames.length - 1); break;
        case ' ': e.preventDefault(); togglePlay(); break;
        case '+': case '=': zoomBy(1.25); break;
        case '-': case '_': zoomBy(1 / 1.25); break;
        case '0': state.zoom.fit = true; fitZoom(); break;
        case '1': setView('raw'); break;
        case '2': setView('det'); break;
        case '3': setView('gt'); break;
        case '4': setView('overlay'); break;
        case '5': setView('trio'); break;
        case 's': case 'S': saveCompare(e.key === 's'); break;
        case 'F1': e.preventDefault(); shortcutsDialog(); break;
        default: break;
      }
    });
    window.addEventListener('resize', debounce(() => {
      const w = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--side-w'), 10) || 0;
      if (w) document.documentElement.style.setProperty('--side-w', clamp(w, 132, Math.round(window.innerWidth * 0.2)) + 'px');
      if (state.zoom.fit) fitZoom(); else applyZoom();
      redraw();
    }, 160));
  }

  /* 拖放文件到窗口任意位置 */
  function bindDrop() {
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((t) => window.addEventListener(t, stop, false));
    window.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (!dt) return;
      const files = Array.from(dt.files || []);
      const imgs = files.filter((f) => isImageName(f.name));
      const wts = files.filter((f) => /\.(pt|pth|ckpt)$/i.test(f.name));
      if (imgs.length) {
        imgs.sort(natCmp);
        addFrames(imgs.map((f) => ({
          path: f.webkitRelativePath || f.name, name: f.name, url: URL.createObjectURL(f), file: f,
        })));
      }
      if (wts.length) onPickWeight([wts[0]]);
    });
  }

  /* ───────────────────────── 启动 ───────────────────────── */

  function boot() {
    buildMenus();
    bindStage();
    bindSidebar();
    bindTransport();
    bindViewbar();
    bindKeys();
    bindDrop();
    initPixelReadout();
    setView(state.view);
    renderSidebar();
    syncControls();
    syncChips();
    readoutReset();
    probeApi();
    toast('检视台就绪：菜单「文件」加载图像，「模型」加载权重', 'ok', 4200);
  }

  ns.app = { state, saveCompare, ensureDet, select, toast };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})(window.DSLD);

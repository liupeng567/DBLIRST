/* DSLD 检视台 —— 通用工具层：DOM / 弹窗 / 提示 / 下载 / 排序。 */
window.DSLD = window.DSLD || {};
(function (ns) {
  'use strict';

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.prototype.slice.call((root || document).querySelectorAll(sel));

  function el(tag, attrs, kids) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (k === 'html') node.innerHTML = v;
        else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
        else if (k === 'dataset') Object.assign(node.dataset, v);
        else node.setAttribute(k, v === true ? '' : v);
      }
    }
    (kids || []).forEach((c) => {
      if (c === null || c === undefined || c === false) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  function fmtBytes(n) {
    if (!n && n !== 0) return '—';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i += 1; }
    return (i ? n.toFixed(1) : n | 0) + ' ' + u[i];
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined) return '—';
    return ms >= 100 ? ms.toFixed(0) + ' ms' : ms.toFixed(1) + ' ms';
  }

  function fmtTime(d) {
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  function stamp() {
    const p = (n) => String(n).padStart(2, '0');
    const d = new Date();
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      const ta = el('textarea', { style: { position: 'fixed', opacity: '0' } });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
      ta.remove();
      return ok;
    }
  }

  function toast(msg, kind, ms) {
    const wrap = $('#toasts');
    if (!wrap) return;
    const node = el('div', { class: 'toast' + (kind ? ' ' + kind : ''), text: msg });
    wrap.appendChild(node);
    const life = ms || (kind === 'err' ? 6000 : 3200);
    setTimeout(() => {
      node.style.transition = 'opacity .2s';
      node.style.opacity = '0';
      setTimeout(() => node.remove(), 220);
    }, life);
    while (wrap.children.length > 6) wrap.firstChild.remove();
  }

  /* 模态框：opts = {title, body(Node|string), actions:[{label, primary, danger, onClick(close)}], width} */
  function modal(opts) {
    const mask = $('#modal-mask');
    mask.innerHTML = '';
    mask.hidden = false;
    const body = el('div', { class: 'modal-body' });
    if (typeof opts.body === 'string') body.innerHTML = opts.body;
    else if (opts.body) body.appendChild(opts.body);

    const close = () => { mask.hidden = true; mask.innerHTML = ''; document.removeEventListener('keydown', onKey, true); };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
    document.addEventListener('keydown', onKey, true);

    const foot = el('div', { class: 'modal-foot' });
    (opts.actions || [{ label: '关闭', primary: true }]).forEach((a) => {
      foot.appendChild(el('button', {
        class: 'btn' + (a.primary ? ' primary' : '') + (a.danger ? ' danger' : ''),
        text: a.label,
        onclick: () => (a.onClick ? a.onClick(close) : close()),
      }));
    });

    const box = el('div', { class: 'modal', style: opts.width ? { width: opts.width } : null }, [
      el('div', { class: 'modal-head' }, [
        el('span', { text: opts.title || '' }),
        el('button', { class: 'mini', text: '✕', onclick: close }),
      ]),
      body,
      opts.actions && opts.actions.length ? foot : null,
    ]);
    mask.appendChild(box);
    mask.onclick = (e) => { if (e.target === mask) close(); };
    return { close, body, box };
  }

  function confirmBox(title, msg, onOk) {
    modal({
      title,
      body: el('div', { class: 'hint', html: msg }),
      actions: [
        { label: '取消' },
        { label: '确定', primary: true, onClick: (close) => { close(); onOk(); } },
      ],
    });
  }

  /* 自然排序：seq2 < seq10，帧 002 < 010 */
  const collator = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' });
  const natCmp = (a, b) => collator.compare(String(a), String(b));

  function debounce(fn, ms) {
    let t = 0;
    return function () {
      const args = arguments;
      clearTimeout(t);
      t = setTimeout(() => fn.apply(null, args), ms);
    };
  }

  function savePref(k, v) {
    try {
      if (v === undefined) return JSON.parse(localStorage.getItem('dsld.' + k) || 'null');
      localStorage.setItem('dsld.' + k, JSON.stringify(v));
    } catch (e) { /* file:// 下 localStorage 可能不可用 */ }
    return null;
  }

  function imgLoad(src) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('图像解码失败: ' + src));
      im.src = src;
    });
  }

  async function canvasToBlob(canvas) {
    if (canvas.toBlob) {
      return new Promise((r) => canvas.toBlob(r, 'image/png'));
    }
    const url = canvas.toDataURL('image/png');
    const res = await fetch(url);
    return res.blob();
  }

  ns.$ = $;
  ns.$$ = $$;
  ns.el = el;
  ns.clamp = clamp;
  ns.toast = toast;
  ns.modal = modal;
  ns.confirmBox = confirmBox;
  ns.download = download;
  ns.copyText = copyText;
  ns.fmtBytes = fmtBytes;
  ns.fmtMs = fmtMs;
  ns.fmtTime = fmtTime;
  ns.stamp = stamp;
  ns.natCmp = natCmp;
  ns.debounce = debounce;
  ns.savePref = savePref;
  ns.imgLoad = imgLoad;
  ns.canvasToBlob = canvasToBlob;
})(window.DSLD);
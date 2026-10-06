/* 静态自检：① app.js 里引用的 #id 是否都在 index.html 中存在
 *          ② index.html 里的 <script>/<link> 目标文件是否存在
 *          ③ 无浏览器环境下加载 util.js / infer.js 能否正常初始化（不抛错）
 * 用法：node web/selfcheck.js   （纯临时校验脚本，不属于交付物） */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const web = __dirname;
const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
const appJs = fs.readFileSync(path.join(web, 'js', 'app.js'), 'utf8');

let bad = 0;

/* ① id 契约 */
const htmlIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
const usedIds = new Set([...appJs.matchAll(/\$\('#([a-zA-Z0-9_-]+)'\)/g)].map((m) => m[1]));
const missing = [...usedIds].filter((id) => !htmlIds.has(id));
console.log(`[1] index.html 定义 id ${htmlIds.size} 个；app.js 引用 ${usedIds.size} 个`);
if (missing.length) { bad++; console.log('    缺失 id:', missing.join(', ')); }
else console.log('    OK  全部命中');

/* ② 资源引用 */
const assets = [...html.matchAll(/(?:src|href)="([^"]+)"/g)]
  .map((m) => m[1])
  .filter((u) => !/^(https?:|data:|#)/.test(u));
for (const a of assets) {
  const p = path.join(web, a);
  const ok = fs.existsSync(p);
  if (!ok) bad++;
  console.log(`[2] ${ok ? 'OK ' : 'NG '} ${a}`);
}

/* ③ 无 DOM 环境下加载两个纯逻辑文件 */
const sandbox = {
  console,
  performance,
  Intl,
  document: {
    createElement: () => ({ getContext: () => null, width: 0, height: 0, toDataURL: () => '' }),
    addEventListener() {}, querySelector: () => null, querySelectorAll: () => [],
  },
  localStorage: { getItem: () => null, setItem() {} },
  navigator: {},
  fetch: () => Promise.reject(new Error('no net')),
  setTimeout, clearTimeout, ImageData: function () {},
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
for (const f of ['js/util.js', 'js/infer.js']) {
  try {
    vm.runInContext(fs.readFileSync(path.join(web, f), 'utf8'), sandbox, { filename: f });
    console.log(`[3] OK  ${f} 载入无异常`);
  } catch (e) {
    bad++;
    console.log(`[3] NG  ${f}: ${e.message}`);
  }
}

/* 纯函数抽查 */
const { Infer } = sandbox.DSLD;
const cases = [
  ['frameNoOf  ITTD/frames/001.bmp', Infer.frameNoOf('D:/x/Annotation/1/001.bmp'), 1],
  ['frameNoOf  seq_0007/042.bmp', Infer.frameNoOf('D:/x/seq_0007/042.bmp'), 42],
  ['seqOf      cache/ittd/seq_0031', Infer.seqOf('D:/x/cache/ittd/seq_0031/frames/010.bmp'), 'seq_0031'],
  ['seqOf      Annotation/12/...', Infer.seqOf('D:/x/Annotation/12/007.bmp'), '12'],
  ['isImageName a.bmp', Infer.isImageName('a.bmp'), true],
  ['isImageName a.txt', Infer.isImageName('a.txt'), false],
];
for (const [name, got, want] of cases) {
  const ok = got === want;
  if (!ok) bad++;
  console.log(`[4] ${ok ? 'OK ' : 'NG '} ${name} → ${JSON.stringify(got)}${ok ? '' : ' (期望 ' + JSON.stringify(want) + ')'}`);
}

/* 连通域：两个 3×3 方块，间距 5 px */
{
  const w = 20; const h = 8;
  const mask = new Float32Array(w * h);
  const put = (x0, y0) => { for (let y = y0; y < y0 + 3; y++) for (let x = x0; x < x0 + 3; x++) mask[y * w + x] = 5; };
  put(1, 2); put(12, 3);
  const comps = Infer.components(mask, w, h, 4);
  const ok = comps.length === 2 && comps.every((b) => b.area === 9 && Math.abs(b.score - 5) < 1e-6);
  if (!ok) bad++;
  console.log(`[4] ${ok ? 'OK ' : 'NG '} components → ${JSON.stringify(comps.map((b) => [b.x1, b.y1, b.x2, b.y2, b.area]))}`);
}

/* 盒式均值：常量图均值等于常量 */
{
  const w = 8; const h = 6;
  const src = new Float32Array(w * h).fill(7);
  const out = Infer.boxMean(src, w, h, 2);
  const ok = Array.from(out).every((v) => Math.abs(v - 7) < 1e-5);
  if (!ok) bad++;
  console.log(`[4] ${ok ? 'OK ' : 'NG '} boxMean 常量保持 → ${out[0]}`);
}

console.log(bad ? `\n自检失败 ${bad} 项` : '\n自检全部通过');
process.exit(bad ? 1 : 0);
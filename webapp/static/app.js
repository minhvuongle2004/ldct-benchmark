/* ========================================================================
   app.js — EDR-REDNet Visualizer
   ======================================================================== */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  mode: 'mayo',
  patIdx: 0,
  sliceIdx: 0,
  huMin: -160,
  huMax: 245,
  showDiff: false,
  showEdge: false,
  // ablation
  tx: 200, ty: 250, ts: 20,
  bx: 100, by: 250, bs: 30,
  // zoom
  zx: 180, zy: 220, zw: 120, zh: 120,
  // chart instances
  charts: {},
  boxCharts: {},
  // debounce timer
  inferTimer: null,
  ablationTimer: null,
  zoomTimer: null,
};

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  fetchDeviceInfo();
  fetchPatients();
  loadBoxplot();
});

// ── Device Info ────────────────────────────────────────────────────────────
async function fetchDeviceInfo() {
  try {
    const r = await fetch('/api/info');
    const d = await r.json();
    const badge = document.getElementById('deviceBadge');
    if (d.cuda_name) {
      badge.textContent = `✅ GPU: ${d.cuda_name}`;
      badge.style.color = 'var(--green)';
    } else {
      badge.textContent = '⚠️ CPU mode (no GPU detected)';
      badge.style.color = 'var(--orange)';
    }
  } catch {
    document.getElementById('deviceBadge').textContent = '❌ Server not reachable';
  }
}

// ── Patients ───────────────────────────────────────────────────────────────
async function fetchPatients() {
  try {
    const r = await fetch('/api/patients');
    const patients = await r.json();
    const sel = document.getElementById('patientSelect');
    sel.innerHTML = '';
    patients.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.idx;
      opt.textContent = `${p.id} (${p.n_slices} slices)`;
      sel.appendChild(opt);
    });
    if (patients.length > 0) {
      const maxSlice = patients[0].n_slices - 1;
      const slider = document.getElementById('sliceSlider');
      slider.max = maxSlice;
      slider.value = Math.floor(maxSlice / 2);
      state.sliceIdx = parseInt(slider.value);
      document.getElementById('sliceVal').textContent = state.sliceIdx;
      runInfer();
      runAblation();
      runZoom();
    }
  } catch (e) {
    console.error('Failed to fetch patients:', e);
  }
}

function onPatientChange() {
  const sel = document.getElementById('patientSelect');
  state.patIdx = parseInt(sel.value);
  // Update slice max
  const opt = sel.options[sel.selectedIndex];
  const match = opt.textContent.match(/\((\d+) slices\)/);
  if (match) {
    const maxSlice = parseInt(match[1]) - 1;
    const slider = document.getElementById('sliceSlider');
    slider.max = maxSlice;
    slider.value = Math.floor(maxSlice / 2);
    state.sliceIdx = parseInt(slider.value);
    document.getElementById('sliceVal').textContent = state.sliceIdx;
  }
  debouncedInfer();
}

function onSliceChange(val) {
  state.sliceIdx = parseInt(val);
  document.getElementById('sliceVal').textContent = val;
  debouncedInfer();
}

function onHUChange() {
  state.huMin = parseInt(document.getElementById('huMinSlider').value);
  state.huMax = parseInt(document.getElementById('huMaxSlider').value);
  document.getElementById('huMinVal').textContent = state.huMin;
  document.getElementById('huMaxVal').textContent = state.huMax;
  debouncedInfer();
}

function onOptionsChange() {
  state.showDiff = document.getElementById('showDiff').checked;
  state.showEdge = document.getElementById('showEdge').checked;
  debouncedInfer();
}

// ── Debounce ───────────────────────────────────────────────────────────────
function debouncedInfer(ms = 400) {
  clearTimeout(state.inferTimer);
  state.inferTimer = setTimeout(runInfer, ms);
}

// ── Tab Switching ──────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  document.getElementById(`tab-${name}-btn`).classList.add('active');
}

// ── Sidebar Toggle ─────────────────────────────────────────────────────────
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const mw = document.querySelector('.main-wrapper');
  if (window.innerWidth > 900) {
    sb.classList.toggle('collapsed');
    mw.classList.toggle('expanded');
  } else {
    sb.classList.toggle('visible');
  }
}

// ── Mode Switching ─────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  document.getElementById('modeMayo').classList.toggle('active', mode === 'mayo');
  document.getElementById('modeUpload').classList.toggle('active', mode === 'upload');
  document.getElementById('mayoControls').classList.toggle('hidden', mode !== 'mayo');
  document.getElementById('uploadControls').classList.toggle('hidden', mode !== 'upload');
}

// ── Inference (Tab 1) ──────────────────────────────────────────────────────
async function runInfer() {
  if (state.mode !== 'mayo') return;
  showLoading('Đang chạy Inference...');
  try {
    const params = new URLSearchParams({
      pat_idx: state.patIdx,
      slice_idx: state.sliceIdx,
      hu_min: state.huMin,
      hu_max: state.huMax,
      show_diff: state.showDiff,
      show_edge: state.showEdge,
    });
    const r = await fetch(`/api/slice?${params}`);
    const data = await r.json();
    renderInferImages(data);
    renderMetrics(data.metrics);
    if (state.showDiff && data.diff) renderDiff(data.diff);
    if (state.showEdge && data.edge) renderEdge(data.edge);
    document.getElementById('diffSection').classList.toggle('hidden', !state.showDiff || !data.diff);
    document.getElementById('edgeSection').classList.toggle('hidden', !state.showEdge || !data.edge);
  } catch (e) {
    console.error('Inference failed:', e);
  } finally {
    hideLoading();
  }
}

function renderInferImages(data) {
  const m = {ldct: 'img-ldct', redcnn: 'img-redcnn', edr: 'img-edr', ndct: 'img-ndct'};
  for (const [key, id] of Object.entries(m)) {
    const el = document.getElementById(id);
    if (el && data.images[key]) { el.src = data.images[key]; el.style.opacity = 1; }
  }
}

function renderMetrics(metrics) {
  const edr = metrics.edr || {};
  const red = metrics.redcnn || {};

  const defs = [
    {id: 'mc-ssim-edr',  label: 'SSIM (EDR)',      key: 'SSIM',      fmt: v => v.toFixed(4)},
    {id: 'mc-psnr-edr',  label: 'PSNR (EDR)',      key: 'PSNR',      fmt: v => v.toFixed(2) + ' dB'},
    {id: 'mc-edge-edr',  label: 'Edge SSIM (EDR)', key: 'Edge_SSIM', fmt: v => v.toFixed(4)},
    {id: 'mc-vif-edr',   label: 'VIF (EDR)',        key: 'VIF',       fmt: v => v != null ? v.toFixed(4) : '—'},
  ];
  defs.forEach(def => {
    const card = document.getElementById(def.id);
    if (!card) return;
    const val = edr[def.key];
    const redVal = red[def.key];
    const valEl = card.querySelector('.metric-value');
    const deltaEl = card.querySelector('.metric-delta');
    valEl.textContent = val != null ? def.fmt(val) : '—';
    if (val != null && redVal != null) {
      const delta = val - redVal;
      const sign = delta >= 0 ? '+' : '';
      deltaEl.textContent = `${sign}${delta.toFixed(4)} vs RED-CNN`;
      deltaEl.className = 'metric-delta ' + (delta >= 0 ? 'pos' : 'neg');
    }
  });

  // Table
  const tbody = document.getElementById('metricsTableBody');
  tbody.innerHTML = '';
  const keys = ['SSIM', 'PSNR', 'VIF', 'Edge_SSIM'];
  const labels = {'SSIM': 'SSIM ↑', 'PSNR': 'PSNR (dB) ↑', 'VIF': 'VIF ↑', 'Edge_SSIM': 'Edge SSIM ↑'};
  keys.forEach(k => {
    const rv = red[k], ev = edr[k];
    if (rv == null && ev == null) return;
    const delta = (rv != null && ev != null) ? (ev - rv) : null;
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${labels[k]}</td>
      <td>${rv != null ? rv.toFixed(4) : '—'}</td>
      <td class="${ev != null && rv != null && ev >= rv ? 'cell-best' : ''}">${ev != null ? ev.toFixed(4) : '—'}</td>
      <td class="${delta != null ? (delta >= 0 ? 'cell-pos' : 'cell-neg') : ''}">${delta != null ? (delta >= 0 ? '+' : '') + delta.toFixed(4) : '—'}</td>
    `;
    tbody.appendChild(row);
  });
  document.getElementById('metricsTableSection').classList.remove('hidden');
}

function renderDiff(diff) {
  ['ldct', 'redcnn', 'edr'].forEach(k => {
    const el = document.getElementById(`diff-${k}`);
    if (el && diff[k]) el.src = diff[k];
  });
}

function renderEdge(edge) {
  ['ldct', 'redcnn', 'edr', 'ndct'].forEach(k => {
    const el = document.getElementById(`edge-${k}`);
    if (el && edge[k]) el.src = edge[k];
  });
}

// ── Upload (Tab 1) ─────────────────────────────────────────────────────────
function onFileSelect(type, input) {
  const name = input.files[0]?.name || 'Chọn file .dcm';
  document.getElementById(`${type}FileName`).textContent = name;
  const ldct = document.getElementById('ldctFile');
  document.getElementById('uploadBtn').disabled = !ldct.files.length;
}

async function runUpload() {
  const ldct = document.getElementById('ldctFile');
  const ndct = document.getElementById('ndctFile');
  if (!ldct.files.length) return;

  showLoading('Đang xử lý file DICOM...');
  try {
    const fd = new FormData();
    fd.append('ldct_file', ldct.files[0]);
    if (ndct.files.length) fd.append('ndct_file', ndct.files[0]);
    fd.append('hu_min', state.huMin);
    fd.append('hu_max', state.huMax);

    const r = await fetch('/api/upload', {method: 'POST', body: fd});
    const data = await r.json();
    renderInferImages(data);
    if (data.metrics && Object.keys(data.metrics).length) renderMetrics(data.metrics);
  } catch (e) {
    console.error('Upload failed:', e);
  } finally {
    hideLoading();
  }
}

// ── Ablation (Tab 2) ───────────────────────────────────────────────────────
function onROIChange() {
  state.tx = parseInt(document.getElementById('txSlider').value);
  state.ty = parseInt(document.getElementById('tySlider').value);
  state.ts = parseInt(document.getElementById('tsSlider').value);
  state.bx = parseInt(document.getElementById('bxSlider').value);
  state.by = parseInt(document.getElementById('bySlider').value);
  state.bs = parseInt(document.getElementById('bsSlider').value);
  document.getElementById('txVal').textContent = state.tx;
  document.getElementById('tyVal').textContent = state.ty;
  document.getElementById('tsVal').textContent = state.ts;
  document.getElementById('bxVal').textContent = state.bx;
  document.getElementById('byVal').textContent = state.by;
  document.getElementById('bsVal').textContent = state.bs;
  // Just update preview indicator, don't auto-run inference (expensive)
}

async function runAblation() {
  showLoading('Đang chạy Ablation (4 Variants)...');
  try {
    const params = new URLSearchParams({
      pat_idx: state.patIdx, slice_idx: state.sliceIdx,
      hu_min: state.huMin, hu_max: state.huMax,
      tx: state.tx, ty: state.ty, ts: state.ts,
      bx: state.bx, by: state.by, bs: state.bs,
    });
    const r = await fetch(`/api/ablation?${params}`);
    const data = await r.json();
    renderAblation(data);
  } catch(e) {
    console.error('Ablation failed:', e);
  } finally {
    hideLoading();
  }
}

const VARIANT_ORDER = ['LDCT (Input)', 'A — RED-CNN (Baseline)', 'B — +EdgeBlock', 'C — +Sobel Input', 'D — Full EDR-REDNet', 'NDCT (Ref)'];
const VARIANT_COLORS = {
  'A — RED-CNN (Baseline)': '#888888',
  'B — +EdgeBlock': '#4e9af1',
  'C — +Sobel Input': '#f1a74e',
  'D — Full EDR-REDNet': '#2ecc71',
};
const METRIC_KEYS = ['SSIM', 'PSNR', 'VIF', 'Edge_SSIM', 'CNR', 'HU_Dev'];

function renderAblation(data) {
  // ROI Preview
  const roiImg = document.getElementById('roi-preview');
  if (data.roi_preview) roiImg.src = data.roi_preview;

  // Find best per metric (among variants, not LDCT/NDCT)
  const variantKeys = Object.keys(data.metrics).filter(k => data.metrics[k] != null);
  const bestVals = {};
  METRIC_KEYS.forEach(mk => {
    const vals = variantKeys.map(k => data.metrics[k]?.[mk]).filter(v => v != null);
    if (!vals.length) return;
    bestVals[mk] = mk === 'HU_Dev' ? Math.min(...vals) : Math.max(...vals);
  });

  // Variant grid
  const grid = document.getElementById('ablationGrid');
  grid.innerHTML = '';
  const allLabels = ['LDCT (Input)', ...variantKeys, 'NDCT (Ref)'];
  allLabels.forEach(lbl => {
    const img = data.images[lbl];
    const card = document.createElement('div');
    const isBest = lbl === 'D — Full EDR-REDNet';
    card.className = `abl-card${isBest ? ' best' : ''}`;

    const shortLbl = lbl.split('—')[0].trim();
    const color = data.colors?.[lbl] || '#888';
    let metricsHtml = '';
    const m = data.metrics[lbl];
    if (m) {
      METRIC_KEYS.forEach(mk => {
        if (m[mk] == null) return;
        const isBestMk = bestVals[mk] != null && Math.abs(m[mk] - bestVals[mk]) < 1e-9;
        const fmt = mk === 'PSNR' ? m[mk].toFixed(2) : m[mk].toFixed(4);
        metricsHtml += `<div class="abl-metric-row"><span>${mk}</span><span class="${isBestMk ? 'best' : ''}">${isBestMk ? '⭐ ' : ''}${fmt}</span></div>`;
      });
    }

    card.innerHTML = `
      <div class="abl-header" style="border-left: 3px solid ${color}">${shortLbl}</div>
      <div class="img-wrapper">${img ? `<img class="ct-img" src="${img}" />` : '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:0.7rem">Chưa có model</div>'}</div>
      <div class="abl-metrics">${metricsHtml}</div>
    `;
    grid.appendChild(card);
  });

  // Ablation table
  const tbody = document.getElementById('ablationTableBody');
  tbody.innerHTML = '';
  variantKeys.forEach(vname => {
    const m = data.metrics[vname];
    if (!m) return;
    const row = document.createElement('tr');
    const cells = [vname.replace(' — ', '<br><small style="color:var(--text-sec)">'), ...METRIC_KEYS.map(mk => {
      if (m[mk] == null) return '—';
      const isBest = bestVals[mk] != null && Math.abs(m[mk] - bestVals[mk]) < 1e-9;
      const fmt = mk === 'PSNR' ? m[mk].toFixed(2) : m[mk].toFixed(4);
      return `<span class="${isBest ? 'cell-best' : ''}">${isBest ? '⭐ ' : ''}${fmt}</span>`;
    })];
    row.innerHTML = cells.map(c => `<td>${c}</td>`).join('');
    tbody.appendChild(row);
  });

  // Bar charts
  METRIC_KEYS.forEach(mk => {
    const canvasId = `chart-${mk}`;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const labels = variantKeys.map(k => k.split('—')[0].trim());
    const vals = variantKeys.map(k => data.metrics[k]?.[mk] ?? 0);
    const colors = variantKeys.map(k => VARIANT_COLORS[k] || '#63cab7');
    const min = Math.min(...vals), max = Math.max(...vals);
    const range = max - min || 0.001;
    if (state.charts[canvasId]) state.charts[canvasId].destroy();
    state.charts[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: { labels, datasets: [{ data: vals, backgroundColor: colors, borderWidth: 2,
        borderColor: vals.map((v, i) => Math.abs(v - (mk === 'HU_Dev' ? min : max)) < 1e-9 ? '#fbbf24' : 'transparent') }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: {display: false}, title: {display: true, text: mk, color: '#e8edf5', font: {size: 12, weight: 'bold'}} },
        scales: {
          x: { ticks: { color: '#8892a4', font: {size: 9} }, grid: {color: 'rgba(255,255,255,0.05)'} },
          y: { min: min - range * 0.15, max: max + range * 0.15, ticks: { color: '#8892a4' }, grid: {color: 'rgba(255,255,255,0.05)'} }
        }
      }
    });
  });
}

// ── Zoom (Tab 3) ───────────────────────────────────────────────────────────
function updateZoom() {
  state.zx = parseInt(document.getElementById('zxSlider').value);
  state.zy = parseInt(document.getElementById('zySlider').value);
  state.zw = parseInt(document.getElementById('zwSlider').value);
  state.zh = parseInt(document.getElementById('zhSlider').value);
  document.getElementById('zxVal').textContent = state.zx;
  document.getElementById('zyVal').textContent = state.zy;
  document.getElementById('zwVal').textContent = state.zw;
  document.getElementById('zhVal').textContent = state.zh;
}

async function runZoom() {
  showLoading('Đang tạo hình Zoom...');
  try {
    const params = new URLSearchParams({
      pat_idx: state.patIdx, slice_idx: state.sliceIdx,
      x: state.zx, y: state.zy, w: state.zw, h: state.zh,
      hu_min: state.huMin, hu_max: state.huMax,
    });
    const r = await fetch(`/api/zoom?${params}`);
    const data = await r.json();
    renderZoom(data);
  } catch(e) {
    console.error('Zoom failed:', e);
  } finally {
    hideLoading();
  }
}

function makeZoomImgCard(label, src, wrapClass) {
  return `<div class="zoom-img-card">
    <div class="zoom-img-label">${label}</div>
    <div class="${wrapClass}"><img src="${src}" loading="lazy" /></div>
  </div>`;
}

function renderZoom(data) {
  const fullGrid = document.getElementById('zoomFullGrid');
  const cropGrid = document.getElementById('zoomCropGrid');
  const diffGrid = document.getElementById('zoomDiffGrid');
  const sobelGrid = document.getElementById('zoomSobelGrid');

  fullGrid.innerHTML = '';
  cropGrid.innerHTML = '';
  diffGrid.innerHTML = '';
  sobelGrid.innerHTML = '';

  for (const [lbl, src] of Object.entries(data.full || {})) {
    fullGrid.innerHTML += makeZoomImgCard(lbl, src, 'zoom-full-wrap');
  }
  for (const [lbl, src] of Object.entries(data.zoom || {})) {
    cropGrid.innerHTML += makeZoomImgCard(lbl, src, 'zoom-img-wrap');
  }
  for (const [lbl, src] of Object.entries(data.diff || {})) {
    diffGrid.innerHTML += makeZoomImgCard(lbl, src, 'zoom-img-wrap');
  }
  for (const [lbl, src] of Object.entries(data.sobel || {})) {
    sobelGrid.innerHTML += makeZoomImgCard(lbl, src, 'zoom-img-wrap');
  }
}

// ── Boxplot (Tab 3) ────────────────────────────────────────────────────────
async function loadBoxplot() {
  const status = document.getElementById('boxplotStatus');
  try {
    const r = await fetch('/api/boxplot');
    if (!r.ok) {
      status.className = 'status-banner warning';
      status.textContent = '⚠️ Chưa có file per_patient_scores.csv. Hãy chạy evaluate_statistical_test.py trước.';
      return;
    }
    const data = await r.json();
    status.className = 'status-banner success';
    status.textContent = '✅ Đã tải dữ liệu boxplot thành công.';
    renderBoxplots(data);
  } catch {
    status.className = 'status-banner warning';
    status.textContent = '⚠️ Không thể tải dữ liệu boxplot.';
  }
}

const BP_COLORS = ['#888888', '#4e9af1', '#f1a74e', '#2ecc71'];

function renderBoxplots(data) {
  const metrics = ['PSNR', 'SSIM', 'Edge_SSIM'];
  const labels_map = {PSNR: 'PSNR (dB) ↑', SSIM: 'SSIM ↑', Edge_SSIM: 'Edge SSIM ↑'};
  const variants = Object.keys(data.data);

  metrics.forEach(metric => {
    const canvasId = `bp-${metric}`;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Build boxplot datasets from raw arrays — compute min/q1/med/q3/max
    const datasets = variants.map((vname, i) => {
      const vals = (data.data[vname]?.[metric] || []).slice().sort((a, b) => a - b);
      if (!vals.length) return null;
      const q = p => { const idx = (vals.length - 1) * p; const lo = Math.floor(idx), hi = Math.ceil(idx); return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo); };
      return {
        label: vname.split('—')[0].trim(),
        data: [{ min: vals[0], q1: q(0.25), median: q(0.5), q3: q(0.75), max: vals[vals.length-1] }],
        backgroundColor: BP_COLORS[i] + '55',
        borderColor: BP_COLORS[i],
        borderWidth: 2,
      };
    }).filter(Boolean);

    if (state.boxCharts[canvasId]) state.boxCharts[canvasId].destroy();

    // Use bar chart to simulate boxplot (Chart.js boxplot requires plugin)
    // Render median + range as bar + error bars via custom drawing
    const medians = datasets.map(d => d.data[0].median);
    const q1s = datasets.map(d => d.data[0].q1);
    const q3s = datasets.map(d => d.data[0].q3);
    const colors = datasets.map((_, i) => BP_COLORS[i]);
    const lbls = datasets.map(d => d.label);

    state.boxCharts[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: lbls,
        datasets: [
          { label: 'Median', data: medians, backgroundColor: colors.map(c => c + '88'), borderColor: colors, borderWidth: 2 },
          { label: 'Q1', data: q1s,    backgroundColor: 'transparent', borderColor: 'transparent', borderWidth: 0 },
          { label: 'Q3', data: q3s,    backgroundColor: 'transparent', borderColor: 'transparent', borderWidth: 0 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: {display: false},
          title: {display: true, text: labels_map[metric], color: '#e8edf5', font: {size: 12, weight: 'bold'}},
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const i = ctx.dataIndex;
                const d = datasets[i]?.data[0];
                return d ? [`Q1: ${d.q1.toFixed(4)}`, `Median: ${d.median.toFixed(4)}`, `Q3: ${d.q3.toFixed(4)}`] : [];
              }
            }
          }
        },
        scales: {
          x: { ticks: { color: '#8892a4', font: {size: 9} }, grid: {color: 'rgba(255,255,255,0.05)'} },
          y: { ticks: { color: '#8892a4' }, grid: {color: 'rgba(255,255,255,0.05)'} }
        }
      }
    });
  });

  // P-value table
  if (data.pvalues) {
    const pvalDiv = document.getElementById('pvalTable');
    pvalDiv.classList.remove('hidden');
    let html = '<h4 class="subsection-title">Kết quả Wilcoxon Signed-Rank Test</h4><table class="metrics-table"><thead><tr>';
    const keys = Object.keys(data.pvalues[0]);
    html += keys.map(k => `<th>${k}</th>`).join('') + '</tr></thead><tbody>';
    data.pvalues.forEach(row => {
      html += '<tr>' + keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>';
    });
    html += '</tbody></table>';
    pvalDiv.innerHTML = html;
  }
}

// ── Loading Helpers ────────────────────────────────────────────────────────
function showLoading(text = 'Đang xử lý...') {
  document.getElementById('loadingText').textContent = text;
  document.getElementById('loadingOverlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loadingOverlay').classList.add('hidden');
}

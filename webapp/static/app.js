/* ========================================================================
   app.js — EDR-REDNet Visualizer
   ======================================================================== */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  mode: 'mayo',
  patIdx: 0,
  sliceIdx: 0,
  uploadSessionId: null,
  uploadSliceIdx: 0,
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
      const sliceInput = document.getElementById('sliceVal');
      sliceInput.max = maxSlice;
      sliceInput.value = state.sliceIdx;
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
    const sliceInput = document.getElementById('sliceVal');
    sliceInput.max = maxSlice;
    sliceInput.value = state.sliceIdx;
  }
  debouncedUpdate();
}

function onSliceChange(val) {
  state.sliceIdx = parseInt(val);
  document.getElementById('sliceVal').value = val;
  debouncedUpdate();
}

function onSliceInput(val) {
  let parsed = parseInt(val);
  const max = parseInt(document.getElementById('sliceSlider').max);
  if (isNaN(parsed) || parsed < 0) parsed = 0;
  if (parsed > max) parsed = max;
  
  document.getElementById('sliceSlider').value = parsed;
  document.getElementById('sliceVal').value = parsed;
  state.sliceIdx = parsed;
  debouncedUpdate();
}

function onHUChange() {
  state.huMin = parseInt(document.getElementById('huMinSlider').value);
  state.huMax = parseInt(document.getElementById('huMaxSlider').value);
  document.getElementById('huMinVal').textContent = state.huMin;
  document.getElementById('huMaxVal').textContent = state.huMax;
  debouncedUpdate();
}

function onOptionsChange() {
  state.showDiff = document.getElementById('showDiff').checked;
  state.showEdge = document.getElementById('showEdge').checked;
  debouncedUpdate();
}

// ── Debounce ───────────────────────────────────────────────────────────────
function debouncedUpdate(ms = 400) {
  clearTimeout(state.inferTimer);
  state.inferTimer = setTimeout(() => {
    if (document.getElementById('tab-infer').classList.contains('active')) {
      if (state.mode === 'mayo') runInfer();
      else if (state.mode === 'upload' && state.uploadSessionId) runUploadSlice();
    }
    if (document.getElementById('tab-ablation').classList.contains('active') && state.mode === 'mayo') runAblation();
    if (document.getElementById('tab-paper').classList.contains('active') && state.mode === 'mayo') runZoom();
  }, ms);
}

// ── Tab Switching ──────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  document.getElementById(`tab-${name}-btn`).classList.add('active');
  debouncedUpdate(50); // Refresh data for the newly opened tab
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
function onFolderSelect(input) {
  const files = input.files;
  if (!files || !files.length) return;
  
  // Count how many .dcm files are in the selected folder
  let dcmCount = 0;
  for (let i = 0; i < files.length; i++) {
    if (files[i].name.toLowerCase().endsWith('.dcm')) dcmCount++;
  }
  
  document.getElementById('ldctFolderName').textContent = `Đã chọn ${dcmCount} file .dcm`;
  
  if (dcmCount > 0) {
    runFolderUpload(files);
  } else {
    alert("Không tìm thấy file .dcm nào trong thư mục đã chọn!");
  }
}

async function runFolderUpload(files) {
  showLoading('Đang tải lên và xử lý thư mục...');
  try {
    const fd = new FormData();
    for (let i = 0; i < files.length; i++) {
      if (files[i].name.toLowerCase().endsWith('.dcm')) {
        fd.append('files', files[i]);
      }
    }

    const r = await fetch('/api/upload_folder', {method: 'POST', body: fd});
    const data = await r.json();
    
    if (r.ok) {
      state.uploadSessionId = data.session_id;
      const maxSlice = data.total_slices - 1;
      
      // Init slider
      const slider = document.getElementById('uploadSliceSlider');
      const valInput = document.getElementById('uploadSliceVal');
      slider.max = maxSlice;
      slider.value = Math.floor(maxSlice / 2);
      valInput.max = maxSlice;
      valInput.value = slider.value;
      state.uploadSliceIdx = parseInt(slider.value);
      
      document.getElementById('uploadSliderContainer').classList.remove('hidden');
      
      // Fetch the first slice to show
      runUploadSlice();
    } else {
      alert("Lỗi tải lên: " + data.detail);
    }
  } catch (e) {
    console.error('Upload failed:', e);
    alert("Lỗi tải lên: " + e.message);
  } finally {
    hideLoading();
  }
}

function onUploadSliceChange(val) {
  state.uploadSliceIdx = parseInt(val);
  document.getElementById('uploadSliceVal').value = val;
  debouncedUpdate();
}

function onUploadSliceInput(val) {
  let parsed = parseInt(val);
  const max = parseInt(document.getElementById('uploadSliceSlider').max);
  if (isNaN(parsed) || parsed < 0) parsed = 0;
  if (parsed > max) parsed = max;
  
  document.getElementById('uploadSliceSlider').value = parsed;
  document.getElementById('uploadSliceVal').value = parsed;
  state.uploadSliceIdx = parsed;
  debouncedUpdate();
}

async function runUploadSlice() {
  if (!state.uploadSessionId) return;
  showLoading('Đang khử nhiễu lát cắt...');
  try {
    const params = new URLSearchParams({
      session_id: state.uploadSessionId,
      slice_idx: state.uploadSliceIdx,
      hu_min: state.huMin,
      hu_max: state.huMax
    });
    const r = await fetch(`/api/upload_slice?${params}`);
    const data = await r.json();
    renderInferImages(data);
    // Hide diff/edge for upload since NDCT is not available
    document.getElementById('diffSection').classList.add('hidden');
    document.getElementById('edgeSection').classList.add('hidden');
    // Hide metrics table since we don't calculate metrics
    document.getElementById('metricsTableSection').classList.add('hidden');
    
    // Reset metric cards
    ['mc-ssim-edr', 'mc-psnr-edr', 'mc-edge-edr', 'mc-vif-edr'].forEach(id => {
      const card = document.getElementById(id);
      if (card) {
        card.querySelector('.metric-value').textContent = '—';
        card.querySelector('.metric-delta').textContent = '—';
      }
    });
  } catch (e) {
    console.error('Upload slice failed:', e);
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

// ── Sybil Prediction (SSE) ─────────────────────────────────────────────────
let sybilEventSource = null;
let sybilChartInstance = null;

function runSybilPrediction() {
  const btn = document.getElementById('sybilBtn');
  btn.disabled = true;

  // Show modal, reset state
  const modal = document.getElementById('sybilModal');
  modal.classList.remove('hidden');
  document.getElementById('sybilProgress').style.display = 'block';
  document.getElementById('sybilResults').classList.add('hidden');
  document.getElementById('sybilCloseBtn').classList.add('hidden');
  document.getElementById('sybilProgressBar').style.width = '0%';
  document.getElementById('sybilPercentText').textContent = '0%';
  document.getElementById('sybilStageText').textContent = 'Đang khởi tạo...';

  // Open SSE
  let url = `/api/predict_sybil?pat_idx=${state.patIdx}`;
  if (state.mode === 'upload' && state.uploadSessionId) {
    url = `/api/predict_sybil?session_id=${state.uploadSessionId}`;
  } else if (state.mode === 'upload' && !state.uploadSessionId) {
    alert("Vui lòng tải lên một thư mục DICOM trước khi chạy Sybil!");
    btn.disabled = false;
    modal.classList.add('hidden');
    return;
  }
  sybilEventSource = new EventSource(url);

  sybilEventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    // Update progress bar
    const pct = data.percent || 0;
    document.getElementById('sybilProgressBar').style.width = pct + '%';
    document.getElementById('sybilPercentText').textContent = Math.round(pct) + '%';
    document.getElementById('sybilStageText').textContent = data.message || '';

    if (data.stage === 'done') {
      sybilEventSource.close();
      sybilEventSource = null;
      btn.disabled = false;

      // Show results
      document.getElementById('sybilProgress').style.display = 'none';
      document.getElementById('sybilResults').classList.remove('hidden');
      document.getElementById('sybilCloseBtn').classList.remove('hidden');
      renderSybilResults(data.results);
    }
  };

  sybilEventSource.onerror = function(err) {
    console.error('Sybil SSE error:', err);
    sybilEventSource.close();
    sybilEventSource = null;
    btn.disabled = false;
    document.getElementById('sybilStageText').textContent = '❌ Lỗi kết nối. Hãy thử lại.';
    document.getElementById('sybilCloseBtn').classList.remove('hidden');
  };
}

function closeSybilModal() {
  document.getElementById('sybilModal').classList.add('hidden');
  if (sybilEventSource) {
    sybilEventSource.close();
    sybilEventSource = null;
  }
  document.getElementById('sybilBtn').disabled = false;
}

function renderSybilResults(results) {
  // Patient info
  const info = document.getElementById('sybilPatientInfo');
  info.innerHTML = `📋 Bệnh nhân: <strong>${results.patient_id}</strong> — ${results.total_slices} lát cắt đã phân tích`;

  // Table
  const tbody = document.getElementById('sybilTableBody');
  tbody.innerHTML = '';

  const rows = [
    { label: '🔴 LDCT (Gốc)',     scores: results.scores_ldct,   cls: 'row-ldct' },
    { label: '🔵 RED-CNN',         scores: results.scores_redcnn, cls: 'row-redcnn' },
    { label: '🟢 EDR-REDNet',      scores: results.scores_edr,    cls: 'row-edr' },
  ];

  // For each year, find the lowest risk (best)
  const numYears = Math.max(
    results.scores_ldct?.length || 0,
    results.scores_redcnn?.length || 0,
    results.scores_edr?.length || 0
  );

  const bestPerYear = [];
  for (let y = 0; y < numYears; y++) {
    const vals = rows.map(r => {
      const s = r.scores?.[y];
      return (typeof s === 'number') ? s : Infinity;
    });
    bestPerYear.push(Math.min(...vals));
  }

  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.className = row.cls;
    let html = `<td>${row.label}</td>`;
    const scores = row.scores || [];
    for (let y = 0; y < numYears; y++) {
      const val = scores[y];
      if (typeof val === 'number') {
        const isBest = Math.abs(val - bestPerYear[y]) < 0.001;
        html += `<td class="${isBest ? 'cell-best' : ''}">${val.toFixed(2)}%</td>`;
      } else {
        html += `<td style="color:#ef4444;font-size:0.7rem">${val || '—'}</td>`;
      }
    }
    tr.innerHTML = html;
    tbody.appendChild(tr);
  });

  // Chart
  renderSybilChart(results, numYears);
}

function renderSybilChart(results, numYears) {
  const canvas = document.getElementById('sybilChart');
  if (sybilChartInstance) {
    sybilChartInstance.destroy();
    sybilChartInstance = null;
  }

  const labels = Array.from({length: numYears}, (_, i) => `Năm ${i + 1}`);

  const datasets = [
    {
      label: 'LDCT (Gốc)',
      data: (results.scores_ldct || []).map(v => typeof v === 'number' ? v : null),
      borderColor: '#f87171',
      backgroundColor: 'rgba(248, 113, 113, 0.15)',
      borderWidth: 2.5,
      pointRadius: 5,
      pointBackgroundColor: '#f87171',
      tension: 0.3,
      fill: true,
    },
    {
      label: 'RED-CNN',
      data: (results.scores_redcnn || []).map(v => typeof v === 'number' ? v : null),
      borderColor: '#60a5fa',
      backgroundColor: 'rgba(96, 165, 250, 0.15)',
      borderWidth: 2.5,
      pointRadius: 5,
      pointBackgroundColor: '#60a5fa',
      tension: 0.3,
      fill: true,
    },
    {
      label: 'EDR-REDNet',
      data: (results.scores_edr || []).map(v => typeof v === 'number' ? v : null),
      borderColor: '#34d399',
      backgroundColor: 'rgba(52, 211, 153, 0.15)',
      borderWidth: 2.5,
      pointRadius: 5,
      pointBackgroundColor: '#34d399',
      tension: 0.3,
      fill: true,
    },
  ];

  sybilChartInstance = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { size: 11 } }
        },
        title: {
          display: true,
          text: 'Nguy cơ Ung thư phổi theo Năm (%)',
          color: '#e8edf5',
          font: { size: 13, weight: 'bold' }
        },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#8892a4' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#8892a4', callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          title: { display: true, text: 'Nguy cơ (%)', color: '#8892a4' }
        }
      }
    }
  });
}

// ── Sybil History ──────────────────────────────────────────────────────────
let sybilHistoryData = [];

async function showSybilHistory() {
  document.getElementById('sybilHistoryModal').classList.remove('hidden');
  const listEl = document.getElementById('sybilHistoryList');
  listEl.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:2rem;">Đang tải lịch sử...</div>';
  
  try {
    const r = await fetch('/api/sybil_history');
    if (!r.ok) throw new Error('Failed to fetch history');
    sybilHistoryData = await r.json();
    
    if (sybilHistoryData.length === 0) {
      listEl.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:2rem;">Chưa có lịch sử dự đoán nào.</div>';
      return;
    }
    
    let html = '';
    sybilHistoryData.forEach((item, index) => {
      const riskLdct = item.scores_ldct && item.scores_ldct.length > 0 ? (typeof item.scores_ldct[0] === 'number' ? item.scores_ldct[0].toFixed(2) + '%' : 'Error') : '—';
      const riskEdr = item.scores_edr && item.scores_edr.length > 0 ? (typeof item.scores_edr[0] === 'number' ? item.scores_edr[0].toFixed(2) + '%' : 'Error') : '—';
      
      html += `
        <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:12px 16px; margin-bottom:10px; cursor:pointer; transition:0.2s;" 
             onmouseover="this.style.borderColor='#4f46e5'" 
             onmouseout="this.style.borderColor='#334155'"
             onclick="viewHistoryItem(${index})">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="font-weight:600; color:#e8edf5; font-size:1.05rem;">Bệnh nhân: ${item.patient_id}</div>
            <div style="color:#94a3b8; font-size:0.85rem;">🕒 ${item.timestamp || ''}</div>
          </div>
          <div style="display:flex; gap:15px; color:#94a3b8; font-size:0.9rem;">
            <div>Số lát cắt: <span style="color:#e8edf5;">${item.total_slices}</span></div>
            <div>Rủi ro Năm 1 (Gốc): <span style="color:#f87171;">${riskLdct}</span></div>
            <div>Rủi ro Năm 1 (EDR): <span style="color:#34d399;">${riskEdr}</span></div>
          </div>
        </div>
      `;
    });
    listEl.innerHTML = html;
  } catch (e) {
    console.error(e);
    listEl.innerHTML = '<div style="color:#ef4444;text-align:center;padding:2rem;">Lỗi tải lịch sử!</div>';
  }
}

function closeSybilHistoryModal() {
  document.getElementById('sybilHistoryModal').classList.add('hidden');
}

function viewHistoryItem(index) {
  const data = sybilHistoryData[index];
  if (!data) return;
  
  closeSybilHistoryModal();
  
  // Show sybil modal directly with results
  const modal = document.getElementById('sybilModal');
  modal.classList.remove('hidden');
  document.getElementById('sybilProgress').style.display = 'none';
  document.getElementById('sybilResults').classList.remove('hidden');
  document.getElementById('sybilCloseBtn').classList.remove('hidden');
  
  renderSybilResults(data);
}

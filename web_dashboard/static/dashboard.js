/* =========================================================
   PERM Edge Control Center — Dashboard JS
   - Red forecast line plotted 30s AHEAD of current telemetry
   - Animated toast notifications for scaling events
   - Pulsing alert overlay when thresholds are crossed
   ========================================================= */

let cpuChart = null;
let latChart = null;
let activePendingRequestId = null;
let isPolling = false;
let _lastScalingHistoryLen = 0;
let _lastAlertHistoryLen = 0;

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  scheduleNextPoll(0);
});

function scheduleNextPoll(delayMs = 1000) {
  setTimeout(async () => {
    await pollDashboardState();
    scheduleNextPoll(1000);
  }, delayMs);
}

/* ─── CHART INIT ──────────────────────────────────────────── */
function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.03)', drawBorder: false },
        ticks: { color: '#64748b', font: { family: 'Inter', size: 10 }, maxTicksLimit: 12, maxRotation: 0 }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
        ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } }
      }
    },
    plugins: {
      legend: {
        position: 'top', align: 'end',
        labels: {
          color: '#cbd5e1',
          font: { family: 'Inter', size: 11, weight: '500' },
          boxWidth: 10, usePointStyle: true, pointStyle: 'circle'
        }
      },
      tooltip: {
        backgroundColor: '#0f172a', titleColor: '#f8fafc', bodyColor: '#cbd5e1',
        borderColor: '#1e293b', borderWidth: 1, padding: 10, cornerRadius: 6,
        titleFont: { family: 'Inter', size: 11, weight: '600' },
        bodyFont: { family: 'JetBrains Mono', size: 11 }
      }
    }
  };

  const ctxCpu = document.getElementById('chart-cpu').getContext('2d');
  cpuChart = new Chart(ctxCpu, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Telemetry CPU % (1s)',
          data: [],
          borderColor: '#38bdf8',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
          backgroundColor: 'rgba(56, 189, 248, 0.07)',
          order: 2
        },
        {
          label: '30s ML Forecast ↗',
          data: [],
          borderColor: '#f43f5e',
          borderWidth: 2.5,
          borderDash: [5, 3],
          pointRadius: 0,
          tension: 0.3,
          fill: false,
          spanGaps: true,
          order: 1
        },
        {
          label: 'SLO Warning (70%)',
          data: [],
          borderColor: 'rgba(245, 158, 11, 0.55)',
          borderWidth: 1,
          borderDash: [2, 4],
          pointRadius: 0,
          order: 3
        },
        {
          label: 'SLO Critical (85%)',
          data: [],
          borderColor: 'rgba(244, 63, 94, 0.55)',
          borderWidth: 1,
          borderDash: [2, 4],
          pointRadius: 0,
          order: 3
        }
      ]
    },
    options: {
      ...commonOptions,
      scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 105 } }
    }
  });

  const ctxLat = document.getElementById('chart-lat').getContext('2d');
  latChart = new Chart(ctxLat, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Telemetry Latency (ms)',
          data: [],
          borderColor: '#10b981',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
          backgroundColor: 'rgba(16, 185, 129, 0.07)',
          order: 2
        },
        {
          label: '30s ML Forecast ↗',
          data: [],
          borderColor: '#fbbf24',
          borderWidth: 2.5,
          borderDash: [5, 3],
          pointRadius: 0,
          tension: 0.3,
          fill: false,
          spanGaps: true,
          order: 1
        },
        {
          label: 'SLO Warning (150ms)',
          data: [],
          borderColor: 'rgba(245, 158, 11, 0.55)',
          borderWidth: 1,
          borderDash: [2, 4],
          pointRadius: 0,
          order: 3
        },
        {
          label: 'SLO Critical (300ms)',
          data: [],
          borderColor: 'rgba(244, 63, 94, 0.55)',
          borderWidth: 1,
          borderDash: [2, 4],
          pointRadius: 0,
          order: 3
        }
      ]
    },
    options: commonOptions
  });
}

/* ─── POLL ────────────────────────────────────────────────── */
async function pollDashboardState() {
  if (isPolling) return;
  isPolling = true;
  try {
    const resp = await fetch('/api/dashboard/state');
    if (!resp.ok) return;
    const data = await resp.json();

    updateMetrics(data);
    updateCharts(data);
    updateRemediation(data.remediation, data.controller_events);
    updateAlertsTable(data.alerts_history, data.controller_events);
    checkForNewScalingEvents(data.remediation);
    checkForNewAlerts(data.active_alerts, data.alerts_history);
  } catch (err) {
    console.debug('Dashboard fetch error:', err);
  } finally {
    isPolling = false;
  }
}

/* ─── METRICS ─────────────────────────────────────────────── */
function updateMetrics(data) {
  const telem = data.telemetry_latest || {};
  const pred  = data.prediction_latest || {};
  const remed = data.remediation || {};
  const ctrlState = data.controller_state || {};

  // Controller mode badge
  if (ctrlState.mode) {
    const badge = document.getElementById('controller-mode-badge');
    if (badge) badge.innerText = ctrlState.mode.toUpperCase();
  }

  // Pool utilization main gauge
  const poolUtil = (data.pool_utilization !== undefined)
    ? data.pool_utilization
    : (telem.worker_cpu_percent !== undefined
        ? Math.min(100.0, Math.max(0.0, (telem.worker_cpu_percent * 16.0) / Math.max(1, telem.active_workers || 2)))
        : 0.0);

  const elCpu = document.getElementById('val-cpu');
  if (elCpu) {
    elCpu.innerText = `${poolUtil.toFixed(1)}%`;
    elCpu.style.color = poolUtil >= 85 ? '#f43f5e'
                      : poolUtil >= 70 ? '#f59e0b'
                      : '#38bdf8';
  }

  if (telem.avg_latency_ms !== undefined) {
    document.getElementById('val-lat').innerText = `${telem.avg_latency_ms.toFixed(1)} ms`;
  }
  if (pred.predicted_latency_30s !== undefined) {
    const predLat = pred.predicted_latency_30s.toFixed(1);
    const drivers = (pred.top_drivers_latency || []).slice(0,2)
      .map(d => `<span class="driver-pill mono">⚡ ${d}</span>`).join(' ');
    const color = pred.predicted_latency_30s >= 300 ? '#f43f5e'
                : pred.predicted_latency_30s >= 150 ? '#f59e0b'
                : '#94a3b8';
    document.getElementById('sub-lat').innerHTML =
      `<span style="color:${color};font-weight:700;">${predLat} ms</span> ${drivers}`;
  }

  const activeWorkers = (ctrlState.current_workers !== undefined)
    ? ctrlState.current_workers
    : (telem.active_workers || remed.active_workers || 2);
  document.getElementById('val-workers').innerText = `${activeWorkers} / 8`;
  document.getElementById('sub-workers').innerText =
    `Mode: ${ctrlState.mode ? ctrlState.mode.toUpperCase() : 'DEMO CONTROLLER'}`;

  // System health badge
  const ctrlEvents = data.controller_events || [];
  const alertState = ctrlState.alert_state || 'NORMAL';
  const statusEl   = document.getElementById('val-status');
  const subAlertsEl = document.getElementById('sub-alerts');

  if (alertState === 'CRITICAL') {
    statusEl.innerHTML = `<span class="status-badge status-badge-crit pulse-badge">🔴 POOL CRITICAL (≥85%)</span>`;
    subAlertsEl.innerText = 'Pool Utilization Critical';
  } else if (alertState === 'WARN') {
    statusEl.innerHTML = `<span class="status-badge status-badge-warn pulse-badge">🟡 POOL WARNING (≥70%)</span>`;
    subAlertsEl.innerText = 'Pool Utilization Warning';
  } else {
    statusEl.innerHTML = `<span class="status-badge status-badge-ok">✅ SYSTEM NORMAL</span>`;
    subAlertsEl.innerText = 'Pool Utilization < 70%';
  }
}

/* ─── CHARTS — Smooth Leading Forecast Line & 30s Future Zone ── */
function updateCharts(data) {
  const telemHistory = data.telemetry_history || [];
  const predHistory  = data.prediction_history || [];
  if (telemHistory.length === 0) return;

  const lastTelemTs = new Date(telemHistory[telemHistory.length - 1].timestamp).getTime();

  // Create timeline: telemetry history + 30s into the future (5s steps)
  const fullTimeline = [...telemHistory.map(r => new Date(r.timestamp).getTime())];
  const lastTs = fullTimeline[fullTimeline.length - 1];
  for (let i = 1; i <= 6; i++) {  // 6 steps × 5s = +30s future zone
    fullTimeline.push(lastTs + i * 5000);
  }

  const labels = fullTimeline.map(ts => {
    const t = new Date(ts);
    return t.toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
  });

  // Pool utilization actuals (null for future points)
  const cpuActual = fullTimeline.map((ts, idx) => {
    if (idx < telemHistory.length) {
      const r = telemHistory[idx];
      const wCpu = r.worker_cpu_percent !== undefined ? r.worker_cpu_percent : (r.cpu_percent || 0);
      const wCount = r.active_workers || 2;
      return Math.min(100.0, Math.max(0.0, (wCpu * 16.0) / Math.max(1, wCount)));
    }
    return null;
  });
  const latActual = fullTimeline.map((ts, idx) =>
    idx < telemHistory.length ? telemHistory[idx].avg_latency_ms : null
  );

  const cpuWarn = new Array(labels.length).fill(70);
  const cpuCrit = new Array(labels.length).fill(85);
  const latWarn = new Array(labels.length).fill(150);
  const latCrit = new Array(labels.length).fill(300);

  // Latest generated prediction (used for Future Zone projection)
  const latestPred = predHistory.length > 0 ? predHistory[predHistory.length - 1] : null;

  // ── CPU FORECAST ───────────────────────────────────────────────────
  let lastCpuPred = null;
  const cpuForecast = fullTimeline.map((ts, idx) => {
    if (idx >= telemHistory.length && latestPred) {
      return latestPred.predicted_cpu_30s;
    }
    const match = predHistory.find(p => Math.abs(new Date(p.timestamp).getTime() - ts) <= 4000)
               || predHistory.find(p => {
                    const targetTs = p.target_timestamp_30s ? new Date(p.target_timestamp_30s).getTime() : 0;
                    return Math.abs(targetTs - ts) <= 4000;
                  });
    if (match) lastCpuPred = match.predicted_cpu_30s;
    return match ? match.predicted_cpu_30s : lastCpuPred;
  });

  // ── LATENCY FORECAST ───────────────────────────────────────────────
  let lastLatPred = null;
  const latForecast = fullTimeline.map((ts, idx) => {
    if (idx >= telemHistory.length && latestPred) {
      return latestPred.predicted_latency_30s;
    }
    const match = predHistory.find(p => Math.abs(new Date(p.timestamp).getTime() - ts) <= 4000)
               || predHistory.find(p => {
                    const targetTs = p.target_timestamp_30s ? new Date(p.target_timestamp_30s).getTime() : 0;
                    return Math.abs(targetTs - ts) <= 4000;
                  });
    if (match) lastLatPred = match.predicted_latency_30s;
    return match ? match.predicted_latency_30s : lastLatPred;
  });

  cpuChart.data.labels           = labels;
  cpuChart.data.datasets[0].label = 'Pool Utilization % (3s smoothed)';
  cpuChart.data.datasets[0].data = cpuActual;
  cpuChart.data.datasets[1].data = cpuForecast;
  cpuChart.data.datasets[2].data = cpuWarn;
  cpuChart.data.datasets[3].data = cpuCrit;
  cpuChart.update('none');

  latChart.data.labels           = labels;
  latChart.data.datasets[0].data = latActual;
  latChart.data.datasets[1].data = latForecast;
  latChart.data.datasets[2].data = latWarn;
  latChart.data.datasets[3].data = latCrit;
  latChart.update('none');
}

/* ─── REMEDIATION & SCALING AUDIT LOG ─────────────────────── */
function updateRemediation(remed, controllerEvents) {
  if (controllerEvents && controllerEvents.length > 0) {
    const scaleList = controllerEvents.filter(e => ['SCALE_UP', 'SCALE_DOWN'].includes(e.type));
    const tblHistory = document.getElementById('tbl-history');
    if (scaleList.length === 0) {
      tblHistory.innerHTML = `<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:1.5rem;">No scaling events logged yet</td></tr>`;
    } else {
      tblHistory.innerHTML = scaleList.slice(-10).reverse().map(h => {
        const t = new Date(h.time).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
        const badge = h.type === 'SCALE_UP'
          ? '<span class="tag-auto mono" style="color:#10b981;">SCALE_UP</span>'
          : '<span class="tag-manual mono" style="color:#f59e0b;">SCALE_DOWN</span>';
        const rulePart = (h.reason && h.reason.includes('rule=')) ? h.reason.split(';')[0].replace('rule=', '') : 'reactive';
        return `<tr>
          <td class="mono">${t}</td>
          <td>${badge}</td>
          <td class="mono" style="font-weight:600;">${h.value} workers</td>
          <td class="mono" style="color:var(--color-primary);">${rulePart}</td>
          <td style="color:var(--text-muted);font-size:0.75rem;">${h.reason}</td>
        </tr>`;
      }).join('');
    }
  }
}

/* ─── ALERTS TABLE (controller_events.csv) ────────────────── */
function updateAlertsTable(alerts, controllerEvents) {
  const tbl = document.getElementById('tbl-alerts');
  if (controllerEvents && controllerEvents.length > 0) {
    const alertList = controllerEvents.filter(e => ['WARN', 'CRITICAL', 'RECOVERY'].includes(e.type));
    if (alertList.length === 0) {
      tbl.innerHTML = `<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:1.5rem;">No controller alerts logged yet</td></tr>`;
      return;
    }
    tbl.innerHTML = alertList.slice(-10).reverse().map(a => {
      const t = new Date(a.time).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
      const badge = a.type === 'CRITICAL'
        ? '<span class="status-badge status-badge-crit">CRITICAL</span>'
        : (a.type === 'WARN'
            ? '<span class="status-badge status-badge-warn">WARN</span>'
            : '<span class="status-badge status-badge-ok">RECOVERY</span>');
      return `<tr>
        <td class="mono">${t}</td>
        <td>${badge}</td>
        <td class="mono">${a.signal || 'pool_utilization'}</td>
        <td class="mono" style="font-weight:600;">${a.value}%</td>
        <td style="color:var(--text-main);">${a.reason || ''}</td>
      </tr>`;
    }).join('');
    return;
  }

  if (!alerts || alerts.length === 0) {
    tbl.innerHTML = `<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:1.5rem;">No active alerts recorded</td></tr>`;
    return;
  }
  tbl.innerHTML = alerts.slice(-8).reverse().map(a => {
    const t = new Date(a.timestamp).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
    const badge = a.severity === 'CRITICAL'
      ? '<span class="status-badge status-badge-crit">CRITICAL</span>'
      : '<span class="status-badge status-badge-warn">WARNING</span>';
    return `<tr>
      <td class="mono">${t}</td>
      <td style="font-weight:500;">${a.metric}</td>
      <td>${badge}</td>
      <td class="mono">${typeof a.forecast_value_30s === 'number' ? a.forecast_value_30s.toFixed(1) : a.forecast_value_30s}</td>
      <td class="mono" style="color:var(--color-primary);font-weight:600;">~${a.lead_time_sec}s</td>
    </tr>`;
  }).join('');
}

/* ─── NEW SCALING EVENT → TOAST ───────────────────────────── */
function checkForNewScalingEvents(remed) {
  if (!remed) return;
  const history = remed.scaling_history || [];
  if (history.length > _lastScalingHistoryLen) {
    const latest = history[0]; // status endpoint returns scaling_history reversed (latest at index 0)
    if (latest) showScalingToast(latest);
  }
  _lastScalingHistoryLen = history.length;
}

function showScalingToast(event) {
  if (event.status === 'DECLINED') {
    const icon     = '⚠️';
    const colorCls = 'toast-warning';
    const title    = 'Manual Scaling Declined';
    const body     = `Operator dismissed scaling proposal — capacity remains constrained at ${event.workers_before} cores.`;
    const sub      = 'Expect CPU saturation & latency queuing bottleneck under load.';
    _spawnToast(icon, title, body, sub, colorCls, 8000);
    return;
  }

  const isAuto   = event.mode === 'AUTO';
  const icon     = isAuto ? '⚡' : '✅';
  const colorCls = isAuto ? 'toast-auto' : 'toast-manual';
  const title    = isAuto ? 'Auto-Scaling Triggered' : 'Manual Scaling Approved';
  const body     = `${event.workers_before} → ${event.workers_after} worker cores allocated`;
  const sub      = event.action || '';

  _spawnToast(icon, title, body, sub, colorCls, 6000);
}

/* ─── NEW ALERT → TOAST ───────────────────────────────────── */
function checkForNewAlerts(activeAlerts, alertHistory) {
  const histLen = alertHistory ? alertHistory.length : 0;
  if (histLen > _lastAlertHistoryLen) {
    const latest = alertHistory[alertHistory.length - 1];
    if (latest) showAlertToast(latest);
  }
  _lastAlertHistoryLen = histLen;
}

function showAlertToast(alert) {
  const isCrit   = alert.severity === 'CRITICAL';
  const icon     = isCrit ? '🚨' : '⚠️';
  const title    = isCrit ? 'PROACTIVE CRITICAL ALERT' : 'PROACTIVE WARNING ALERT';
  const body     = `${alert.metric} forecast: ${
    typeof alert.forecast_value_30s === 'number' ? alert.forecast_value_30s.toFixed(1) : alert.forecast_value_30s
  } — SLO breach in ~${alert.lead_time_sec}s`;
  const sub      = `Current value: ${typeof alert.current_value === 'number' ? alert.current_value.toFixed(1) : alert.current_value}`;
  const colorCls = isCrit ? 'toast-critical' : 'toast-warning';

  _spawnToast(icon, title, body, sub, colorCls, 8000);
}

/* ─── TOAST ENGINE ────────────────────────────────────────── */
function _spawnToast(icon, title, body, sub, colorClass, durationMs) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `perm-toast ${colorClass}`;
  toast.innerHTML = `
    <div class="toast-icon">${icon}</div>
    <div class="toast-body">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${body}</div>
      ${sub ? `<div class="toast-sub">${sub}</div>` : ''}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
    <div class="toast-progress"></div>
  `;

  container.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { toast.classList.add('toast-visible'); });
  });

  // Progress bar animation
  const bar = toast.querySelector('.toast-progress');
  if (bar) {
    bar.style.transition = `width ${durationMs}ms linear`;
    requestAnimationFrame(() => { requestAnimationFrame(() => { bar.style.width = '0%'; }); });
  }

  // Auto-dismiss
  setTimeout(() => {
    toast.classList.remove('toast-visible');
    setTimeout(() => toast.remove(), 400);
  }, durationMs);
}

/* ─── USER ACTIONS ────────────────────────────────────────── */
async function setScalingMode(mode) {
  try {
    await fetch('/api/scaling/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    pollDashboardState();
  } catch (e) { console.error('Failed to set scaling mode:', e); }
}

async function approvePendingScaling() {
  try {
    await fetch('/api/scaling/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: activePendingRequestId })
    });
    pollDashboardState();
  } catch (e) { console.error('Failed to approve scaling:', e); }
}

async function declinePendingScaling() {
  try {
    await fetch('/api/scaling/decline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: activePendingRequestId })
    });
    pollDashboardState();
  } catch (e) { console.error('Failed to decline scaling:', e); }
}

async function resetCapacity() {
  try {
    await fetch('/api/scaling/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    pollDashboardState();
  } catch (e) { console.error('Failed to reset capacity:', e); }
}

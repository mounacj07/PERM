/**
 * PERM Web Dashboard — Ops Console Engine (dashboard.js)
 * Clean, unified, single-script dashboard client without CDN dependencies.
 * Handles:
 *  - Custom Chart.js annotation plugin (bands, reference lines, vertical markers)
 *  - Rolling 90s time window charts (maintainAspectRatio: false, no animation)
 *  - Service health poller & click-to-view start command modal
 *  - Stale / empty state chart overlays
 *  - Scenario launcher & controller controls
 */

// ─── CHART.JS CUSTOM PLUGIN FOR BANDS & VERTICAL MARKERS ───
const permAnnotationsPlugin = {
  id: 'permAnnotations',
  beforeDraw(chart, args, options) {
    if (!options) return;
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales || !scales.y) return;

    ctx.save();

    // 1. Horizontal Shaded Bands (e.g. 70%-85% warning band, 85%-100% crit band)
    if (Array.isArray(options.bands)) {
      options.bands.forEach(band => {
        const yTopVal = band.yMax !== undefined ? band.yMax : scales.y.max;
        const yBotVal = band.yMin !== undefined ? band.yMin : scales.y.min;
        const yTop = scales.y.getPixelForValue(yTopVal);
        const yBot = scales.y.getPixelForValue(yBotVal);
        const height = yBot - yTop;
        if (height > 0) {
          ctx.fillStyle = band.color || 'rgba(255,255,255,0.04)';
          ctx.fillRect(chartArea.left, yTop, chartArea.width, height);
        }
      });
    }

    // 2. Horizontal Reference Lines (e.g. SLO 300ms, 70% threshold)
    if (Array.isArray(options.lines)) {
      options.lines.forEach(line => {
        if (line.y === undefined) return;
        const yPix = scales.y.getPixelForValue(line.y);
        if (yPix >= chartArea.top && yPix <= chartArea.bottom) {
          ctx.beginPath();
          ctx.strokeStyle = line.color || '#ef4444';
          ctx.lineWidth = line.width || 1;
          if (line.dash) {
            ctx.setLineDash(line.dash);
          } else {
            ctx.setLineDash([]);
          }
          ctx.moveTo(chartArea.left, yPix);
          ctx.lineTo(chartArea.right, yPix);
          ctx.stroke();

          if (line.label) {
            ctx.font = '10px system-ui, sans-serif';
            ctx.fillStyle = line.color || '#ef4444';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.fillText(line.label, chartArea.right - 4, yPix - 2);
          }
        }
      });
    }

    // 3. Vertical Markers (Burst start, scaling actions)
    if (Array.isArray(options.verticalMarkers)) {
      options.verticalMarkers.forEach(vm => {
        if (!vm.timestamp || !scales.x) return;
        // Find x coordinate from timestamp label
        const labels = chart.data.labels || [];
        const idx = labels.indexOf(vm.timestamp);
        if (idx !== -1) {
          const xPix = scales.x.getPixelForValue(idx);
          if (xPix >= chartArea.left && xPix <= chartArea.right) {
            ctx.beginPath();
            ctx.strokeStyle = vm.color || '#38bdf8';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.moveTo(xPix, chartArea.top);
            ctx.lineTo(xPix, chartArea.bottom);
            ctx.stroke();

            if (vm.label) {
              ctx.font = '9px system-ui, sans-serif';
              ctx.fillStyle = vm.color || '#38bdf8';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText(vm.label, xPix, chartArea.top + 2);
            }
          }
        }
      });
    }

    ctx.restore();
  }
};

// Register plugin globally if Chart is available
if (typeof Chart !== 'undefined') {
  Chart.register(permAnnotationsPlugin);
}

// ─── GLOBAL PERM DASHBOARD CONTROLLER ───
window.PERM = {
  isPollingState: false,
  isPollingHealth: false,
  latestState: null,
  latestHealth: null,
  activeRunStartTime: null,

  // Formatters
  formatTime(isoOrSec) {
    if (!isoOrSec) return '--:--:--';
    if (typeof isoOrSec === 'number') {
      const d = new Date(isoOrSec * 1000);
      return d.toTimeString().split(' ')[0];
    }
    try {
      const d = new Date(isoOrSec);
      if (isNaN(d.getTime())) return String(isoOrSec).substring(11, 19) || isoOrSec;
      return d.toTimeString().split(' ')[0];
    } catch (e) {
      return String(isoOrSec);
    }
  },

  formatRelativeTime(ts) {
    if (!ts) return '';
    try {
      const t = typeof ts === 'number' ? (ts > 1e11 ? ts : ts * 1000) : new Date(ts).getTime();
      const diffSec = Math.round((Date.now() - t) / 1000);
      if (diffSec < 2) return 'just now';
      if (diffSec < 60) return `${diffSec}s ago`;
      const diffMin = Math.floor(diffSec / 60);
      if (diffMin < 60) return `${diffMin}m ago`;
      return `${Math.floor(diffMin / 60)}h ago`;
    } catch (e) {
      return '';
    }
  },

  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  },

  computeStats(arr) {
    if (!arr || arr.length === 0) return { min: 0, mean: 0, max: 0 };
    const valid = arr.filter(n => typeof n === 'number' && !isNaN(n));
    if (valid.length === 0) return { min: 0, mean: 0, max: 0 };
    let min = valid[0];
    let max = valid[0];
    let sum = 0;
    for (let i = 0; i < valid.length; i++) {
      if (valid[i] < min) min = valid[i];
      if (valid[i] > max) max = valid[i];
      sum += valid[i];
    }
    return { min, mean: sum / valid.length, max };
  },

  // Common Chart Configuration Factory
  createStandardChartOptions(customOptions = {}) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      layout: { padding: { left: 0, right: 8, top: 4, bottom: 0 } },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)', drawBorder: false },
          ticks: {
            color: '#71717a',
            font: { family: 'system-ui, sans-serif', size: 10 },
            maxTicksLimit: 8,
            maxRotation: 0,
            autoSkip: true
          }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
          ticks: {
            color: '#71717a',
            font: { family: 'system-ui, sans-serif', size: 10 }
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          align: 'end',
          labels: {
            color: '#a1a1aa',
            font: { family: 'system-ui, sans-serif', size: 11, weight: '500' },
            boxWidth: 10,
            boxHeight: 10,
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        tooltip: {
          backgroundColor: '#1b1b20',
          titleColor: '#f4f4f5',
          bodyColor: '#a1a1aa',
          borderColor: '#27272a',
          borderWidth: 1,
          padding: 8,
          cornerRadius: 6,
          titleFont: { family: 'system-ui, sans-serif', size: 11, weight: '600' },
          bodyFont: { family: 'ui-monospace, monospace', size: 11 }
        }
      },
      ...customOptions
    };
  },

  // ─── TOPBAR SERVICE HEALTH MODAL ───
  showServiceCommand(serviceKey) {
    if (!this.latestHealth) return;
    const item = this.latestHealth[serviceKey];
    if (!item) return;

    const modal = document.getElementById('serviceModal');
    const title = document.getElementById('modalServiceTitle');
    const status = document.getElementById('modalServiceStatus');
    const lat = document.getElementById('modalServiceLatency');
    const cmd = document.getElementById('modalServiceCmd');

    if (!modal) return;

    const nameMap = {
      collector: 'Telemetry Collector (Port 5000)',
      inference: 'ML Inference Service (Port 5001)',
      alert: 'Alert Engine (Port 5002)',
      remediation: 'Remediation Service (Port 5003)',
      controller: 'PERM Demo Controller'
    };

    if (title) title.innerText = nameMap[serviceKey] || serviceKey.toUpperCase();
    if (status) {
      status.innerText = item.status ? item.status.toUpperCase() : 'UNKNOWN';
      status.className = item.status === 'online' ? 'status-badge status-badge-ok' : 'status-badge status-badge-crit';
    }
    if (lat) {
      lat.innerText = item.latency_ms !== null && item.latency_ms !== undefined ? `${item.latency_ms.toFixed(1)} ms` : '—';
    }
    if (cmd) {
      cmd.innerText = item.start_cmd || '# No command provided';
    }

    modal.style.display = 'flex';
  },

  closeServiceModal() {
    const modal = document.getElementById('serviceModal');
    if (modal) modal.style.display = 'none';
  },

  copyModalCmd() {
    const cmd = document.getElementById('modalServiceCmd');
    if (cmd && navigator.clipboard) {
      navigator.clipboard.writeText(cmd.innerText).then(() => {
        const btn = document.getElementById('modalCopyBtn');
        if (btn) {
          const old = btn.innerText;
          btn.innerText = 'Copied!';
          setTimeout(() => { btn.innerText = old; }, 1500);
        }
      });
    }
  },

  // ─── HEALTH POLLING ───
  async pollHealth() {
    try {
      const resp = await fetch('/api/dashboard/health');
      if (!resp.ok) return;
      const data = await resp.json();
      this.latestHealth = data;
      this.renderServicePills(data);
    } catch (e) {
      console.debug('Health poll error:', e);
    }
  },

  renderServicePills(health) {
    const services = ['collector', 'inference', 'alert', 'remediation', 'controller'];
    services.forEach(key => {
      const pill = document.getElementById(`pill-${key}`);
      if (!pill) return;
      const info = health[key];
      if (!info) return;

      const isOnline = info.status === 'online';
      pill.className = `service-pill ${isOnline ? 'online' : 'offline'}`;
      pill.title = `${key.toUpperCase()}: ${info.status} (${info.latency_ms !== null ? info.latency_ms + 'ms' : 'unreachable'})\nClick to view start command`;
    });
  },

  // ─── TOPBAR ACTIVE SCENARIO CHIP ───
  updateScenarioChip(launcherStatus) {
    const chip = document.getElementById('topbarScenarioChip');
    const label = document.getElementById('topbarScenarioLabel');
    const bar = document.getElementById('topbarScenarioProgress');
    if (!chip || !label) return;

    if (!launcherStatus || launcherStatus.status === 'idle') {
      label.innerText = 'No Scenario Running';
      chip.className = 'scenario-chip idle';
      if (bar) bar.style.width = '0%';
      return;
    }

    if (launcherStatus.status === 'running') {
      const profile = (launcherStatus.profile || 'run').toUpperCase();
      const elapsed = (launcherStatus.elapsed_sec || 0).toFixed(0);
      const target = (launcherStatus.duration_sec || 65).toFixed(0);
      const isBurst = launcherStatus.is_burst ? 'BURST' : 'CALM';
      label.innerHTML = `<strong>${profile}</strong> [${isBurst}] ${elapsed}s / ${target}s`;
      chip.className = 'scenario-chip running';

      if (bar) {
        const pct = Math.min(100, Math.max(0, ((launcherStatus.elapsed_sec || 0) / (launcherStatus.duration_sec || 65)) * 100));
        bar.style.width = `${pct}%`;
      }
      return;
    }

    if (launcherStatus.status === 'finished') {
      label.innerText = `Finished (${launcherStatus.profile || ''})`;
      chip.className = 'scenario-chip finished';
      if (bar) bar.style.width = '100%';
      return;
    }

    if (launcherStatus.status === 'failed') {
      label.innerText = `Failed (${launcherStatus.profile || ''})`;
      chip.className = 'scenario-chip failed';
      if (bar) bar.style.width = '100%';
      return;
    }
  },

  // ─── STALE STATE OVERLAY ───
  toggleStaleOverlays(isStale) {
    const overlays = document.querySelectorAll('.chart-empty-overlay');
    overlays.forEach(el => {
      el.style.display = isStale ? 'flex' : 'none';
    });
  },

  // ─── MAIN STATE POLLING ───
  async pollState() {
    if (this.isPollingState) return;
    this.isPollingState = true;
    try {
      const resp = await fetch('/api/dashboard/state');
      if (!resp.ok) return;
      const data = await resp.json();
      this.latestState = data;

      // Check stale status
      const history = data.history || [];
      const isStale = !!data.stale || history.length === 0;
      this.toggleStaleOverlays(isStale);

      // Check launcher status for topbar
      if (data.launcher_status) {
        this.updateScenarioChip(data.launcher_status);
      }

      // Delegate to page-specific update handler
      if (typeof window.PERM_PAGE_UPDATE === 'function') {
        window.PERM_PAGE_UPDATE(data);
      }
    } catch (e) {
      console.debug('State poll error:', e);
    } finally {
      this.isPollingState = false;
    }
  },

  init() {
    // Topbar modal close listener
    const modal = document.getElementById('serviceModal');
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) this.closeServiceModal();
      });
    }

    // Health poller (every 2.5s)
    this.pollHealth();
    setInterval(() => this.pollHealth(), 2500);

    // State poller (every 1s)
    this.pollState();
    setInterval(() => this.pollState(), 1000);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.PERM.init();
});

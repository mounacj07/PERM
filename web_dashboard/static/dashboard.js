/**
 * PERM Web Dashboard — Ops Console Engine (dashboard.js)
 * Single-script dashboard client without CDN dependencies.
 * Handles:
 *  - Custom Chart.js annotation plugin (bands, reference lines, vertical markers, future region)
 *  - Rolling time window charts (maintainAspectRatio: false, no animation)
 *  - Service health poller & click-to-view start command modal
 *  - Scaling notifications (toast, Web Audio chime, bell dropdown, desktop alerts)
 *  - Auto / Manual scaling approval controls and pending banner
 *  - Scenario launcher & controller controls
 */

// ─── CHART.JS CUSTOM PLUGIN FOR BANDS, VERTICAL MARKERS & FUTURE REGION ───
const permAnnotationsPlugin = {
  id: 'permAnnotations',
  beforeDraw(chart, args, options) {
    if (!options) return;
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales || !scales.y) return;

    ctx.save();

    // 1. Shaded Future Horizon Region (H2: subtle neutral tint, dashed now line, dynamic forecast +<h>s or Forecast offline)
    if (options.futureRegion && scales.x) {
      const fr = options.futureRegion;
      const labels = chart.data.labels || [];
      const fromIdx = fr.fromIndex !== undefined ? fr.fromIndex : (fr.fromLabel ? labels.indexOf(fr.fromLabel) : -1);
      const toIdx = fr.toIndex !== undefined ? fr.toIndex : (fr.toLabel ? labels.indexOf(fr.toLabel) : labels.length - 1);
      if (fromIdx >= 0 && toIdx >= fromIdx) {
        const xStart = Math.max(chartArea.left, scales.x.getPixelForValue(fromIdx));
        const xEnd = Math.min(chartArea.right, scales.x.getPixelForValue(toIdx));
        const w = xEnd - xStart;
        if (w > 0) {
          // Subtle neutral tint (white at ~4% alpha)
          ctx.fillStyle = fr.color || 'rgba(255, 255, 255, 0.04)';
          ctx.fillRect(xStart, chartArea.top, w, chartArea.height);

          // Thin dashed vertical "now" line
          ctx.beginPath();
          ctx.strokeStyle = fr.borderColor || 'rgba(255, 255, 255, 0.25)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.moveTo(xStart, chartArea.top);
          ctx.lineTo(xStart, chartArea.bottom);
          ctx.stroke();

          // Future Region Label
          if (fr.isOffline) {
            ctx.font = '10px system-ui, sans-serif';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Forecast offline', xStart + (w / 2), chartArea.top + (chartArea.height / 2));
          } else if (fr.label) {
            ctx.font = '9px system-ui, sans-serif';
            ctx.fillStyle = fr.textColor || 'rgba(255, 255, 255, 0.4)';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            ctx.fillText(fr.label, xEnd - 4, chartArea.top + 4);
          }
        }
      }
    }

    // 2. Horizontal Shaded Bands (e.g. 70%-85% warning band, 85%-100% crit band)
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

    // 3. Horizontal Reference Lines (e.g. SLO 300ms, 70% threshold)
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

    // 4. Vertical Scale Markers (H3: thin line + small numbered badge at top, staggered heights index % 3)
    if (Array.isArray(options.verticalMarkers)) {
      options.verticalMarkers.forEach((vm, i) => {
        if (!scales.x) return;
        const labels = chart.data.labels || [];
        let idx = vm.index !== undefined ? vm.index : (vm.timestamp ? labels.indexOf(vm.timestamp) : -1);
        if (idx !== -1) {
          const xPix = scales.x.getPixelForValue(idx);
          if (xPix >= chartArea.left && xPix <= chartArea.right) {
            // Thin vertical line
            ctx.beginPath();
            ctx.strokeStyle = vm.color || '#38bdf8';
            ctx.lineWidth = 1;
            ctx.setLineDash(vm.dash || [2, 2]);
            ctx.moveTo(xPix, chartArea.top);
            ctx.lineTo(xPix, chartArea.bottom);
            ctx.stroke();

            // Small numbered badge at top, staggered heights (index mod 3)
            const badgeIndex = vm.badgeIdx !== undefined ? vm.badgeIdx : i;
            const badgeY = chartArea.top + 8 + (badgeIndex % 3) * 12;
            const badgeRadius = 6.5;

            ctx.beginPath();
            ctx.arc(xPix, badgeY, badgeRadius, 0, 2 * Math.PI);
            ctx.fillStyle = vm.color || '#38bdf8';
            ctx.fill();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 8px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const numText = vm.badgeNum !== undefined ? String(vm.badgeNum) : String(badgeIndex + 1);
            ctx.fillText(numText, xPix, badgeY);
          }
        }
      });
    }

    ctx.restore();
  }
};

if (typeof Chart !== 'undefined') {
  Chart.register(permAnnotationsPlugin);
}

// ─── GLOBAL PERM DASHBOARD CONTROLLER ───
window.PERM = {
  isPollingState: false,
  isPollingHealth: false,
  isPollingLauncher: false,
  latestState: null,
  latestHealth: null,
  activeRunStartTime: null,
  pageLoadTime: Date.now() / 1000,
  audioCtx: null,
  soundEnabled: false,
  desktopAlertsEnabled: false,
  processedEventIds: new Set(),
  recentNotifications: [],
  unreadNotificationsCount: 0,
  activePendingProposal: null,

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

  // Chart Configuration Factory
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

  // ─── NOTIFICATION SUBSYSTEM (SOUND, TOAST, DROPDOWN, DESKTOP) ───
  playNotificationChime() {
    if (!this.soundEnabled) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!this.audioCtx) this.audioCtx = new AudioCtx();
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      const t = this.audioCtx.currentTime;
      const osc1 = this.audioCtx.createOscillator();
      const osc2 = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(587.33, t); // D5
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(880, t + 0.08); // A5

      gain.gain.setValueAtTime(0.08, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc1.start(t);
      osc1.stop(t + 0.15);
      osc2.start(t + 0.08);
      osc2.stop(t + 0.35);
    } catch (e) {
      console.debug('Audio chime error:', e);
    }
  },

  showToast(title, body, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `perm-toast toast-${type}`;
    toast.innerHTML = `
      <div class="toast-title">${this.escapeHtml(title)}</div>
      <div class="toast-body">${this.escapeHtml(body)}</div>
    `;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 300);
    }, 5500);
  },

  toggleSound() {
    const chk = document.getElementById('soundToggleCheckbox');
    this.soundEnabled = chk ? chk.checked : !this.soundEnabled;
    try {
      localStorage.setItem('perm_sound_enabled', this.soundEnabled ? 'true' : 'false');
    } catch (e) {}
    if (this.soundEnabled) {
      this.playNotificationChime();
    }
  },

  toggleNotificationsDropdown() {
    const dd = document.getElementById('notificationsDropdown');
    if (!dd) return;
    const isHidden = dd.style.display === 'none' || !dd.style.display;
    dd.style.display = isHidden ? 'flex' : 'none';
    if (isHidden) {
      this.unreadNotificationsCount = 0;
      const badge = document.getElementById('bellUnreadBadge');
      if (badge) badge.style.display = 'none';
    }
  },

  requestDesktopAlerts() {
    if (!('Notification' in window)) {
      alert('Desktop notifications are not supported in this browser.');
      return;
    }
    Notification.requestPermission().then(permission => {
      const btn = document.getElementById('btnEnableDesktopAlerts');
      if (permission === 'granted') {
        this.desktopAlertsEnabled = true;
        if (btn) btn.innerText = 'Desktop Alerts Enabled';
        this.showToast('Desktop Alerts', 'Desktop notifications enabled for scaling events.', 'ok');
      } else {
        this.desktopAlertsEnabled = false;
        if (btn) btn.innerText = 'Alerts Blocked';
      }
    });
  },

  handleScalingEvents(events) {
    if (!Array.isArray(events) || events.length === 0) return;

    // Filter to events newer than run start (or page load)
    const runStartSec = this.activeRunStartTime ? this.activeRunStartTime : this.pageLoadTime;

    events.forEach((evt, idx) => {
      const evtId = evt.id || `${evt.type}_${evt.time}_${evt.value}_${idx}`;
      if (this.processedEventIds.has(evtId)) return;
      this.processedEventIds.add(evtId);

      let evtSec = 0;
      if (evt.time) {
        evtSec = typeof evt.time === 'number' ? evt.time : new Date(evt.time).getTime() / 1000;
      }
      // Skip if older than current run's start
      if (evtSec < (runStartSec - 1.0)) return;

      const type = evt.type || 'INFO';
      const reason = evt.reason || '';

      // Parse fields from reason: rule, from, to, forecast, capacity, util
      const ruleMatch = reason.match(/rule=([^;]+)/);
      const fromMatch = reason.match(/from=(\d+)/);
      const toMatch = reason.match(/to=(\d+)/);
      const fcMatch = reason.match(/forecast=([0-9.]+)rps/);
      const capMatch = reason.match(/capacity=([0-9.]+)rps/);
      const utilMatch = reason.match(/util=([0-9.]+)%/);

      const rule = ruleMatch ? ruleMatch[1] : (type.toLowerCase());
      const fromW = fromMatch ? fromMatch[1] : '?';
      const toW = toMatch ? toMatch[1] : (evt.value !== undefined ? evt.value : '?');
      const fc = fcMatch ? `${fcMatch[1]} req/s` : null;
      const cap = capMatch ? `${capMatch[1]} req/s` : null;
      const util = utilMatch ? `${utilMatch[1]}%` : null;

      let toastType = 'info';
      let title = '';
      let body = '';

      if (type === 'SCALE_UP') {
        toastType = 'warn';
        title = `Scale Up: ${fromW} -> ${toW} workers`;
        body = `Rule: ${rule}` + (fc && cap ? ` (fc ${fc} vs cap ${cap}${util ? ', util ' + util : ''})` : ` (${reason})`);
      } else if (type === 'SCALE_DOWN') {
        toastType = 'info';
        title = `Scale Down: ${fromW} -> ${toW} workers`;
        body = `Rule: ${rule} (traffic calm, hysteresis elapsed)`;
      } else if (type === 'SCALE_PROPOSED') {
        toastType = 'warn';
        title = `Scale Proposed: ${fromW} -> ${toW} workers`;
        body = `Rule: ${rule}` + (fc && cap ? ` (fc ${fc} vs cap ${cap}${util ? ', util ' + util : ''})` : ` (${reason})`);
      } else if (type === 'SCALE_DISMISSED') {
        toastType = 'crit';
        title = `Proposal Dismissed (Target: ${toW})`;
        body = `Suppressed for 20s. Operator dismissed scaling action.`;
      } else if (type === 'PROPOSAL_EXPIRED') {
        toastType = 'info';
        title = `Proposal Expired`;
        body = `Traffic demand subsided; scaling proposal cancelled.`;
      } else {
        return; // Ignore non-scaling events for toasts
      }

      // Display Toast
      this.showToast(title, body, toastType);

      // Audio Chime
      this.playNotificationChime();

      // Desktop Notification (when tab is hidden)
      if (document.hidden && this.desktopAlertsEnabled && 'Notification' in window && Notification.permission === 'granted') {
        try {
          new Notification(title, { body, icon: '/static/favicon.ico' });
        } catch (e) {}
      }

      // Append to recent notifications list (up to 20)
      this.recentNotifications.unshift({
        id: evtId,
        title,
        body,
        type: toastType,
        time: evt.time || new Date().toISOString()
      });
      if (this.recentNotifications.length > 20) {
        this.recentNotifications.pop();
      }

      this.unreadNotificationsCount += 1;
      this.renderNotificationsDropdown();
    });
  },

  renderNotificationsDropdown() {
    const badge = document.getElementById('bellUnreadBadge');
    if (badge) {
      if (this.unreadNotificationsCount > 0) {
        badge.innerText = this.unreadNotificationsCount;
        badge.style.display = 'block';
      } else {
        badge.style.display = 'none';
      }
    }

    const list = document.getElementById('notificationsList');
    if (!list) return;

    if (this.recentNotifications.length === 0) {
      list.innerHTML = `<div class="notif-empty">No scaling events in current run.</div>`;
      return;
    }

    list.innerHTML = this.recentNotifications.map(n => {
      const timeStr = this.formatTime(n.time);
      const relStr = this.formatRelativeTime(n.time);
      return `
        <div class="notif-item">
          <div class="notif-item-header">
            <span class="notif-item-title">${this.escapeHtml(n.title)}</span>
            <span class="notif-item-time">${timeStr} (${relStr})</span>
          </div>
          <div class="notif-item-body">${this.escapeHtml(n.body)}</div>
        </div>
      `;
    }).join('');
  },

  // ─── AUTO / MANUAL SCALING CONTROLS ───
  async setApprovalMode(mode) {
    try {
      const resp = await fetch('/api/controller/approval_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      if (resp.ok) {
        this.updateApprovalModeUI(mode);
      }
    } catch (e) {
      console.debug('Failed to set approval mode:', e);
    }
  },

  async toggleAskRelease(checked) {
    try {
      await fetch('/api/controller/approval_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ask_before_release: checked })
      });
    } catch (e) {
      console.debug('Failed to toggle ask release:', e);
    }
  },

  toggleSettingsPopover() {
    const pop = document.getElementById('settingsPopover');
    if (!pop) return;
    pop.style.display = (pop.style.display === 'none' || !pop.style.display) ? 'block' : 'none';
  },

  updateApprovalModeUI(mode, askBeforeRelease = false) {
    const isManual = mode === 'manual';
    const btnAuto = document.getElementById('btnSegAuto');
    const btnManual = document.getElementById('btnSegManual');
    const chkRelease = document.getElementById('chkAskRelease');

    if (btnAuto && btnManual) {
      if (isManual) {
        btnManual.classList.add('active');
        btnAuto.classList.remove('active');
      } else {
        btnAuto.classList.add('active');
        btnManual.classList.remove('active');
      }
    }

    if (chkRelease !== null && chkRelease !== undefined) {
      chkRelease.checked = Boolean(askBeforeRelease);
    }

    const miniSel = document.getElementById('miniApprovalSelect');
    if (miniSel && miniSel.value !== mode) {
      miniSel.value = mode;
    }
  },

  async checkPendingProposal(pending) {
    const banner = document.getElementById('pendingProposalBanner');
    if (!banner) return;

    if (!pending || !pending.id) {
      banner.style.display = 'none';
      this.activePendingProposal = null;
      return;
    }

    this.activePendingProposal = pending;
    banner.style.display = 'block';

    const nowSec = Date.now() / 1000;
    const created = pending.created_at || nowSec;
    const elapsed = Math.max(0, Math.floor(nowSec - created));

    const timerBadge = document.getElementById('pendingTimerBadge');
    if (timerBadge) {
      if (elapsed >= 10) {
        timerBadge.className = 'pending-timer-badge crit';
        timerBadge.innerText = `${elapsed}s (OVERLOAD RISK)`;
      } else {
        timerBadge.className = 'pending-timer-badge';
        timerBadge.innerText = `${elapsed}s pending`;
      }
    }

    const textEl = document.getElementById('pendingBannerText');
    if (textEl) {
      const fc = pending.forecast !== undefined ? pending.forecast : '--';
      const cap = pending.capacity !== undefined ? pending.capacity : '--';
      const fromW = pending.from_workers !== undefined ? pending.from_workers : '?';
      const toW = pending.to_workers !== undefined ? pending.to_workers : '?';
      textEl.innerText = `Forecast ${fc} req/s in ~20s vs capacity ${cap} req/s. Recommend ${fromW} -> ${toW} workers.`;
    }
  },

  async approveProposal() {
    if (!this.activePendingProposal) return;
    const btn = document.getElementById('btnApproveProposal');
    if (btn) { btn.disabled = true; btn.innerText = 'Approving...'; }

    try {
      const resp = await fetch('/api/controller/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: this.activePendingProposal.id })
      });
      if (resp.ok) {
        const banner = document.getElementById('pendingProposalBanner');
        if (banner) banner.style.display = 'none';
        this.activePendingProposal = null;
      }
    } catch (e) {
      console.debug('Approve error:', e);
    } finally {
      if (btn) { btn.disabled = false; btn.innerText = '✓ Approve'; }
    }
  },

  async dismissProposal() {
    if (!this.activePendingProposal) return;
    const btn = document.getElementById('btnDismissProposal');
    if (btn) { btn.disabled = true; btn.innerText = 'Dismissing...'; }

    try {
      const resp = await fetch('/api/controller/dismiss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: this.activePendingProposal.id })
      });
      if (resp.ok) {
        const banner = document.getElementById('pendingProposalBanner');
        if (banner) banner.style.display = 'none';
        this.activePendingProposal = null;
      }
    } catch (e) {
      console.debug('Dismiss error:', e);
    } finally {
      if (btn) { btn.disabled = false; btn.innerText = '✕ Dismiss'; }
    }
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
    const modeLabel = (this.latestState && this.latestState.model_mode_label) || '';
    const isPredictive = modeLabel.toLowerCase().includes('predictive');

    const requiredServices = new Set(['collector', 'controller']);
    if (isPredictive) {
      requiredServices.add('inference');
    }

    const services = ['collector', 'inference', 'alert', 'remediation', 'controller'];
    services.forEach(key => {
      const pill = document.getElementById(`pill-${key}`);
      if (!pill) return;
      const info = health[key];
      if (!info) return;

      const isOnline = info.status === 'online';
      const isRequired = requiredServices.has(key);

      pill.classList.remove('online', 'offline', 'offline-required', 'offline-optional');

      const textSpan = pill.querySelector('span:not(.pill-dot):not(.dot)');

      if (isOnline) {
        pill.classList.add('online');
        if (textSpan) {
          textSpan.innerText = key.charAt(0).toUpperCase() + key.slice(1);
        }
      } else if (isRequired) {
        pill.classList.add('offline', 'offline-required');
        if (textSpan) {
          textSpan.innerText = key.charAt(0).toUpperCase() + key.slice(1);
        }
      } else {
        pill.classList.add('offline-optional');
        if (textSpan) {
          textSpan.innerText = `${key.charAt(0).toUpperCase() + key.slice(1)} (not started)`;
        }
      }

      pill.title = `${key.toUpperCase()}: ${info.status} (${info.latency_ms !== null && info.latency_ms !== undefined ? info.latency_ms + 'ms' : 'unreachable'})\nClick to view start command`;
    });
  },

  // ─── TOPBAR & MINI-LAUNCHER SCENARIO CHIP ───
  updateScenarioChip(launcherStatus) {
    const chip = document.getElementById('topbarScenarioChip');
    const label = document.getElementById('topbarScenarioLabel');
    const bar = document.getElementById('topbarScenarioProgress');

    const miniBadge = document.getElementById('miniStatusBadge');
    const miniPhase = document.getElementById('miniPhaseBadge');
    const miniBar = document.getElementById('miniProgressBar');
    const miniTimer = document.getElementById('miniTimerText');
    const miniBtnStart = document.getElementById('miniStartBtn');
    const miniBtnStop = document.getElementById('miniStopBtn');

    if (!launcherStatus || launcherStatus.status === 'idle') {
      if (label) label.innerText = 'No Scenario Running';
      if (chip) chip.className = 'scenario-chip idle';
      if (bar) bar.style.width = '0%';

      if (miniBadge) {
        miniBadge.className = 'status-badge status-badge-idle';
        miniBadge.innerText = 'IDLE';
      }
      if (miniPhase) miniPhase.style.display = 'none';
      if (miniBar) miniBar.style.width = '0%';
      if (miniTimer) miniTimer.innerText = '0.0s / 0s';
      if (miniBtnStart) { miniBtnStart.disabled = false; miniBtnStart.style.display = 'inline-flex'; miniBtnStart.innerHTML = '&blacktriangleright; Start Run'; }
      if (miniBtnStop) { miniBtnStop.style.display = 'none'; }
      return;
    }

    const dur = launcherStatus.duration_sec || 80;
    const elap = launcherStatus.elapsed_sec || 0;
    const profile = (launcherStatus.profile || 'run').toUpperCase();
    const isBurst = Boolean(launcherStatus.is_burst);
    const isFallback = Boolean(launcherStatus.forecast_fallback) || (this.latestState && this.latestState.forecast_fallback);

    if (launcherStatus.start_time && !this.activeRunStartTime) {
      this.activeRunStartTime = launcherStatus.start_time;
    }

    const policy = launcherStatus.controller_policy || (this.latestState && this.latestState.controller_state && this.latestState.controller_state.policy) || (launcherStatus.controller_mode === 'predictive_only' ? 'predictive_only' : '');
    const policyTag = policy ? ` · ${policy}` : '';

    if (launcherStatus.status === 'running') {
      const burstTag = isBurst ? 'BURST' : 'CALM';
      let chipText = `${profile} · ${burstTag}${policyTag} · ${elap.toFixed(0)}/${dur.toFixed(0)}s`;
      if (isFallback) {
        chipText += ' · reactive (fallback)';
      }
      if (label) label.innerText = chipText;
      if (chip) chip.className = isFallback ? 'scenario-chip running fallback' : 'scenario-chip running';

      const pct = Math.min(100, Math.max(0, (elap / dur) * 100));
      if (bar) bar.style.width = `${pct}%`;

      if (miniBadge) {
        miniBadge.className = 'status-badge status-badge-ok';
        miniBadge.innerText = 'RUNNING';
      }
      if (miniPhase) {
        miniPhase.style.display = 'inline-block';
        miniPhase.className = isBurst ? 'status-badge status-badge-warn' : 'status-badge status-badge-ok';
        miniPhase.innerText = isBurst ? 'BURST PHASE' : 'CALM PHASE';
      }
      if (miniBar) miniBar.style.width = `${pct}%`;
      if (miniTimer) miniTimer.innerText = `${elap.toFixed(1)}s / ${dur.toFixed(0)}s`;
      if (miniBtnStart) { miniBtnStart.style.display = 'none'; }
      if (miniBtnStop) { miniBtnStop.style.display = 'inline-flex'; miniBtnStop.disabled = false; }
      return;
    }

    if (launcherStatus.status === 'finished') {
      let finText = `Finished (${profile}${policyTag})`;
      if (isFallback) {
        finText += ' [reactive fallback]';
      }
      if (label) label.innerText = finText;
      if (chip) chip.className = isFallback ? 'scenario-chip finished fallback' : 'scenario-chip finished';
      if (bar) bar.style.width = '100%';

      if (miniBadge) {
        miniBadge.className = 'status-badge status-badge-ok';
        miniBadge.innerText = 'FINISHED';
      }
      if (miniPhase) miniPhase.style.display = 'none';
      if (miniBar) miniBar.style.width = '100%';
      if (miniTimer) miniTimer.innerText = `${(launcherStatus.last_elapsed || elap).toFixed(1)}s / ${dur.toFixed(0)}s`;
      if (miniBtnStart) { miniBtnStart.disabled = false; miniBtnStart.style.display = 'inline-flex'; miniBtnStart.innerHTML = '&blacktriangleright; Start Run'; }
      if (miniBtnStop) { miniBtnStop.style.display = 'none'; }
      return;
    }

    if (launcherStatus.status === 'failed') {
      if (label) label.innerText = `Failed (${profile})`;
      if (chip) chip.className = 'scenario-chip failed';
      if (bar) bar.style.width = '100%';

      if (miniBadge) {
        miniBadge.className = 'status-badge status-badge-crit';
        miniBadge.innerText = 'FAILED';
      }
      if (miniPhase) miniPhase.style.display = 'none';
      if (miniBtnStart) { miniBtnStart.disabled = false; miniBtnStart.style.display = 'inline-flex'; miniBtnStart.innerHTML = '&blacktriangleright; Start Run'; }
      if (miniBtnStop) { miniBtnStop.style.display = 'none'; }
      return;
    }
  },

  // ─── MINI LAUNCHER ACTIONS (LIVE PAGE) ───
  async miniLaunch() {
    const profSel = document.getElementById('miniProfileSelect');
    const modeSel = document.getElementById('miniCtrlModeSelect');
    const appSel = document.getElementById('miniApprovalSelect');
    const ctrlCheck = document.getElementById('miniRunWithCtrl');

    const profile = profSel ? profSel.value : 'spike';
    const mode = modeSel ? modeSel.value : 'predictive';
    const approval = appSel ? appSel.value : 'auto';
    const runCtrl = ctrlCheck ? ctrlCheck.checked : true;

    const btnStart = document.getElementById('miniStartBtn');
    if (btnStart) {
      btnStart.disabled = true;
      btnStart.innerText = 'Starting...';
    }

    this.activeRunStartTime = Date.now() / 1000;

    try {
      const resp = await fetch('/api/launcher/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile: profile,
          seed: 44,
          run_with_controller: runCtrl,
          controller_mode: mode,
          approval_mode: approval
        })
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert(`Failed to start scenario: ${data.error || 'Unknown error'}`);
        if (btnStart) {
          btnStart.disabled = false;
          btnStart.innerHTML = '&blacktriangleright; Start Run';
        }
      } else {
        // Immediate UI update
        if (data.launcher_status) {
          this.updateScenarioChip(data.launcher_status);
        }
        this.pollLauncherStatus();
      }
    } catch (e) {
      alert(`Network error: ${e.message}`);
      if (btnStart) {
        btnStart.disabled = false;
        btnStart.innerHTML = '&blacktriangleright; Start Run';
      }
    }
  },

  async miniStop() {
    const btnStop = document.getElementById('miniStopBtn');
    if (btnStop) {
      btnStop.disabled = true;
      btnStop.innerText = 'Stopping...';
    }
    try {
      await fetch('/api/launcher/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stop_controller: true })
      });
      this.pollLauncherStatus();
    } catch (e) {
      console.debug('Stop error:', e);
    }
  },

  async pollLauncherStatus() {
    try {
      const resp = await fetch('/api/launcher/status');
      if (resp.ok) {
        const data = await resp.json();
        this.updateScenarioChip(data);
      }
    } catch (e) {
      console.debug('Launcher poll error:', e);
    }
  },

  // ─── STALE STATE OVERLAY ───
  toggleStaleOverlays(isStale) {
    const overlays = document.querySelectorAll('.chart-empty-overlay, .chart-stale-overlay');
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

      // Stale overlay check
      const isStale = Boolean(data.stale);
      this.toggleStaleOverlays(isStale);

      // Launcher status check
      if (data.launcher_status) {
        this.updateScenarioChip(data.launcher_status);
      }

      // Check Scaling Events & Trigger Notifications
      if (Array.isArray(data.controller_events)) {
        this.handleScalingEvents(data.controller_events);
      }

      // Check Pending Proposal
      const pending = (data.controller_state && data.controller_state.pending_proposal) || null;
      this.checkPendingProposal(pending);

      // Sync Topbar Approval Mode
      if (data.controller_state && data.controller_state.approval_mode) {
        this.updateApprovalModeUI(data.controller_state.approval_mode, data.controller_state.ask_before_release);
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
    // Restore sound toggle state
    try {
      const savedSound = localStorage.getItem('perm_sound_enabled');
      if (savedSound === 'true') {
        this.soundEnabled = true;
        const chk = document.getElementById('soundToggleCheckbox');
        if (chk) chk.checked = true;
      }
    } catch (e) {}

    // Modal close listeners
    const modal = document.getElementById('serviceModal');
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) this.closeServiceModal();
      });
    }

    // Close notifications dropdown when clicking outside
    document.addEventListener('click', (e) => {
      const dd = document.getElementById('notificationsDropdown');
      const btn = document.getElementById('btnBellNotifications') || document.getElementById('btnNotificationsBell');
      if (dd && dd.style.display !== 'none' && !dd.contains(e.target) && btn && !btn.contains(e.target)) {
        dd.style.display = 'none';
      }

      // Close settings popover when clicking outside
      const pop = document.getElementById('settingsPopover');
      const gear = document.getElementById('btnSettingsPopover');
      if (pop && pop.style.display === 'block' && !pop.contains(e.target) && (!gear || !gear.contains(e.target))) {
        pop.style.display = 'none';
      }
    });

    // Health poller (every 2.5s)
    this.pollHealth();
    setInterval(() => this.pollHealth(), 2500);

    // Main state poller (every 1s)
    this.pollState();
    setInterval(() => this.pollState(), 1000);

    // Fast launcher status poller (every 800ms)
    setInterval(() => this.pollLauncherStatus(), 800);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.PERM.init();
});

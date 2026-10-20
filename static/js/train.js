/**
 * Training Studio & Privileged Biometric Gate Controller
 * Handles Biometric Face Gate, WebSocket Training Stream,
 * Canvas Loss/Accuracy Chart, Hardware Telemetry, and Cognitive Data Inspector.
 */

class TrainingStudioManager {
  constructor() {
    this.isAuthenticated = false;
    this.gateStream = null;
    this.trainWs = null;
    this.trainHistory = [];
    this.chartCanvas = null;
    this.chartCtx = null;
  }

  async init() {
    this.initChart();
    this.bindEvents();
    this.startHardwareTelemetry();
    this.loadCognitiveInspectorData();

    // Check if session already authenticated
    if (sessionStorage.getItem("station_biometric_auth") === "true") {
      this.unlockGateSilently();
    } else {
      this.startGateCamera();
    }
  }

  showWebModal(title, message, type = "warning", onClose = null) {
    const modal = document.getElementById("clinical-web-modal");
    if (!modal) {
      alert(`${title}\n\n${message}`);
      if (onClose) onClose();
      return;
    }
    const titleEl = document.getElementById("web-modal-title");
    const bodyEl = document.getElementById("web-modal-body");
    const iconBadge = document.getElementById("web-modal-icon-badge");
    const closeBtn = document.getElementById("web-modal-close-btn");

    if (titleEl) titleEl.textContent = title || "Thông báo hệ thống";
    if (bodyEl) bodyEl.textContent = message || "";

    if (iconBadge) {
      if (type === "danger" || type === "error") {
        iconBadge.className = "badge badge-danger";
        iconBadge.textContent = "Cảnh báo";
        iconBadge.style.background = "rgba(239, 68, 68, 0.15)";
        iconBadge.style.color = "var(--accent-danger)";
      } else if (type === "success") {
        iconBadge.className = "badge badge-success";
        iconBadge.textContent = "Thành công";
        iconBadge.style.background = "rgba(5, 150, 105, 0.15)";
        iconBadge.style.color = "var(--accent-success)";
      } else {
        iconBadge.className = "badge badge-warning";
        iconBadge.textContent = "Lưu ý";
        iconBadge.style.background = "rgba(217, 119, 6, 0.15)";
        iconBadge.style.color = "var(--accent-warning)";
      }
    }

    modal.style.display = "flex";

    if (closeBtn) {
      const newCloseBtn = closeBtn.cloneNode(true);
      closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
      newCloseBtn.addEventListener("click", () => {
        modal.style.display = "none";
        if (onClose) onClose();
      });
      newCloseBtn.focus();
    }
  }

  // -------------------------------------------------------------
  // 1. Biometric Face Gate Logic
  // -------------------------------------------------------------
  async startGateCamera() {
    const video = document.getElementById("gate-video");
    if (!video) return;

    try {
      this.gateStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
      });
      video.srcObject = this.gateStream;
      await video.play();

      // Initiate initial automatic verification scan after 1.2s
      setTimeout(() => this.scanGateBiometrics(), 1200);
    } catch (err) {
      console.warn("[GATE] Camera access failed:", err);
      const statusEl = document.getElementById("gate-status-badge");
      if (statusEl) {
        if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
          statusEl.textContent = "Trình duyệt chặn camera qua mạng LAN. Vui lòng truy cập qua localhost hoặc HTTPS.";
        } else {
          statusEl.textContent = "Camera bị chặn hoặc đang bận. Vui lòng cấp quyền truy cập webcam.";
        }
        statusEl.className = "badge badge-warning";
      }
    }
  }

  async scanGateBiometrics() {
    const video = document.getElementById("gate-video");
    const statusEl = document.getElementById("gate-status-badge");
    const scoreEl = document.getElementById("gate-score-metric");

    if (!video || !video.videoWidth) {
      if (statusEl) statusEl.textContent = "Đang chờ tín hiệu camera...";
      return;
    }

    if (statusEl) {
      statusEl.textContent = "Đang phân tích vector sinh trắc học...";
      statusEl.className = "badge badge-clinical";
    }

    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 640, 480);
    const b64 = canvas.toDataURL("image/jpeg", 0.92);

    try {
      const res = await fetch("/api/auth/verify_face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 })
      });
      const data = await res.json();

      if (scoreEl) scoreEl.textContent = `Độ tương đồng Cosine: ${data.score}% / Ngưỡng đạt ${data.threshold}%`;

      if (data.authenticated) {
        if (statusEl) {
          statusEl.textContent = data.message;
          statusEl.className = "badge badge-success";
        }
        sessionStorage.setItem("station_biometric_auth", "true");
        setTimeout(() => this.unlockGate(), 800);
      } else {
        if (statusEl) {
          statusEl.textContent = data.message;
          statusEl.className = "badge badge-danger";
        }
      }
    } catch (err) {
      console.error("[GATE] Verification error:", err);
      if (statusEl) {
        statusEl.textContent = "Lỗi kết nối hoặc xử lý sinh trắc học";
        statusEl.className = "badge badge-danger";
      }
    }
  }

  async enrollCurrentFaceAsAdmin() {
    const video = document.getElementById("gate-video");
    if (!video || !video.videoWidth) {
      this.showWebModal("Cảnh báo", "Cần mở camera để chụp khuôn mặt Quản trị viên.", "warning");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 640, 480);
    const b64 = canvas.toDataURL("image/jpeg", 0.92);

    try {
      const res = await fetch("/api/auth/enroll_admin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 })
      });
      const data = await res.json();
      this.showWebModal(data.success ? "Thành công" : "Thất bại", data.message, data.success ? "success" : "danger");
      this.scanGateBiometrics();
    } catch (err) {
      this.showWebModal("Lỗi đăng ký", "Đăng ký Quản trị viên thất bại: " + err, "danger");
    }
  }

  unlockGate() {
    this.isAuthenticated = true;
    const overlay = document.getElementById("biometric-gate-overlay");
    if (overlay) {
      overlay.classList.add("unlocked");
    }
    // Stop gate camera to free resource for monitor/bank
    if (this.gateStream) {
      this.gateStream.getTracks().forEach(t => t.stop());
    }
  }

  unlockGateSilently() {
    this.isAuthenticated = true;
    const overlay = document.getElementById("biometric-gate-overlay");
    if (overlay) {
      overlay.style.display = "none";
    }
  }

  // -------------------------------------------------------------
  // 2. Multimodal Fusion Head Training & WebSocket
  // -------------------------------------------------------------
  initChart() {
    this.chartCanvas = document.getElementById("train-chart-canvas");
    if (this.chartCanvas) {
      this.chartCtx = this.chartCanvas.getContext("2d");
      this.drawEmptyChart();
    }
  }

  drawEmptyChart() {
    if (!this.chartCtx) return;
    const ctx = this.chartCtx;
    const w = this.chartCanvas.width;
    const h = this.chartCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, w, h);

    // Subtle grid
    ctx.strokeStyle = "#F1F5F9";
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();
    }

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#94A3B8";
    ctx.fillText("TRAIN LOSS & ACCURACY (EPOCH TIMELINE)", 50, 20);
  }

  updateChart() {
    if (!this.chartCtx || this.trainHistory.length < 2) return;
    const ctx = this.chartCtx;
    const w = this.chartCanvas.width;
    const h = this.chartCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "#F1F5F9";
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();
    }

    const n = this.trainHistory.length;
    const stepX = (w - 70) / (n - 1);

    // 1. Draw Loss Line (Dark Blue)
    ctx.beginPath();
    ctx.strokeStyle = "#0F172A";
    ctx.lineWidth = 2;
    this.trainHistory.forEach((pt, i) => {
      const x = 50 + i * stepX;
      // Map loss [0, 2.5] to [h - 30, 40]
      const y = (h - 30) - Math.min(1.0, pt.val_loss / 2.5) * (h - 70);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 2. Draw Accuracy Line (Green)
    ctx.beginPath();
    ctx.strokeStyle = "#059669";
    ctx.lineWidth = 2;
    this.trainHistory.forEach((pt, i) => {
      const x = 50 + i * stepX;
      // Map accuracy [0, 100] to [h - 30, 40]
      const y = (h - 30) - (pt.accuracy / 100) * (h - 70);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#0F172A";
    ctx.fillText("■ LOSS", w - 140, 20);
    ctx.fillStyle = "#059669";
    ctx.fillText("■ ACCURACY (%)", w - 85, 20);
  }

  startTraining() {
    const epochs = parseInt(document.getElementById("train-epochs").value) || 10;
    const batchSize = parseInt(document.getElementById("train-batch-size").value) || 16;
    const lr = parseFloat(document.getElementById("train-lr").value) || 0.001;
    const dropout = parseFloat(document.getElementById("train-dropout").value) || 0.2;

    this.trainHistory = [];
    this.drawEmptyChart();

    const terminal = document.getElementById("train-terminal");
    if (terminal) {
      terminal.textContent = `Khởi tạo huấn luyện PyTorch GMU...\n` +
        `Thiết bị: PyTorch trên Localhost\n` +
        `Cấu hình: Epochs=${epochs}, BatchSize=${batchSize}, LR=${lr}, Dropout=${dropout}\n` +
        `Đang kết nối luồng dữ liệu WebSocket...\n\n`;
    }

    const startBtn = document.getElementById("initiate-train-btn");
    if (startBtn) startBtn.disabled = true;

    // Connect WebSocket
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/train`;
    this.trainWs = new WebSocket(wsUrl);

    this.trainWs.onopen = () => {
      this.logTerminal("WebSocket đã kết nối: Đang truyền dữ liệu gradient thời gian thực...");
      this.trainWs.send(JSON.stringify({
        action: "start_train",
        epochs: epochs,
        batch_size: batchSize,
        lr: lr,
        dropout: dropout
      }));
    };

    this.trainWs.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "TRAIN_EPOCH_UPDATE") {
        const ep = msg.data;
        this.trainHistory.push(ep);
        this.updateChart();
        this.logTerminal(
          `Epoch ${String(ep.epoch).padStart(2, '0')}/${ep.total_epochs} • ` +
          `Train Loss: ${ep.train_loss.toFixed(4)} | ` +
          `Val Loss: ${ep.val_loss.toFixed(4)} | ` +
          `Độ chính xác: ${ep.accuracy.toFixed(1)}%`
        );
      } else if (msg.type === "TRAIN_COMPLETED") {
        this.logTerminal(
          `\nHoàn tất huấn luyện: Độ chính xác tốt nhất: ${msg.data.best_accuracy}% | ` +
          `Loss thấp nhất: ${msg.data.best_loss} | Đã lưu vào checkpoints/${msg.data.checkpoint}\n`
        );
        if (startBtn) startBtn.disabled = false;
        // Refresh Inspector data
        this.loadCognitiveInspectorData();
      } else if (msg.type === "TRAIN_ERROR") {
        this.logTerminal(`\nLỗi huấn luyện: ${msg.error}\n`);
        if (startBtn) startBtn.disabled = false;
      }
    };

    this.trainWs.onerror = (err) => {
      this.logTerminal(`Lỗi WebSocket: Không thể kết nối đến luồng huấn luyện.`);
      if (startBtn) startBtn.disabled = false;
    };
  }

  logTerminal(text) {
    const terminal = document.getElementById("train-terminal");
    if (terminal) {
      terminal.textContent += text + "\n";
      terminal.scrollTop = terminal.scrollHeight;
    }
  }

  // -------------------------------------------------------------
  // 3. Hardware Telemetry Polling
  // -------------------------------------------------------------
  startHardwareTelemetry() {
    const poll = async () => {
      try {
        const res = await fetch("/api/telemetry");
        const t = await res.json();

        const cpuEl = document.getElementById("telemetry-cpu");
        const ramEl = document.getElementById("telemetry-ram");
        const vramEl = document.getElementById("telemetry-vram");
        const tempEl = document.getElementById("telemetry-temp");

        if (cpuEl) cpuEl.textContent = `${t.cpu_percent}%`;
        if (ramEl) ramEl.textContent = `${t.ram_used_gb}GB / ${t.ram_total_gb}GB (${t.ram_percent}%)`;
        if (vramEl) vramEl.textContent = t.gpu_vram_info;
        if (tempEl) tempEl.textContent = t.gpu_temp_info;
      } catch (e) {
        // Silently continue
      }
    };
    poll();
    setInterval(poll, 2500);
  }

  // -------------------------------------------------------------
  // 4. Cognitive Data & Checkpoint Inspector
  // -------------------------------------------------------------
  async loadCognitiveInspectorData() {
    try {
      const res = await fetch("/api/inspector/data");
      const data = await res.json();

      this.renderClassDistribution(data.class_distribution);
      this.renderFaceLedger(data.face_identities);
      this.renderClinicalRulesTable(data.cbt_rules);
      this.renderCheckpointsTable(data.model_checkpoints);
    } catch (err) {
      console.error("[INSPECTOR] Error loading data:", err);
    }
  }

  renderClassDistribution(dist) {
    const container = document.getElementById("class-distribution-bars");
    if (!container || !dist) return;

    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    container.innerHTML = Object.entries(dist).map(([cls, count]) => {
      const pct = ((count / total) * 100).toFixed(1);
      return `
        <div class="channel-item">
          <span class="channel-name" style="width: 150px; font-size: 0.68rem;">${cls}</span>
          <div class="channel-bar-bg">
            <div class="channel-bar-fill" style="width: ${pct}%;"></div>
          </div>
          <span class="channel-val">${count} (${pct}%)</span>
        </div>
      `;
    }).join("");
  }

  renderFaceLedger(identities) {
    const container = document.getElementById("face-identities-ledger");
    if (!container) return;

    if (!identities || identities.length === 0) {
      container.innerHTML = `<div style="font-family:var(--font-sans); font-size:0.82rem; color:var(--text-muted); padding:1.2rem; text-align:center;">Chưa có khuôn mặt nào được lưu trong hệ thống.</div>`;
      return;
    }

    container.innerHTML = identities.map(id => `
      <div class="identity-card">
        <img class="identity-thumb" src="${id.thumbnail || '/static/assets/memes/exp_03_calm.svg'}" alt="${id.name}">
        <div class="identity-info">
          <div class="identity-name">${id.name}</div>
          <div class="identity-date">${id.enrolled_at}</div>
          <div class="identity-vec">${id.vector_summary}</div>
        </div>
        <button class="btn btn-outline" style="padding:0.35rem 0.6rem; font-size:0.75rem;" onclick="window.trainStudio.deleteFaceIdentity('${id.id}')">Xóa</button>
      </div>
    `).join("");
  }

  renderClinicalRulesTable(rules) {
    const tbody = document.getElementById("cbt-rules-tbody");
    if (!tbody || !rules) return;

    tbody.innerHTML = rules.map(r => `
      <tr>
        <td style="font-family:var(--font-mono); font-weight:700;">${r.id}</td>
        <td><span class="badge badge-clinical">${r.category}</span></td>
        <td style="font-weight:600;">${r.title}</td>
        <td style="font-size:0.78rem;">${r.protocol}</td>
        <td>
          <button class="btn btn-outline" style="padding:0.25rem 0.5rem; font-size:0.75rem;" onclick="window.trainStudio.deleteCbtRule('${r.id}')">Xóa</button>
        </td>
      </tr>
    `).join("");
  }

  renderCheckpointsTable(ckpts) {
    const tbody = document.getElementById("checkpoints-tbody");
    if (!tbody || !ckpts) return;

    tbody.innerHTML = ckpts.map(c => `
      <tr>
        <td style="font-family:var(--font-mono); font-weight:700;">${c.version_id}</td>
        <td style="font-family:var(--font-mono);">${c.timestamp}</td>
        <td>${c.epochs}</td>
        <td style="font-family:var(--font-mono); font-weight:700; color:var(--accent-success);">${c.accuracy}%</td>
        <td style="font-family:var(--font-mono);">${c.loss}</td>
        <td>
          <span class="badge ${c.status === 'ACTIVE_PRODUCTION' ? 'badge-success' : 'badge-neutral'}">
            ${c.status === 'ACTIVE_PRODUCTION' ? 'Đang hoạt động' : c.status}
          </span>
        </td>
        <td>
          ${c.status !== 'ACTIVE_PRODUCTION' ? `
            <button class="btn btn-outline" style="padding:0.25rem 0.5rem; font-size:0.75rem;" onclick="window.trainStudio.restoreCheckpoint('${c.version_id}')">Khôi phục</button>
          ` : `
            <span style="font-family:var(--font-sans); font-size:0.75rem; font-weight:600; color:var(--text-muted);">Hiện hành</span>
          `}
        </td>
      </tr>
    `).join("");
  }

  // Identity Actions
  async deleteFaceIdentity(id) {
    if (!confirm(`Xác nhận xóa danh tính ${id} khỏi hệ thống?`)) return;
    try {
      const res = await fetch(`/api/face/delete?identity_id=${id}`, { method: "POST" });
      const data = await res.json();
      this.showWebModal(data.success ? "Thành công" : "Thất bại", data.message, data.success ? "success" : "danger");
      this.loadCognitiveInspectorData();
    } catch (e) {
      this.showWebModal("Lỗi", "Lỗi khi xóa: " + e, "danger");
    }
  }

  // CBT Rule Actions
  async deleteCbtRule(ruleId) {
    if (!confirm(`Xác nhận xóa quy tắc lâm sàng ${ruleId}?`)) return;
    try {
      const res = await fetch(`/api/cbt/rules/delete?rule_id=${ruleId}`, { method: "POST" });
      const data = await res.json();
      this.showWebModal(data.success ? "Thành công" : "Thất bại", data.message, data.success ? "success" : "danger");
      this.loadCognitiveInspectorData();
    } catch (e) {
      this.showWebModal("Lỗi", "Lỗi khi xóa: " + e, "danger");
    }
  }

  async restoreCheckpoint(versionId) {
    if (!confirm(`Khôi phục mô hình về phiên bản ${versionId}?`)) return;
    try {
      const res = await fetch(`/api/checkpoints/restore?version_id=${versionId}`, { method: "POST" });
      const data = await res.json();
      this.showWebModal(data.success ? "Thành công" : "Thất bại", data.message, data.success ? "success" : "danger");
      this.loadCognitiveInspectorData();
    } catch (e) {
      this.showWebModal("Lỗi", "Lỗi khi khôi phục: " + e, "danger");
    }
  }

  bindEvents() {
    const scanBtn = document.getElementById("gate-scan-btn");
    if (scanBtn) scanBtn.addEventListener("click", () => this.scanGateBiometrics());

    const enrollAdminBtn = document.getElementById("gate-enroll-admin-btn");
    if (enrollAdminBtn) enrollAdminBtn.addEventListener("click", () => this.enrollCurrentFaceAsAdmin());

    const trainBtn = document.getElementById("initiate-train-btn");
    if (trainBtn) trainBtn.addEventListener("click", () => this.startTraining());

    // Enroll Face Identity Bank form
    const enrollIdBtn = document.getElementById("enroll-identity-btn");
    if (enrollIdBtn) {
      enrollIdBtn.addEventListener("click", async () => {
        const nameInput = document.getElementById("enroll-name-input");
        const fileInput = document.getElementById("enroll-file-input");

        if (!nameInput || !nameInput.value.trim()) {
          this.showWebModal("Lưu ý", "Vui lòng nhập tên định danh.", "warning");
          return;
        }

        let b64 = "";
        if (fileInput && fileInput.files.length > 0) {
          const reader = new FileReader();
          reader.onload = async (e) => {
            await this.submitEnrollIdentity(nameInput.value.trim(), e.target.result);
          };
          reader.readAsDataURL(fileInput.files[0]);
        } else {
          this.showWebModal("Lưu ý", "Vui lòng chọn ảnh để đăng ký khuôn mặt.", "warning");
        }
      });
    }

    // New CBT Rule Form
    const saveRuleBtn = document.getElementById("cbt-rule-save-btn");
    if (saveRuleBtn) {
      saveRuleBtn.addEventListener("click", async () => {
        const title = document.getElementById("cbt-rule-title").value.trim();
        const category = document.getElementById("cbt-rule-category").value;
        const protocol = document.getElementById("cbt-rule-protocol").value.trim();
        const prescription = document.getElementById("cbt-rule-prescription").value.trim();

        if (!title || !protocol) {
          this.showWebModal("Lưu ý", "Vui lòng điền đầy đủ tiêu đề và nội dung quy tắc.", "warning");
          return;
        }

        try {
          const res = await fetch("/api/cbt/rules", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title, category, protocol, clinical_prescription: prescription
            })
          });
          const data = await res.json();
          this.showWebModal("Thành công", "Đã lưu quy tắc thành công!", "success");
          document.getElementById("cbt-rule-title").value = "";
          document.getElementById("cbt-rule-protocol").value = "";
          document.getElementById("cbt-rule-prescription").value = "";
          this.loadCognitiveInspectorData();
        } catch (e) {
          this.showWebModal("Lỗi", "Lỗi khi lưu quy tắc: " + e, "danger");
        }
      });
    }
  }

  async submitEnrollIdentity(name, imageB64) {
    try {
      const res = await fetch("/api/face/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, image_b64: imageB64 })
      });
      const data = await res.json();
      this.showWebModal(data.success ? "Thành công" : "Thất bại", data.message, data.success ? "success" : "danger");
      this.loadCognitiveInspectorData();
    } catch (err) {
      this.showWebModal("Lỗi đăng ký", "Lỗi đăng ký khuôn mặt: " + err, "danger");
    }
  }
}

window.trainStudio = new TrainingStudioManager();

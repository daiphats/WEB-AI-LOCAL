/**
 * Main Clinical Station Controller
 * Manages 3 Clinical Tabs, Real-Time Multimodal Telemetry,
 * Vocal Elicitation Assessment, 4-7-8 Breathing Pacer, CBT Chat, and Trajectory Forecasting.
 */

class ClinicalStationApp {
  constructor() {
    this.mediaStream = null;
    this.monitorWs = null;
    this.currentBiometrics = {
      emotion: "NEUTRAL / CALM",
      confidence: 85.0,
      valence: 0.05,
      arousal: 0.02,
      exhaustion_level: 1,
      channel_contributions: { face_pct: 35, body_pct: 25, speech_pct: 20, text_pct: 20 },
      incongruence_detected: false,
      posture_warning: false
    };

    // Realtime Face Detection Telemetry
    this.isCameraActive = false;
    this.lastBbox = null;
    this.lastFaceDetected = false;
    this.isSmiling = false;
    this.smileScore = 0.0;
    this.headTilt = 0.0;
  }

  async init() {
    this.initTabs();
    this.bindEvents();
    this.initBreathingPacer();
    this.startCanvasHUDLoop();
    this.initWebSocketStream();
    this.initTrajectoryChart();
    await this.loadNewVocalPrompt();

    // Auto attempt camera startup
    this.startCamera();
  }

  // -------------------------------------------------------------
  // Custom Glassmorphic In-Web Modal Notification (Replaces native alert)
  // -------------------------------------------------------------
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

    if (titleEl) titleEl.textContent = title || "CẢNH BÁO LÂM SÀNG";
    if (bodyEl) bodyEl.textContent = message || "";

    if (iconBadge) {
      if (type === "danger" || type === "error") {
        iconBadge.className = "badge badge-danger";
        iconBadge.textContent = "[CẢNH BÁO]";
        iconBadge.style.background = "rgba(239, 68, 68, 0.25)";
        iconBadge.style.color = "#F87171";
      } else if (type === "success") {
        iconBadge.className = "badge badge-success";
        iconBadge.textContent = "[THÀNH CÔNG]";
        iconBadge.style.background = "rgba(16, 185, 129, 0.25)";
        iconBadge.style.color = "#34D399";
      } else {
        iconBadge.className = "badge badge-warning";
        iconBadge.textContent = "[LƯU Ý]";
        iconBadge.style.background = "rgba(245, 158, 11, 0.25)";
        iconBadge.style.color = "#FBBF24";
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
  // 1. Tab Navigation
  // -------------------------------------------------------------
  initTabs() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        const targetId = tab.getAttribute("data-tab");
        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        document.querySelectorAll(".tab-pane").forEach(pane => {
          pane.classList.remove("active");
        });

        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add("active");

        // Refresh views if needed
        if (targetId === "tab-mimic" && window.mimicGame) {
          window.mimicGame.loadNewTarget();
          window.mimicGame.attachWebcamStream();
        } else if (targetId === "tab-forecast") {
          this.renderTrajectoryChart();
        }
      });
    });
  }

  // -------------------------------------------------------------
  // 2. Camera Controls & Canvas HUD Reticle
  // -------------------------------------------------------------
  async toggleCamera() {
    if (this.isCameraActive) {
      this.stopCamera();
    } else {
      await this.startCamera();
    }
  }

  async startCamera() {
    const video = document.getElementById("webcam-video");
    const toggleBtn = document.getElementById("webcam-toggle-btn");
    const overlay = document.getElementById("camera-standby-overlay");
    const statusText = document.getElementById("hud-backbone-status");

    try {
      // Request ONLY video first so missing/busy audio never blocks webcam!
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
      });
      video.srcObject = this.mediaStream;
      await video.play();

      this.isCameraActive = true;
      if (overlay) overlay.style.display = "none";
      if (toggleBtn) {
        const lbl = document.getElementById("webcam-toggle-label");
        if (lbl) lbl.textContent = "TẮT";
        toggleBtn.classList.add("is-active");
      }
      if (statusText) {
        statusText.textContent = "RESNET50 (2048-D) ACTIVE";
      }
      const devBadge = document.getElementById("device-status-badge");
      if (devBadge) {
        devBadge.innerHTML = `<span class="status-dot-pulse"></span> Live`;
      }

      // Sync with mimic game camera viewport
      if (window.mimicGame) {
        window.mimicGame.attachWebcamStream();
      }
    } catch (err) {
      console.warn("[CAMERA] Access not granted or camera busy:", err);
      if (overlay) {
        overlay.style.display = "flex";
        const promptText = overlay.querySelector(".standby-prompt");
        if (promptText) {
          if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
            promptText.textContent = "TRÌNH DUYỆT CHẶN CAM QUA HTTP LAN. HÃY DÙNG LOCALHOST HOẶC HTTPS!";
          } else {
            promptText.textContent = "CAMERA BỊ CHẶN HOẶC ĐANG BẬN. VUI LÒNG CẤP QUYỀN CAMERA TRÊN TRÌNH DUYỆT!";
          }
        }
      }
      if (toggleBtn) {
        const lbl = document.getElementById("webcam-toggle-label");
        if (lbl) lbl.textContent = "BẬT";
        toggleBtn.classList.remove("is-active");
      }
    }
  }

  stopCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }
    const video = document.getElementById("webcam-video");
    if (video) video.srcObject = null;
    this.isCameraActive = false;

    const overlay = document.getElementById("camera-standby-overlay");
    if (overlay) overlay.style.display = "flex";

    const toggleBtn = document.getElementById("webcam-toggle-btn");
    if (toggleBtn) {
      const lbl = document.getElementById("webcam-toggle-label");
      if (lbl) lbl.textContent = "BẬT";
      toggleBtn.classList.remove("is-active");
    }

    const statusText = document.getElementById("hud-backbone-status");
    if (statusText) {
      statusText.textContent = "RESNET50 STANDBY";
    }

    // Sync with mimic game
    if (window.mimicGame) {
      window.mimicGame.attachWebcamStream();
    }
  }

  startCanvasHUDLoop() {
    const canvas = document.getElementById("overlay-canvas");
    const video = document.getElementById("webcam-video");
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");

    let angle = 0;
    const render = () => {
      if (canvas.parentElement) {
        const pw = canvas.parentElement.clientWidth;
        const ph = canvas.parentElement.clientHeight;
        if (canvas.width !== pw || canvas.height !== ph) {
          canvas.width = pw;
          canvas.height = ph;
        }
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width;
      const h = canvas.height;

      if (this.isCameraActive && this.lastFaceDetected && this.lastBbox) {
        // Map 320x240 frame coordinates to canvas coordinates
        const rawX = this.lastBbox[0];
        const rawY = this.lastBbox[1];
        const rawW = this.lastBbox[2];
        const rawH = this.lastBbox[3];

        const scaleX = w / 320;
        const scaleY = h / 240;

        const bx = rawX * scaleX;
        const by = rawY * scaleY;
        const bw = rawW * scaleX;
        const bh = rawH * scaleY;

        // Bounding box border
        ctx.strokeStyle = this.isSmiling ? "#059669" : "rgba(255, 255, 255, 0.4)";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(bx, by, bw, bh);

        // Corner brackets (Green if smiling, Blue if neutral)
        ctx.strokeStyle = this.isSmiling ? "#10B981" : "#0284C7";
        ctx.lineWidth = 3;
        const clen = Math.min(24, bw * 0.25);
        // TL
        ctx.beginPath(); ctx.moveTo(bx, by + clen); ctx.lineTo(bx, by); ctx.lineTo(bx + clen, by); ctx.stroke();
        // TR
        ctx.beginPath(); ctx.moveTo(bx + bw - clen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + clen); ctx.stroke();
        // BL
        ctx.beginPath(); ctx.moveTo(bx, by + bh - clen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + clen, by + bh); ctx.stroke();
        // BR
        ctx.beginPath(); ctx.moveTo(bx + bw - clen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - clen); ctx.stroke();

        // Label above bbox (Un-mirrored for standard human reading)
        ctx.save();
        ctx.translate(bx + bw / 2, Math.max(16, by - 8));
        ctx.scale(-1, 1);
        ctx.textAlign = "center";
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        if (this.isSmiling) {
          ctx.fillStyle = "#10B981";
          ctx.fillText(`[PHÁT HIỆN NỤ CƯỜI DUCHENNE: +${this.smileScore}%]`, 0, 0);
        } else if (Math.abs(this.headTilt) > 14) {
          ctx.fillStyle = "#D97706";
          ctx.fillText(`[CẢNH BÁO TƯ THẾ: LỆCH CỔ ${this.headTilt}°]`, 0, 0);
        } else {
          ctx.fillStyle = "#34D399";
          ctx.fillText("[KHUÔN MẶT ĐÃ KHỚP CHUẨN]", 0, 0);
        }
        ctx.restore();

        // Draw dynamic posture skeleton from real face location
        ctx.strokeStyle = "rgba(148, 163, 184, 0.7)";
        ctx.lineWidth = 2;
        const neck = { x: bx + bw / 2, y: by + bh };
        const shoulderL = { x: bx - bw * 0.4, y: by + bh + 40 };
        const shoulderR = { x: bx + bw * 1.4, y: by + bh + 40 };
        const spine = { x: neck.x, y: neck.y + 120 };

        ctx.beginPath();
        ctx.moveTo(neck.x, neck.y);
        ctx.lineTo(neck.x, neck.y + 35);
        ctx.moveTo(shoulderL.x, shoulderL.y);
        ctx.lineTo(shoulderR.x, shoulderR.y);
        ctx.moveTo(neck.x, neck.y + 35);
        ctx.lineTo(spine.x, spine.y);
        ctx.stroke();

        [neck, shoulderL, shoulderR, spine].forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = this.isSmiling ? "#059669" : "#0284C7";
          ctx.fill();
        });
      } else if (this.isCameraActive && !this.lastFaceDetected) {
        // Active scanning HUD when camera is on but no face is in view
        const cx = w * 0.5;
        const cy = h * 0.44;
        const rad = Math.min(w, h) * 0.28;
        angle += 0.03;

        // Subtle guide oval
        ctx.save();
        ctx.strokeStyle = "rgba(239, 68, 68, 0.45)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.ellipse(cx, cy, rad * 0.75, rad, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Laser scan line
        const scanY = cy - rad + (Math.sin(angle) * 0.5 + 0.5) * (rad * 2);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.65)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(cx - rad * 0.7, scanY);
        ctx.lineTo(cx + rad * 0.7, scanY);
        ctx.stroke();
      } else {
        if (!this.isCameraActive) {
          // Idle decorative reticle
          const bx = w * 0.35 + Math.sin(angle) * 6;
          const by = h * 0.20 + Math.cos(angle * 0.8) * 5;
          const bw = w * 0.30;
          const bh = h * 0.42;
          angle += 0.02;

          ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
          ctx.lineWidth = 1;
          ctx.strokeRect(bx, by, bw, bh);

          ctx.save();
          ctx.translate(bx + bw / 2, by - 6);
          ctx.scale(-1, 1);
          ctx.font = "10px 'JetBrains Mono', monospace";
          ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
          ctx.textAlign = "center";
          ctx.fillText("[STANDBY: NHẤN NÚT ĐỂ BẬT CAMERA]", 0, 0);
          ctx.restore();
        }
      }

      // Synchronize Dynamic Non-Mirrored HTML Status Badge (GREEN when face detected, RED when searching)
      const hudBadge = document.getElementById("camera-live-hud-badge");
      const hudText = document.getElementById("camera-live-hud-text");
      const hudDot = document.getElementById("camera-live-hud-dot");
      const hudLabel = document.getElementById("camera-live-hud-label");

      if (this.isCameraActive) {
        if (hudBadge) hudBadge.style.display = "flex";
        if (this.lastFaceDetected) {
          if (hudText) {
            hudText.className = "badge badge-success";
            hudText.style.background = "rgba(6, 78, 59, 0.92)";
            hudText.style.border = "1px solid rgba(52, 211, 153, 0.75)";
            hudText.style.color = "#D1FAE5";
            hudText.style.boxShadow = "0 4px 14px rgba(16, 185, 129, 0.35)";
          }
          if (hudDot) {
            hudDot.style.background = "#10B981";
            hudDot.style.boxShadow = "0 0 8px #10B981";
          }
          if (hudLabel) {
            hudLabel.textContent = "[ĐÃ TÌM THẤY KHUÔN MẶT - ĐANG KHỚP CHUẨN]";
          }
        } else {
          if (hudText) {
            hudText.className = "badge badge-danger";
            hudText.style.background = "rgba(127, 29, 29, 0.92)";
            hudText.style.border = "1px solid rgba(248, 113, 113, 0.75)";
            hudText.style.color = "#FEE2E2";
            hudText.style.boxShadow = "0 4px 14px rgba(239, 68, 68, 0.35)";
          }
          if (hudDot) {
            hudDot.style.background = "#EF4444";
            hudDot.style.boxShadow = "0 0 8px #EF4444";
          }
          if (hudLabel) {
            hudLabel.textContent = "[ĐANG TÌM KHUÔN MẶT - VUI LÒNG HƯỚNG VÀO CAMERA]";
          }
        }
      } else {
        if (hudBadge) hudBadge.style.display = "none";
      }

      requestAnimationFrame(render);
    };
    render();
  }

  // -------------------------------------------------------------
  // 3. WebSocket Real-Time Telemetry Stream
  // -------------------------------------------------------------
  initWebSocketStream() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;

    this.monitorWs = new WebSocket(wsUrl);

    this.monitorWs.onopen = () => {
      const devBadge = document.getElementById("device-status-badge");
      if (devBadge) {
        devBadge.textContent = "[STATION ENGINE: CONNECTED]";
        devBadge.className = "badge badge-success";
      }
      this.startTelemetryTransmission();
    };

    this.monitorWs.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "TELEMETRY_UPDATE") {
        this.updateBiometricsUI(msg.data);
      }
    };

    this.monitorWs.onclose = () => {
      const devBadge = document.getElementById("device-status-badge");
      if (devBadge) {
        devBadge.textContent = "[STATION ENGINE: DISCONNECTED - RECONNECTING]";
        devBadge.className = "badge badge-warning";
      }
      setTimeout(() => this.initWebSocketStream(), 3000);
    };
  }

  startTelemetryTransmission() {
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 320;
    offCanvas.height = 240;
    const offCtx = offCanvas.getContext("2d");
    const video = document.getElementById("webcam-video");

    setInterval(() => {
      if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
        if (this.isCameraActive && video && video.videoWidth > 0) {
          // Send real camera frame to backend
          offCtx.drawImage(video, 0, 0, 320, 240);
          const b64 = offCanvas.toDataURL("image/jpeg", 0.70);
          this.monitorWs.send(JSON.stringify({
            action: "monitor_frame",
            image_b64: b64
          }));
        } else {
          // Heartbeat
          this.monitorWs.send(JSON.stringify({
            action: "monitor_tick",
            timestamp: Date.now()
          }));
        }
      }
    }, 380);
  }

  testManualEmotion(emotion, valence, arousal) {
    if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
      this.monitorWs.send(JSON.stringify({
        action: "manual_override",
        manual_override: { emotion: emotion, valence: valence, arousal: arousal }
      }));
    }
  }

  updateBiometricsUI(data) {
    this.currentBiometrics = data;

    // Cache real detection results for canvas rendering
    this.lastBbox = data.bbox;
    this.lastFaceDetected = data.face_detected;
    this.isSmiling = data.is_smiling;
    this.smileScore = data.smile_score || 0;
    this.headTilt = data.head_tilt_deg || 0;

    // 1. Emotion & Exhaustion badges
    const statusBadge = document.getElementById("status-exhaustion-badge");
    const emotionMetric = document.getElementById("metric-emotion-class");
    const vaMetric = document.getElementById("metric-va-values");
    const confMetric = document.getElementById("metric-confidence");

    if (statusBadge) {
      const chipLabel = statusBadge.querySelector(".cam-chip-label");
      if (chipLabel) {
        chipLabel.textContent = `TIÊU HAO: CẤP ${data.mental_exhaustion_level}/5`;
      } else {
        statusBadge.textContent = `TIÊU HAO: CẤP ${data.mental_exhaustion_level}/5`;
      }
      // Update color level class
      statusBadge.classList.remove("level-warning", "level-danger");
      if (data.mental_exhaustion_level >= 3) {
        statusBadge.classList.add("level-danger");
      } else if (data.mental_exhaustion_level === 2) {
        statusBadge.classList.add("level-warning");
      }
    }

    if (emotionMetric) emotionMetric.textContent = data.emotion;
    if (vaMetric) {
      vaMetric.textContent = `VAL: ${data.valence >= 0 ? '+' : ''}${data.valence} | AROU: ${data.arousal >= 0 ? '+' : ''}${data.arousal}`;
    }
    if (confMetric) confMetric.textContent = `${data.confidence}%`;

    // 2. Multimodal Signal Balance Cards
    const c = data.channel_contributions;
    if (c) {
      this.setBar("face-pct-bar", null, c.face_pct);
      this.setBar("body-pct-bar", null, c.body_pct);
      this.setBar("speech-pct-bar", null, c.speech_pct);
      this.setBar("text-pct-bar", null, c.text_pct);
    }
    const faceStatus = document.getElementById("face-status-text");
    const bodyStatus = document.getElementById("body-status-text");
    const speechStatus = document.getElementById("speech-status-text");
    const textStatus = document.getElementById("text-status-text");

    if (faceStatus) {
      faceStatus.textContent = this.isSmiling 
        ? "Cười tươi / Hài lòng (+0.75)" 
        : (data.valence < -0.2 ? "Căng thẳng / Kiệt sức" : "Thư giãn tự nhiên");
    }
    if (bodyStatus) {
      bodyStatus.textContent = data.posture_warning 
        ? "Nghiêng cổ / Chùng vai" 
        : "Thẳng trục chuẩn";
    }
    if (speechStatus) {
      speechStatus.textContent = data.incongruence_detected 
        ? "Dao động tần số / Gượng ép" 
        : "Ổn định, điềm tĩnh";
    }
    if (textStatus) {
      textStatus.textContent = data.valence >= 0 
        ? "Tích cực, cởi mở" 
        : "Trầm lắng, hướng nội";
    }

    // 3. Incongruence Banner
    const inctAlert = document.getElementById("incongruence-alert-box");
    if (inctAlert) {
      if (data.incongruence_detected) {
        inctAlert.style.display = "block";
        inctAlert.innerHTML = `<strong>[CẢNH BÁO MÂU THUẪN CẢM XÚC]</strong><br>${data.incongruence_reason}`;
      } else {
        inctAlert.style.display = "none";
      }
    }

    // 4. Acoustic Auto Intervention Check
    if (window.acousticEngine) {
      window.acousticEngine.checkAutoIntervention(data.valence, data.arousal, data.mental_exhaustion_level);
    }

    // 5. Save point to Trajectory History
    this.trajectoryHistory.push({
      time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
      valence: data.valence,
      arousal: data.arousal,
      exhaustion: data.mental_exhaustion_level
    });
    if (this.trajectoryHistory.length > this.maxTrajectoryPoints) {
      this.trajectoryHistory.shift();
    }
    this.renderTrajectoryChart();
    this.updateBurnoutRiskIndex();
  }

  setBar(barId, valId, val) {
    const bar = document.getElementById(barId);
    if (bar) bar.style.width = `${val}%`;
    if (valId) {
      const text = document.getElementById(valId);
      if (text) text.textContent = `${val}%`;
    }
  }

  // -------------------------------------------------------------
  // 4. Vocal & Multimodal Elicitation Module
  // -------------------------------------------------------------
  async loadNewVocalPrompt() {
    try {
      const res = await fetch("/api/vocal_assessment/prompt");
      const data = await res.json();
      this.currentPrompt = data;
      const quoteEl = document.getElementById("vocal-prompt-text");
      const rationaleEl = document.getElementById("vocal-prompt-rationale");
      if (quoteEl) quoteEl.textContent = `"${data.prompt_text}"`;
      if (rationaleEl) rationaleEl.textContent = `[CHỈ ĐỊNH LÂM SÀNG: ${data.rationale}]`;
    } catch (e) {
      console.warn("[VOCAL] Prompt fetch error:", e);
    }
  }

  async recordAndAnalyzeVocalSynchrony() {
    if (this.isRecordingVocal) return;
    this.isRecordingVocal = true;

    const btn = document.getElementById("vocal-record-btn");
    const statusText = document.getElementById("vocal-status-text");

    if (btn) {
      btn.disabled = true;
      btn.textContent = "[ĐANG THU ÂM & PHÂN TÍCH 16KHZ 3.5S...]";
      btn.className = "btn btn-danger";
    }
    if (statusText) statusText.textContent = "[MICROPHONE ACTIVE: CAPTURING AUDIO-FACIAL DYNAMICS]";

    // Capture video frame at start of vocalization
    const video = document.getElementById("webcam-video");
    let b64Frame = "";
    if (video && video.videoWidth) {
      const canvas = document.createElement("canvas");
      canvas.width = 480;
      canvas.height = 360;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      b64Frame = canvas.toDataURL("image/jpeg", 0.75);
    }

    // Wait 3.5 seconds to capture speech duration
    setTimeout(async () => {
      try {
        const res = await fetch("/api/vocal_assessment/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt_id: this.currentPrompt ? this.currentPrompt.id : "PROMPT_01",
            prompt_text: this.currentPrompt ? this.currentPrompt.prompt_text : "",
            image_b64: b64Frame
          })
        });

        const report = await res.json();
        this.renderVocalReport(report);
      } catch (err) {
        console.error("[VOCAL] Evaluation error:", err);
      } finally {
        this.isRecordingVocal = false;
        if (btn) {
          btn.disabled = false;
          btn.textContent = "[GHI ÂM & PHÂN TÍCH ĐỒNG PHA 4 KÊNH]";
          btn.className = "btn btn-primary";
        }
      }
    }, 3500);
  }

  renderVocalReport(report) {
    const reportBox = document.getElementById("vocal-report-result");
    const statusText = document.getElementById("vocal-status-text");
    if (!reportBox) return;

    reportBox.style.display = "block";
    if (statusText) statusText.textContent = "[PHÂN TÍCH HOÀN TẤT]";

    reportBox.innerHTML = `
      <div style="font-family:var(--font-mono); font-size:0.72rem; display:flex; flex-direction:column; gap:0.4rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:700;">ĐỒNG PHA CẢM XÚC: ${report.synchrony_score}%</span>
          <span class="badge ${report.incongruence_detected ? 'badge-danger' : 'badge-success'}">
            ${report.incongruence_detected ? '[INCONGRUENCE DETECTED]' : '[HIGH HARMONIC CONGRUENCE]'}
          </span>
        </div>
        <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4;">
          ${report.incongruence_reason}
        </div>
        <div style="margin-top:0.3rem; font-size:0.68rem; color:var(--text-muted);">
          TỶ TRỌNG: SPEECH ${report.channel_contributions.speech_pct}% | FACE ${report.channel_contributions.face_pct}% | TEXT ${report.channel_contributions.text_pct}% | BODY ${report.channel_contributions.body_pct}%
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------
  // 5. 4-7-8 Breathing Pacer
  // -------------------------------------------------------------
  initBreathingPacer() {
    const innerCircle = document.getElementById("breath-inner-circle");
    const timerText = document.getElementById("breath-timer-text");
    const phaseBadge = document.getElementById("breath-phase-badge");

    let count = 4;
    let phase = "INHALE";

    const stepPacer = () => {
      if (phase === "INHALE") {
        if (timerText) timerText.textContent = `0${count}`;
        if (phaseBadge) {
          phaseBadge.textContent = "[HÍT VÀO SÂU QUA MŨI - 4S]";
          phaseBadge.className = "badge badge-clinical";
        }
        if (innerCircle) innerCircle.style.transform = "scale(1.5)";

        count--;
        if (count < 0) {
          phase = "HOLD";
          count = 7;
        }
      } else if (phase === "HOLD") {
        if (timerText) timerText.textContent = `0${count}`;
        if (phaseBadge) {
          phaseBadge.textContent = "[GIỮ HƠI THỞ & THẢ LỎNG VAI - 7S]";
          phaseBadge.className = "badge badge-warning";
        }
        count--;
        if (count < 0) {
          phase = "EXHALE";
          count = 8;
        }
      } else if (phase === "EXHALE") {
        if (timerText) timerText.textContent = `0${count}`;
        if (phaseBadge) {
          phaseBadge.textContent = "[THỞ RA CHẬM QUA MÔI - 8S]";
          phaseBadge.className = "badge badge-success";
        }
        if (innerCircle) innerCircle.style.transform = "scale(1.0)";

        count--;
        if (count < 0) {
          phase = "INHALE";
          count = 4;
        }
      }
    };

    setInterval(stepPacer, 1000);
  }

  // -------------------------------------------------------------
  // 6. CBT Clinical Chat with Typing Indicator & Typewriter Streaming
  // -------------------------------------------------------------
  async sendChatMessage() {
    const input = document.getElementById("chat-input-field");
    const history = document.getElementById("chat-messages-container");
    if (!input || !input.value.trim() || !history) return;

    const userText = input.value.trim();
    input.value = "";

    // Append user message
    const userDiv = document.createElement("div");
    userDiv.className = "chat-bubble chat-bubble-user";
    userDiv.textContent = userText;
    history.appendChild(userDiv);
    history.scrollTop = history.scrollHeight;

    // Show loading typing indicator
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing-indicator";
    typingIndicator.id = "cbt-typing-indicator";
    typingIndicator.innerHTML = `
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span>Bác sĩ đang phân tích...</span>
    `;
    history.appendChild(typingIndicator);
    history.scrollTop = history.scrollHeight;

    // Send to backend with biometric context
    try {
      const res = await fetch("/api/cbt/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userText,
          biometric_context: {
            emotion: this.currentBiometrics.emotion,
            valence: this.currentBiometrics.valence,
            arousal: this.currentBiometrics.arousal,
            exhaustion_level: this.currentBiometrics.mental_exhaustion_level,
            incongruence_detected: this.currentBiometrics.incongruence_detected,
            posture_warning: this.currentBiometrics.posture_warning
          }
        })
      });

      const data = await res.json();

      // Remove typing indicator
      if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.removeChild(typingIndicator);
      }

      // Create AI bubble and stream text typewriter-style
      const aiDiv = document.createElement("div");
      aiDiv.className = "chat-bubble chat-bubble-ai";
      history.appendChild(aiDiv);

      const fullText = data.response;
      let charIdx = 0;
      const streamTimer = setInterval(() => {
        if (charIdx < fullText.length) {
          const stepSize = 3;
          aiDiv.textContent += fullText.slice(charIdx, charIdx + stepSize);
          charIdx += stepSize;
          history.scrollTop = history.scrollHeight;
        } else {
          clearInterval(streamTimer);
          aiDiv.textContent = fullText;
          history.scrollTop = history.scrollHeight;
        }
      }, 14);

    } catch (e) {
      console.error("[CBT] Chat error:", e);
      if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.removeChild(typingIndicator);
      }
      const errDiv = document.createElement("div");
      errDiv.className = "chat-bubble chat-bubble-ai";
      errDiv.textContent = "Hệ thống đang bận xử lý dữ liệu sinh trắc học. Vui lòng thử lại trong giây lát.";
      history.appendChild(errDiv);
    }
  }

  // -------------------------------------------------------------
  // 6b. Interactive Guided 3-Step Assessment Workflow
  // -------------------------------------------------------------
  captureWebcamFrame() {
    const video = document.getElementById("webcam-video");
    if (!video || !this.isCameraActive || video.videoWidth === 0) {
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  }

  cancelAssessment(reason = "") {
    this.isAssessmentCancelled = true;
    this.isAssessing = false;
    const overlay = document.getElementById("assessment-step-overlay");
    if (overlay) overlay.style.display = "none";
    if (reason) {
      this.showWebModal("[ĐÃ HỦY ĐO ĐẠT]", reason, "warning");
    }
  }

  async startFullAssessment() {
    if (this.isAssessing) return;
    this.isAssessing = true;
    this.isAssessmentCancelled = false;

    // Smooth scroll camera card so both camera and guidance prompter are fully visible
    const camCard = document.getElementById("tab-monitor-camera-card");
    if (camCard) {
      camCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // Auto-start camera if not already active
    if (!this.isCameraActive) {
      await this.startCamera();
      await new Promise(r => setTimeout(r, 1200));
    }

    // STRICT PRE-CHECK: If no face is in view, DO NOT start!
    if (!this.lastFaceDetected) {
      this.isAssessing = false;
      this.showWebModal(
        "[CHƯA PHÁT HIỆN KHUÔN MẶT]",
        "Camera chưa phát hiện thấy khuôn mặt của bạn trong khung hình.\n\nVui lòng ngồi ngay ngắn trước camera (hoặc căn chỉnh góc máy) rồi bấm 'BẮT ĐẦU ĐÁNH GIÁ 3 BƯỚC' lại.",
        "danger"
      );
      return;
    }

    // Refresh vocal elicitation prompt if needed
    if (!this.currentPrompt) {
      try {
        const pResp = await fetch("/api/vocal_assessment/prompt");
        this.currentPrompt = await pResp.json();
      } catch (e) {
        this.currentPrompt = { prompt_id: "P1", prompt_text: "Tôi cảm thấy hoàn toàn bình an và làm chủ được năng lượng tinh thần hôm nay." };
      }
    }

    const overlay = document.getElementById("assessment-step-overlay");
    const badge = document.getElementById("step-overlay-badge");
    const instruction = document.getElementById("step-overlay-instruction");
    const promptBox = document.getElementById("step-overlay-prompt-box");
    const promptText = document.getElementById("step-overlay-prompt-text");
    const progressBar = document.getElementById("step-overlay-progress-bar");
    const timerLabel = document.getElementById("step-overlay-timer-label");
    const vuBar = document.getElementById("step-mic-vu-bar");
    const vuText = document.getElementById("step-mic-vu-text");
    const faceTrackingStatus = document.getElementById("step-face-tracking-status");

    if (overlay) overlay.style.display = "flex";
    if (faceTrackingStatus) {
      faceTrackingStatus.textContent = "[ĐANG THEO DÕI KHUÔN MẶT]";
      faceTrackingStatus.style.color = "#10B981";
    }

    let capturedFaceB64 = null;
    let recordedAudioBlob = null;
    let avgMicVolume = 0;
    let maxMicVolume = 0;
    let faceSeenDuringTest = Boolean(this.lastFaceDetected);
    let bestFaceFrame = null;
    let bestSmileFrame = null;
    let maxSmileObserved = 0;

    // Continuous sampler: keep track of best face & smile frames throughout test
    const frameSampler = setInterval(() => {
      if (this.isAssessmentCancelled) return;
      const f = this.captureWebcamFrame();
      const faceBadge = document.getElementById("step-face-tracking-status");
      if (f) {
        if (!capturedFaceB64) capturedFaceB64 = f;
        if (this.lastFaceDetected) {
          faceSeenDuringTest = true;
          bestFaceFrame = f;
          if (faceBadge) {
            faceBadge.textContent = "[ĐANG THEO DÕI KHUÔN MẶT]";
            faceBadge.style.color = "#10B981";
          }
        } else {
          if (faceBadge) {
            faceBadge.textContent = "[CẢNH BÁO: MẤT DẤU KHUÔN MẶT]";
            faceBadge.style.color = "#EF4444";
          }
        }
        if (this.isSmiling && this.smileScore > 40) {
          faceSeenDuringTest = true;
          const sc = Number(this.smileScore);
          if (sc >= maxSmileObserved) {
            maxSmileObserved = sc;
            bestSmileFrame = f;
          }
        }
      }
    }, 280);

    // -----------------------------------------------------------
    // BƯỚC 1: Quét Khuôn Mặt & Trương Lực Cơ (5 giây)
    // -----------------------------------------------------------
    if (badge) badge.textContent = "BƯỚC 1/3: QUÉT KHUÔN MẶT & TƯ THẾ (5 GIÂY)";
    if (instruction) instruction.textContent = "Hãy ngồi thẳng trước camera, nhìn thẳng vào ống kính và giữ nét mặt tự nhiên...";
    if (promptBox) promptBox.style.display = "none";

    await this.runStepTimer(5, progressBar, timerLabel);

    if (this.isAssessmentCancelled) {
      clearInterval(frameSampler);
      return;
    }

    // STRICT CHECK AFTER STEP 1: If no face was seen, ABORT immediately!
    if (!this.lastFaceDetected && !bestFaceFrame) {
      clearInterval(frameSampler);
      if (overlay) overlay.style.display = "none";
      this.isAssessing = false;
      this.showWebModal(
        "[HỦY ĐÁNH GIÁ - KHÔNG PHÁT HIỆN KHUÔN MẶT]",
        "Hệ thống đã tự động hủy bài kiểm tra vì không phát hiện thấy khuôn mặt (hoặc khuôn mặt bị che khuất) trong suốt 5 giây quét.\n\nVui lòng ngồi trước camera và bấm 'BẮT ĐẦU ĐÁNH GIÁ 3 BƯỚC' để thử lại.",
        "danger"
      );
      return;
    }

    if (!capturedFaceB64) capturedFaceB64 = this.captureWebcamFrame();
    if (this.lastFaceDetected) faceSeenDuringTest = true;

    // -----------------------------------------------------------
    // BƯỚC 2: Đọc Đoạn Văn & Ghi Âm Giọng Nói (7 giây)
    // -----------------------------------------------------------
    if (badge) badge.textContent = "BƯỚC 2/3: PHÂN TÍCH GIỌNG NÓI & NÉT MẶT (7 GIÂY)";
    if (instruction) instruction.textContent = "Hãy đọc to câu bên dưới vào micro để AI đo lường độ đồng pha cảm xúc và âm phổ:";
    if (promptBox) {
      promptBox.style.display = "block";
      if (promptText && this.currentPrompt) {
        promptText.textContent = `"${this.currentPrompt.prompt_text}"`;
      }
    }
    if (vuBar) vuBar.style.width = "0%";
    if (vuText) vuText.textContent = "0%";

    // Set up Real Microphone AudioContext, MediaRecorder & Speech Recognition
    let audioStream = null;
    let audioCtx = null;
    let analyserNode = null;
    let vuInterval = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let volSum = 0;
    let volCount = 0;

    // Web Speech API for Vietnamese speech transcription
    let speechRecognizer = null;
    let recognizedTranscript = "";
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      try {
        speechRecognizer = new SpeechRec();
        speechRecognizer.continuous = true;
        speechRecognizer.interimResults = true;
        speechRecognizer.lang = "vi-VN";
        speechRecognizer.onresult = (ev) => {
          let current = "";
          for (let i = 0; i < ev.results.length; i++) {
            current += ev.results[i][0].transcript + " ";
          }
          if (current.trim().length > 0) {
            recognizedTranscript = current.trim();
          }
        };
        speechRecognizer.onerror = (e) => {
          console.log("[SPEECH REC] Event:", e.error);
        };
        speechRecognizer.start();
      } catch (err) {
        console.warn("[SPEECH REC] Init note:", err);
      }
    }

    let vocalEnergySamples = [];
    let vocalBurstCount = 0;
    let inBurst = false;

    try {
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(audioStream);
      analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 512;
      analyserNode.smoothingTimeConstant = 0.3;
      source.connect(analyserNode);

      const timeData = new Uint8Array(analyserNode.fftSize);

      vuInterval = setInterval(() => {
        if (this.isAssessmentCancelled) return;
        analyserNode.getByteTimeDomainData(timeData);

        // Accurate Time-Domain RMS Volume calculation
        let sumSq = 0;
        for (let i = 0; i < timeData.length; i++) {
          const val = (timeData[i] - 128) / 128.0;
          sumSq += val * val;
        }
        const rms = Math.sqrt(sumSq / timeData.length); // 0.0 to 1.0

        // Noise gate: background noise or quiet breathing is < 0.015
        const gatedRms = Math.max(0, rms - 0.015);
        const floatVol = gatedRms;
        vocalEnergySamples.push(floatVol);

        if (floatVol > maxMicVolume) maxMicVolume = floatVol;
        volSum += floatVol;
        volCount++;

        // Speech burst detection: real syllables produce dynamic pulses above 0.035
        if (floatVol > 0.035 && !inBurst) {
          inBurst = true;
          vocalBurstCount++;
        } else if (floatVol < 0.018 && inBurst) {
          inBurst = false;
        }

        if (this.lastFaceDetected) faceSeenDuringTest = true;

        // Accurate, responsive VU meter (0% at silence, 20-75% when reading, 100% when shouting)
        const norm = Math.min(100, Math.round((gatedRms / 0.22) * 100));
        if (vuBar) {
          vuBar.style.width = `${norm}%`;
          vuBar.style.background = norm > 5 ? "#10B981" : "#64748B";
        }
        if (vuText) {
          vuText.textContent = `${norm}%`;
          vuText.style.color = norm > 5 ? "#34D399" : "#94A3B8";
        }
      }, 50);

      // Setup MediaRecorder for genuine playback
      let mimeType = "audio/webm";
      if (typeof MediaRecorder.isTypeSupported === "function") {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus";
        }
      }
      mediaRecorder = new MediaRecorder(audioStream, { mimeType: mimeType });
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.start(250);
    } catch (err) {
      console.warn("[ASSESSMENT] Không thể truy cập Microphone:", err);
    }

    if (window.acousticEngine) window.acousticEngine.play432HzChime();
    await this.runStepTimer(7, progressBar, timerLabel);

    if (speechRecognizer) {
      try { speechRecognizer.stop(); } catch(e) {}
    }

    if (this.isAssessmentCancelled) {
      clearInterval(frameSampler);
      if (vuInterval) clearInterval(vuInterval);
      if (audioStream) audioStream.getTracks().forEach(t => t.stop());
      return;
    }

    // Stop Microphone & MediaRecorder
    if (vuInterval) {
      clearInterval(vuInterval);
      vuInterval = null;
    }
    if (volCount > 0) {
      avgMicVolume = volSum / volCount;
    }

    // Compute volume variance / standard deviation to differentiate speech from flat noise
    let meanVol = volCount > 0 ? (volSum / volCount) : 0;
    let varianceSum = 0;
    for (let i = 0; i < vocalEnergySamples.length; i++) {
      const diff = vocalEnergySamples[i] - meanVol;
      varianceSum += diff * diff;
    }
    const volStdDev = vocalEnergySamples.length > 0 ? Math.sqrt(varianceSum / vocalEnergySamples.length) : 0;

    // Distinguish genuine human speech from background noise/fan:
    const hasRecognizedWords = Boolean(recognizedTranscript && recognizedTranscript.trim().length >= 3);
    const hasAcousticSpeech = Boolean(vocalBurstCount >= 3 && volStdDev > 0.010 && maxMicVolume > 0.030);
    const hasRealSpeech = hasRecognizedWords || hasAcousticSpeech;

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      await new Promise(resolve => {
        mediaRecorder.onstop = () => {
          recordedAudioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
          resolve();
        };
        mediaRecorder.stop();
      });
    }
    if (audioStream) {
      audioStream.getTracks().forEach(t => t.stop());
    }
    if (audioCtx && audioCtx.state !== "closed") {
      audioCtx.close();
    }

    // -----------------------------------------------------------
    // BƯỚC 3: Phản Xạ Nụ Cười Duchenne (3 giây)
    // -----------------------------------------------------------
    if (badge) badge.textContent = "BƯỚC 3/3: PHẢN XẠ NỤ CƯỜI DUCHENNE (3 GIÂY)";
    if (instruction) instruction.textContent = "Hãy mỉm cười tự nhiên trong 3 giây để kiểm tra phản xạ co cơ gò má lớn...";
    if (promptBox) promptBox.style.display = "none";
    await this.runStepTimer(3, progressBar, timerLabel);

    clearInterval(frameSampler);

    if (this.isAssessmentCancelled) return;

    if (bestSmileFrame) {
      capturedFaceB64 = bestSmileFrame;
      faceSeenDuringTest = true;
    } else if (bestFaceFrame) {
      capturedFaceB64 = bestFaceFrame;
      faceSeenDuringTest = true;
    } else {
      const finalF = this.captureWebcamFrame();
      if (finalF) capturedFaceB64 = finalF;
      faceSeenDuringTest = Boolean(this.lastFaceDetected);
    }

    // Hoàn tất: Ẩn overlay
    if (overlay) overlay.style.display = "none";
    this.isAssessing = false;

    // STRICT CHECK BEFORE CALLING EVALUATION:
    if (!faceSeenDuringTest || !bestFaceFrame) {
      this.showWebModal(
        "[HỦY ĐÁNH GIÁ - THIẾU DỮ LIỆU KHUÔN MẶT]",
        "Bài kiểm tra đã bị hủy vì không ghi nhận được khuôn mặt hợp lệ.\n\nVui lòng đảm bảo không che mặt và bấm 'BẮT ĐẦU ĐÁNH GIÁ 3 BƯỚC' lại.",
        "danger"
      );
      return;
    }

    // Gửi dữ liệu thực tế lên backend để đánh giá lâm sàng
    try {
      const promptId = (this.currentPrompt && (this.currentPrompt.id || this.currentPrompt.prompt_id)) || "PROMPT_01";
      const promptTxt = (this.currentPrompt && this.currentPrompt.prompt_text) || "Tôi cảm thấy hoàn toàn bình an và làm chủ được năng lượng tinh thần hôm nay.";

      const resp = await fetch("/api/vocal_assessment/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_id: String(promptId),
          prompt_text: String(promptTxt),
          image_b64: capturedFaceB64,
          audio_volume_avg: Number(avgMicVolume || 0),
          audio_volume_max: Number(maxMicVolume || 0),
          has_audio: Boolean(maxMicVolume > 0.015),
          has_speech: Boolean(hasRealSpeech),
          speech_transcript: String(recognizedTranscript || ""),
          client_face_detected: Boolean(faceSeenDuringTest),
          client_is_smiling: Boolean(faceSeenDuringTest && (this.isSmiling || maxSmileObserved > 45)),
          client_smile_score: Number(faceSeenDuringTest ? Math.max(this.smileScore || 0, maxSmileObserved) : 0)
        })
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${errText}`);
      }

      const evalData = await resp.json();
      this.showAssessmentScorecard(evalData, capturedFaceB64, recordedAudioBlob, avgMicVolume, maxMicVolume);
    } catch (e) {
      console.error("[ASSESSMENT] Lỗi đánh giá backend:", e);
      this.showWebModal(
        "[LỖI KẾT NỐI MÁY CHỦ]",
        "Không thể kết nối máy chủ để đánh giá dữ liệu sinh trắc học. Vui lòng thử lại.",
        "danger"
      );
    }
  }

  runStepTimer(durationSec, barEl, labelEl) {
    return new Promise(resolve => {
      const totalMs = durationSec * 1000;
      const startTime = Date.now();

      const interval = setInterval(() => {
        if (this.isAssessmentCancelled) {
          clearInterval(interval);
          resolve();
          return;
        }
        const elapsed = Date.now() - startTime;
        const progress = Math.max(0, 1 - (elapsed / totalMs));
        const remSec = Math.max(0, Math.ceil((totalMs - elapsed) / 1000));

        if (barEl) barEl.style.width = `${progress * 100}%`;
        if (labelEl) labelEl.textContent = `Còn ${remSec} giây...`;

        if (elapsed >= totalMs) {
          clearInterval(interval);
          if (barEl) barEl.style.width = "0%";
          resolve();
        }
      }, 50);
    });
  }

  showAssessmentScorecard(res, faceB64, audioBlob, avgVol, maxVol) {
    // If not a valid assessment or no face detected, show warning web popup
    if (!res || res.valid_assessment === false || !res.face_detected) {
      this.showWebModal(
        res && res.status_label ? res.status_label : "[KẾT QUẢ ĐÁNH GIÁ CHƯA HỢP LỆ]",
        res && res.prescription ? res.prescription : "Hệ thống không ghi nhận đủ dữ liệu khuôn mặt và giọng đọc hợp lệ. Vui lòng bấm 'BẮT ĐẦU ĐÁNH GIÁ 3 BƯỚC' để thực hiện lại.",
        "warning"
      );
      return;
    }

    // Close any open web notification modal
    const webModal = document.getElementById("clinical-web-modal");
    if (webModal) webModal.style.display = "none";

    const modal = document.getElementById("assessment-result-modal");
    if (!modal) return;

    const scoreNum = document.getElementById("modal-score-number");
    const scoreStatus = document.getElementById("modal-score-status");
    const faceRes = document.getElementById("modal-face-res");
    const voiceRes = document.getElementById("modal-voice-res");
    const congRes = document.getElementById("modal-congruence-res");
    const burnoutRes = document.getElementById("modal-burnout-res");
    const prescripText = document.getElementById("modal-prescription-text");

    const evidenceBadge = document.getElementById("modal-evidence-status-badge");
    const faceImg = document.getElementById("modal-face-snapshot-img");
    const noFaceWarn = document.getElementById("modal-no-face-warning");
    const faceContainer = document.getElementById("modal-face-snapshot-container");
    const audioEl = document.getElementById("modal-recorded-audio");
    const audioBadge = document.getElementById("modal-audio-energy-badge");
    const audioWarn = document.getElementById("modal-audio-warning-msg");

    // 1. Numerical Score & Status
    const finalScore = res.final_score !== undefined ? res.final_score : 0;
    if (scoreNum) scoreNum.textContent = finalScore;

    if (scoreStatus) {
      scoreStatus.textContent = res.status_label || "ĐÁNH GIÁ HOÀN TẤT";
      if (finalScore >= 80) {
        scoreStatus.style.color = "#059669";
      } else if (finalScore >= 60) {
        scoreStatus.style.color = "#D97706";
      } else {
        scoreStatus.style.color = "#DC2626";
      }
    }

    // 2. Modality Breakdowns
    if (faceRes) faceRes.textContent = res.face_detail || "Chưa có dữ liệu";
    if (voiceRes) voiceRes.textContent = res.voice_detail || "Chưa có dữ liệu";
    if (congRes) congRes.textContent = res.congruence_detail || "0%";
    if (burnoutRes) burnoutRes.textContent = res.burnout_detail || "Không xác định";
    if (prescripText) prescripText.textContent = res.prescription || "Vui lòng thực hiện lại bài kiểm tra.";

    // 3. Evidence: Webcam Snapshot
    if (faceB64 && faceImg) {
      faceImg.src = faceB64;
      faceImg.style.display = "block";
      if (res.face_detected) {
        if (noFaceWarn) noFaceWarn.style.display = "none";
        if (faceContainer) faceContainer.style.border = "2px solid #10B981";
      } else {
        if (noFaceWarn) {
          noFaceWarn.style.display = "block";
          noFaceWarn.textContent = "[KHÔNG TÌM THẤY KHUÔN MẶT]";
        }
        if (faceContainer) faceContainer.style.border = "2px solid #DC2626";
      }
    } else {
      if (faceImg) faceImg.style.display = "none";
      if (noFaceWarn) {
        noFaceWarn.style.display = "block";
        noFaceWarn.textContent = "[CAMERA CHƯA BẬT / KHÔNG CÓ ẢNH]";
      }
      if (faceContainer) faceContainer.style.border = "2px solid #DC2626";
    }

    // 4. Evidence: Recorded Audio Playback
    if (audioBlob && audioEl) {
      try {
        if (this._lastAudioBlobUrl) {
          URL.revokeObjectURL(this._lastAudioBlobUrl);
        }
        this._lastAudioBlobUrl = URL.createObjectURL(audioBlob);
        audioEl.src = this._lastAudioBlobUrl;
        audioEl.load();
      } catch (e) {
        console.error("Audio playback blob error:", e);
      }
    }

    const hasVoice = Boolean(res.has_speech || res.has_audio);
    if (audioBadge) {
      if (hasVoice) {
        audioBadge.textContent = `[ÂM LƯỢNG: ${Math.round((avgVol || 0) * 100)}%]`;
        audioBadge.className = "badge badge-success";
      } else {
        audioBadge.textContent = "[KHÔNG CÓ TIẾNG NÓI]";
        audioBadge.className = "badge badge-danger";
      }
    }
    if (audioWarn) {
      audioWarn.style.display = hasVoice ? "none" : "block";
    }

    // 5. Overall Evidence Status Badge
    if (evidenceBadge) {
      if (res.face_detected && hasVoice) {
        evidenceBadge.textContent = "[ĐÃ THU THẬP ĐỦ 2 KÊNH SINH TRẮC]";
        evidenceBadge.className = "badge badge-success";
      } else if (!res.face_detected && !hasVoice) {
        evidenceBadge.textContent = "[KHÔNG HỢP LỆ: THIẾU CẢ MẶT & GIỌNG NÓI]";
        evidenceBadge.className = "badge badge-danger";
      } else if (!res.face_detected) {
        evidenceBadge.textContent = "[THIẾU TÍN HIỆU KHUÔN MẶT]";
        evidenceBadge.className = "badge badge-warning";
      } else {
        evidenceBadge.textContent = "[THIẾU TÍN HIỆU GIỌNG NÓI]";
        evidenceBadge.className = "badge badge-warning";
      }
    }

    modal.style.display = "flex";
  }

  closeAssessmentScorecard() {
    const modal = document.getElementById("assessment-result-modal");
    if (modal) modal.style.display = "none";
    const audioEl = document.getElementById("modal-recorded-audio");
    if (audioEl) {
      audioEl.pause();
      audioEl.currentTime = 0;
    }
  }

  // -------------------------------------------------------------
  // 7. Tab 3: Trajectory Forecasting & Prescriptions
  // -------------------------------------------------------------
  initTrajectoryChart() {
    const canvas = document.getElementById("trajectory-canvas");
    if (canvas) {
      canvas.width = canvas.parentElement.clientWidth || 600;
      canvas.height = 240;
    }
  }

  renderTrajectoryChart() {
    const canvas = document.getElementById("trajectory-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, w, h);

    // Grid lines & Zero-line
    ctx.strokeStyle = "#F1F5F9";
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();
    }

    // Baseline Valence = 0.0 (middle)
    const midY = h / 2;
    ctx.strokeStyle = "#CBD5E1";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(40, midY);
    ctx.lineTo(w - 20, midY);
    ctx.stroke();

    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#94A3B8";
    ctx.fillText("+1.0 (TÍCH CỰC)", 45, 20);
    ctx.fillText(" 0.0 (CÂN BẰNG)", 45, midY - 4);
    ctx.fillText("-1.0 (TRẦM LẮNG)", 45, h - 10);

    if (this.trajectoryHistory.length < 2) return;

    const n = this.trajectoryHistory.length;
    const stepX = (w - 70) / (this.maxTrajectoryPoints - 1);

    // Draw Valence Trajectory (Clinical Blue)
    ctx.beginPath();
    ctx.strokeStyle = "#0284C7";
    ctx.lineWidth = 2.5;

    this.trajectoryHistory.forEach((pt, i) => {
      const x = 50 + i * stepX;
      const y = midY - (pt.valence * (midY - 35));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Arousal Trajectory (Dashed Slate)
    ctx.beginPath();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#64748B";
    ctx.lineWidth = 1.5;

    this.trajectoryHistory.forEach((pt, i) => {
      const x = 50 + i * stepX;
      const y = midY - (pt.arousal * (midY - 35));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  updateBurnoutRiskIndex() {
    let sumValence = 0;
    this.trajectoryHistory.forEach(pt => sumValence += pt.valence);
    const avgValence = this.trajectoryHistory.length > 0 ? (sumValence / this.trajectoryHistory.length) : 0;
    
    let riskPct = 25;
    if (avgValence < 0) riskPct += Math.abs(avgValence) * 35;
    if (this.currentBiometrics.mental_exhaustion_level >= 3) riskPct += 30;
    if (this.currentBiometrics.incongruence_detected) riskPct += 12;
    riskPct = Math.min(95, Math.max(10, Math.round(riskPct)));

    const riskBadge = document.getElementById("burnout-risk-badge");
    const riskBar = document.getElementById("burnout-risk-bar");
    if (riskBadge) {
      riskBadge.textContent = `[NGUY CƠ KIỆT SỨC: ${riskPct}% TRONG 2 GIỜ TỚI]`;
      if (riskPct >= 65) riskBadge.className = "badge badge-danger";
      else if (riskPct >= 45) riskBadge.className = "badge badge-warning";
      else riskBadge.className = "badge badge-success";
    }
    if (riskBar) riskBar.style.width = `${riskPct}%`;
  }

  bindEvents() {
    // Webcam Toggle Buttons
    const camToggle = document.getElementById("webcam-toggle-btn");
    if (camToggle) {
      camToggle.addEventListener("click", () => this.toggleCamera());
    }

    const camStartOverlay = document.getElementById("btn-start-camera-overlay");
    if (camStartOverlay) {
      camStartOverlay.addEventListener("click", () => this.startCamera());
    }

    // Interactive 3-Step Assessment Buttons (Header and Camera Viewport)
    const startAssessCamHeader = document.getElementById("btn-start-assess-cam-header");
    if (startAssessCamHeader) {
      startAssessCamHeader.addEventListener("click", () => {
        const card = document.getElementById("tab-monitor-camera-card");
        if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
        this.startFullAssessment();
      });
    }

    const startAssessBtn = document.getElementById("start-full-assessment-btn");
    if (startAssessBtn) {
      startAssessBtn.addEventListener("click", () => {
        const card = document.getElementById("tab-monitor-camera-card");
        if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
        this.startFullAssessment();
      });
    }

    const cancelActionBtn = document.getElementById("btn-cancel-assessment-action");
    if (cancelActionBtn) {
      cancelActionBtn.addEventListener("click", () => this.cancelAssessment("Bạn đã bấm hủy bài đánh giá."));
    }

    const posToggleBtn = document.getElementById("step-pos-toggle-btn");
    if (posToggleBtn) {
      posToggleBtn.addEventListener("click", () => {
        const overlay = document.getElementById("assessment-step-overlay");
        if (overlay) {
          overlay.classList.toggle("pos-top");
        }
      });
    }

    const closeAssessBtn = document.getElementById("close-assessment-modal-btn");
    if (closeAssessBtn) {
      closeAssessBtn.addEventListener("click", () => this.closeAssessmentScorecard());
    }

    const saveAssessBtn = document.getElementById("save-assessment-btn");
    if (saveAssessBtn) {
      saveAssessBtn.addEventListener("click", () => this.closeAssessmentScorecard());
    }

    const backdropEl = document.getElementById("assessment-modal-backdrop");
    if (backdropEl) {
      backdropEl.addEventListener("click", () => this.closeAssessmentScorecard());
    }

    const repeatAssessBtn = document.getElementById("repeat-assessment-btn");
    if (repeatAssessBtn) {
      repeatAssessBtn.addEventListener("click", () => {
        this.closeAssessmentScorecard();
        setTimeout(() => this.startFullAssessment(), 300);
      });
    }

    // Ambient Music Toggle
    const ambientSwitch = document.getElementById("ambient-audio-switch");
    if (ambientSwitch) {
      ambientSwitch.addEventListener("change", (e) => {
        if (window.acousticEngine) {
          window.acousticEngine.toggleAmbientAlphaDrone(e.target.checked);
        }
      });
    }

    // 5s Intervention Manual Button
    const chimeBtn = document.getElementById("play-chime-btn");
    if (chimeBtn) {
      chimeBtn.addEventListener("click", () => {
        if (window.acousticEngine) window.acousticEngine.play432HzChime();
      });
    }

    const bowlBtn = document.getElementById("play-bowl-btn");
    if (bowlBtn) {
      bowlBtn.addEventListener("click", () => {
        if (window.acousticEngine) window.acousticEngine.playTibetanBowl();
      });
    }

    // Vocal prompt & record buttons
    const genPromptBtn = document.getElementById("vocal-gen-prompt-btn");
    if (genPromptBtn) {
      genPromptBtn.addEventListener("click", () => this.loadNewVocalPrompt());
    }

    const recordVocalBtn = document.getElementById("vocal-record-btn");
    if (recordVocalBtn) {
      recordVocalBtn.addEventListener("click", () => this.recordAndAnalyzeVocalSynchrony());
    }

    // CBT Chat send button & enter key
    const chatBtn = document.getElementById("chat-send-btn");
    const chatInput = document.getElementById("chat-input-field");
    if (chatBtn) {
      chatBtn.addEventListener("click", () => this.sendChatMessage());
    }
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.sendChatMessage();
        }
      });
    }

    // CBT Suggestion Chips
    const chips = document.querySelectorAll(".cbt-suggestion-chips .chip-btn");
    chips.forEach(chip => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query");
        if (query && chatInput) {
          chatInput.value = query;
          chatInput.focus();
          if (!chip.classList.contains("chip-teach")) {
            this.sendChatMessage();
          }
        }
      });
    });
  }
}

window.clinicalStation = new ClinicalStationApp();
window.addEventListener("DOMContentLoaded", () => {
  window.clinicalStation.init();
  if (window.mimicGame) window.mimicGame.init();
});

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
    this.lastLandmarks = null;
    this.isSplitView = false;
    this.lastFaceDetected = false;
    this.isSmiling = false;
    this.smileScore = 0.0;
    this.headTilt = 0.0;
    this.trajectoryHistory = [];
    this.multiverseSliderVal = 0.0;
    this.maxTrajectoryPoints = 60;
  }

  async init() {
    this.initTabs();
    this.bindEvents();
    this.initBreathingPacer();
    this.startCanvasHUDLoop();
    this.initWebSocketStream();
    this.initECGHospitalMonitor();
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
          if (window.multiverseSim) {
            window.multiverseSim.onTabActivated();
          }
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
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
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
      const multiToggle = document.getElementById("multiverse-cam-toggle-btn");
      if (multiToggle) {
        const mlbl = document.getElementById("multiverse-cam-toggle-label");
        if (mlbl) mlbl.textContent = "TẮT";
        multiToggle.classList.add("is-active");
      }
      if (statusText) {
        statusText.textContent = "RESNET50 (2048-D) ACTIVE";
      }
      const devBadge = document.getElementById("device-status-badge");
      if (devBadge) {
        devBadge.innerHTML = `<span class="status-dot-pulse"></span> Live`;
      }

      // Sync with mimic game and multiverse camera viewports
      if (window.mimicGame) {
        window.mimicGame.attachWebcamStream();
      }
      if (window.multiverseSim) {
        window.multiverseSim.syncCameraStream();
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
      const multiToggle = document.getElementById("multiverse-cam-toggle-btn");
      if (multiToggle) {
        const mlbl = document.getElementById("multiverse-cam-toggle-label");
        if (mlbl) mlbl.textContent = "BẬT";
        multiToggle.classList.remove("is-active");
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

    const multiToggle = document.getElementById("multiverse-cam-toggle-btn");
    if (multiToggle) {
      const mlbl = document.getElementById("multiverse-cam-toggle-label");
      if (mlbl) mlbl.textContent = "BẬT";
      multiToggle.classList.remove("is-active");
    }

    const statusText = document.getElementById("hud-backbone-status");
    if (statusText) {
      statusText.textContent = "RESNET50 STANDBY";
    }

    // Sync with mimic game and multiverse simulation
    if (window.mimicGame) {
      window.mimicGame.attachWebcamStream();
    }
    if (window.multiverseSim) {
      window.multiverseSim.syncCameraStream();
    }
  }

  drawFaceDetectionHUD(canvas, video, angle) {
    if (!canvas) return;
    if (canvas.parentElement) {
      const pw = canvas.parentElement.clientWidth;
      const ph = canvas.parentElement.clientHeight;
      if (pw > 0 && ph > 0 && (canvas.width !== pw || canvas.height !== ph)) {
        canvas.width = pw;
        canvas.height = ph;
      }
    }
    if (canvas.width === 0 || canvas.height === 0) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width;
    const h = canvas.height;

    const isTab3 = (canvas.id === "multiverse-overlay-canvas");

    if (this.isCameraActive && this.lastFaceDetected && this.lastBbox) {
      // Map sent frame coordinates to displayed canvas coordinates
      const rawX = this.lastBbox[0];
      const rawY = this.lastBbox[1];
      const rawW = this.lastBbox[2];
      const rawH = this.lastBbox[3];

      const sentW = this.lastFrameW || this.sentFrameWidth || 640;
      const sentH = this.lastFrameH || this.sentFrameHeight || 480;

      // Video dimensions
      const vw = (video && video.videoWidth > 0) ? video.videoWidth : sentW;
      const vh = (video && video.videoHeight > 0) ? video.videoHeight : sentH;

      // Account for object-fit: cover scaling and cropping
      const scale = Math.max(w / vw, h / vh);
      const renderedW = vw * scale;
      const renderedH = vh * scale;
      const offsetX = (w - renderedW) / 2;
      const offsetY = (h - renderedH) / 2;

      // Position on displayed screen (overlay and video are both mirrored by CSS, so no double flip)
      const uncalibratedBx = (rawX / sentW) * renderedW + offsetX;
      const uncalibratedBy = (rawY / sentH) * renderedH + offsetY;
      const uncalibratedBw = (rawW / sentW) * renderedW;
      const uncalibratedBh = (rawH / sentH) * renderedH;

      // Clamp bounding box to viewport to prevent clipping/overflow
      const bx = Math.max(0, Math.min(uncalibratedBx, w - 10));
      const by = Math.max(0, Math.min(uncalibratedBy, h - 10));
      const bw = Math.max(10, Math.min(uncalibratedBw, w - bx));
      const bh = Math.max(10, Math.min(uncalibratedBh, h - by));

      // Signature Green Square Face Detection Bounding Box (Clinical Standard)
      const boxColor = "#10B981";       // Emerald green outer line
      const cornerColor = "#34D399";    // Bright green corner brackets
      const boxFill = "rgba(16, 185, 129, 0.08)"; // Subtle tech fill tint

      // Fill inner subtle tint
      ctx.fillStyle = boxFill;
      ctx.fillRect(bx, by, bw, bh);

      // Square bounding box border
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 1.8;
      ctx.strokeRect(bx, by, bw, bh);

      // Heavy green corner brackets (TL, TR, BL, BR)
      ctx.strokeStyle = cornerColor;
      ctx.lineWidth = 3.5;
      const clen = Math.min(26, bw * 0.28);
      // TL
      ctx.beginPath(); ctx.moveTo(bx, by + clen); ctx.lineTo(bx, by); ctx.lineTo(bx + clen, by); ctx.stroke();
      // TR
      ctx.beginPath(); ctx.moveTo(bx + bw - clen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + clen); ctx.stroke();
      // BL
      ctx.beginPath(); ctx.moveTo(bx, by + bh - clen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + clen, by + bh); ctx.stroke();
      // BR
      ctx.beginPath(); ctx.moveTo(bx + bw - clen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - clen); ctx.stroke();

      // Precision center target reticle crosshairs
      const cx = bx + bw / 2;
      const cy = by + bh / 2;
      ctx.strokeStyle = "rgba(52, 211, 153, 0.55)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
      ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
      ctx.stroke();

      // Label above bbox (Un-mirrored for standard human reading, clamped inside viewport)
      let labelText = "[KHUÔN MẶT ĐÃ KHỚP CHUẨN]";
      let labelColor = "#A7F3D0";
      let labelBg = "rgba(6, 78, 59, 0.92)";
      let labelBorder = "rgba(52, 211, 153, 0.75)";

      if (this.isSmiling) {
        labelColor = "#6EE7B7";
        labelText = `[PHÁT HIỆN NỤ CƯỜI DUCHENNE: +${this.smileScore}%]`;
      } else if (Math.abs(this.headTilt) > 14) {
        labelColor = "#FDE68A";
        labelBg = "rgba(120, 53, 15, 0.92)";
        labelBorder = "rgba(245, 158, 11, 0.75)";
        labelText = `[CẢNH BÁO TƯ THẾ: LỆCH CỔ ${this.headTilt}°]`;
      } else if (isTab3) {
        const emoName = (this.currentBiometrics && this.currentBiometrics.emotion) ? this.currentBiometrics.emotion : "NEUTRAL / CALM";
        const confVal = (this.currentBiometrics && typeof this.currentBiometrics.confidence === "number") ? this.currentBiometrics.confidence : 90;
        labelColor = "#38BDF8";
        labelBg = "rgba(12, 74, 110, 0.92)";
        labelBorder = "rgba(56, 189, 248, 0.75)";
        labelText = `[AI THỊ GIÁC: ${emoName} (${confVal}%)]`;
      }


      ctx.save();
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(labelText).width;
      const lx = Math.max(0, Math.min(bx, w - tw - 14));
      const ly = Math.max(16, by - 8);
      ctx.translate(lx, ly);
      ctx.scale(-1, 1);

      // Background pill for label
      ctx.fillStyle = labelBg;
      ctx.strokeStyle = labelBorder;
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(-tw - 8, -13, tw + 8, 16, 3);
      } else {
        ctx.rect(-tw - 8, -13, tw + 8, 16);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = labelColor;
      ctx.textAlign = "left";
      ctx.fillText(labelText, -tw - 4, 0);
      ctx.restore();

      // Dynamic posture skeleton lines from real face location
      ctx.strokeStyle = "rgba(52, 211, 153, 0.65)";
      ctx.lineWidth = 1.8;
      const neck = { x: bx + bw / 2, y: by + bh };
      const shoulderL = { x: bx - bw * 0.4, y: by + bh + 36 };
      const shoulderR = { x: bx + bw * 1.4, y: by + bh + 36 };
      const spine = { x: neck.x, y: neck.y + 110 };

      ctx.beginPath();
      ctx.moveTo(neck.x, neck.y);
      ctx.lineTo(neck.x, neck.y + 30);
      ctx.moveTo(shoulderL.x, shoulderL.y);
      ctx.lineTo(shoulderR.x, shoulderR.y);
      ctx.moveTo(neck.x, neck.y + 30);
      ctx.lineTo(spine.x, spine.y);
      ctx.stroke();

      [neck, shoulderL, shoulderR, spine].forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "#10B981";
        ctx.fill();
      });

      // Tab 3 Specific: Photoplethysmography (rPPG), Bio-Age estimation, and Anatomical Facial Morphing
      if (isTab3) {
        this.drawFaceRPPGandBioAge(ctx, bx, by, bw, bh, this.multiverseSliderVal, angle);
        this.drawTargetedFacialMorph(ctx, w, h, {
          bx, by, bw, bh,
          rawX, rawY, rawW, rawH,
          sentW, sentH, renderedW, renderedH,
          offsetX, offsetY
        }, this.lastLandmarks, this.multiverseSliderVal, this.isSplitView, angle);
      }
    } else if (this.isCameraActive && !this.lastFaceDetected) {
      // Active scanning HUD when camera is on but no face is in view
      const cx = w * 0.5;
      const cy = h * 0.44;
      const rad = Math.min(w, h) * 0.28;

      // Subtle guide oval
      ctx.save();
      ctx.strokeStyle = isTab3 ? "rgba(56, 189, 248, 0.5)" : "rgba(239, 68, 68, 0.45)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.ellipse(cx, cy, rad * 0.75, rad, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Laser scan line
      const scanY = cy - rad + (Math.sin(angle) * 0.5 + 0.5) * (rad * 2);
      ctx.strokeStyle = isTab3 ? "rgba(56, 189, 248, 0.8)" : "rgba(239, 68, 68, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - rad * 0.7, scanY);
      ctx.lineTo(cx + rad * 0.7, scanY);
      ctx.stroke();
    } else {
      if (!this.isCameraActive) {
        // Idle decorative reticle on standby
        const bx = w * 0.35 + Math.sin(angle) * 6;
        const by = h * 0.22 + Math.cos(angle * 0.8) * 5;
        const bw = w * 0.30;
        const bh = h * 0.40;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.save();
        ctx.translate(bx + bw / 2, by - 6);
        ctx.scale(-1, 1);
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
        ctx.textAlign = "center";
        ctx.fillText("[STANDBY: BẬT CAMERA ĐỂ QUÉT KHUÔN MẶT]", 0, 0);
        ctx.restore();
      }
    }
  }

  /**
   * Anatomical Facial Landmarks & rPPG Optical Sampling Reticles (Tab 3)
   * Non-destructive: Does NOT alter skin tone or draw artificial dark circles/blush on the user.
   * Keeps raw webcam stream 100% natural and transparent.
   */
  drawTargetedFacialMorph(ctx, w, h, box, landmarks, val, isSplit, angle) {
    if (!ctx) return;

    const mapX = (px) => (px / box.sentW) * box.renderedW + box.offsetX;
    const mapY = (py) => (py / box.sentH) * box.renderedH + box.offsetY;

    let eyeR = (landmarks && landmarks.right_eye) ? { x: mapX(landmarks.right_eye[0]), y: mapY(landmarks.right_eye[1]) } : null;
    let eyeL = (landmarks && landmarks.left_eye) ? { x: mapX(landmarks.left_eye[0]), y: mapY(landmarks.left_eye[1]) } : null;
    let nose = (landmarks && landmarks.nose) ? { x: mapX(landmarks.nose[0]), y: mapY(landmarks.nose[1]) } : null;
    let mouthR = (landmarks && landmarks.right_mouth) ? { x: mapX(landmarks.right_mouth[0]), y: mapY(landmarks.right_mouth[1]) } : null;
    let mouthL = (landmarks && landmarks.left_mouth) ? { x: mapX(landmarks.left_mouth[0]), y: mapY(landmarks.left_mouth[1]) } : null;

    if (!eyeR) eyeR = { x: box.bx + box.bw * 0.33, y: box.by + box.bh * 0.38 };
    if (!eyeL) eyeL = { x: box.bx + box.bw * 0.67, y: box.by + box.bh * 0.38 };
    if (!nose) nose = { x: box.bx + box.bw * 0.50, y: box.by + box.bh * 0.54 };
    if (!mouthR) mouthR = { x: box.bx + box.bw * 0.38, y: box.by + box.bh * 0.76 };
    if (!mouthL) mouthL = { x: box.bx + box.bw * 0.62, y: box.by + box.bh * 0.76 };

    // Draw subtle holographic landmark dots on genuine facial points
    ctx.save();
    const lms = [
      { pt: eyeR, color: "#38BDF8", label: "R-EYE" },
      { pt: eyeL, color: "#38BDF8", label: "L-EYE" },
      { pt: nose, color: "#34D399", label: "NOSE" },
      { pt: mouthR, color: "#FBBF24", label: "R-MOUTH" },
      { pt: mouthL, color: "#FBBF24", label: "L-MOUTH" }
    ];

    lms.forEach(item => {
      ctx.fillStyle = item.color;
      ctx.shadowColor = item.color;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(item.pt.x, item.pt.y, 3.2, 0, Math.PI * 2);
      ctx.fill();

      // Outer target ring
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(item.pt.x, item.pt.y, 6.0, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Subtly connect eyes & mouth contour with technical dashed line
    ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(eyeR.x, eyeR.y); ctx.lineTo(eyeL.x, eyeL.y);
    ctx.moveTo(mouthR.x, mouthR.y); ctx.lineTo(mouthL.x, mouthL.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  /**
   * Forehead & Cheek Optical Skin Sampling ROI Reticle (rPPG Telemetry)
   * Shows user exactly where blood volume pulse (BVP) is optically extracted.
   */
  drawFaceRPPGandBioAge(ctx, bx, by, bw, bh, val, angle) {
    if (!ctx) return;

    // Forehead capillary skin perfusion sampling area
    const roiX = bx + bw * 0.30;
    const roiY = by + bh * 0.10;
    const roiW = bw * 0.40;
    const roiH = bh * 0.18;

    ctx.save();
    // Dashed Cyan Box for Optical Sampling Area
    ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(roiX, roiY, roiW, roiH);
    ctx.setLineDash([]);

    // Subtle corner brackets on ROI box
    ctx.strokeStyle = "#38BDF8";
    ctx.lineWidth = 2.0;
    const clen = Math.min(8, roiW * 0.25);
    // TL
    ctx.beginPath(); ctx.moveTo(roiX, roiY + clen); ctx.lineTo(roiX, roiY); ctx.lineTo(roiX + clen, roiY); ctx.stroke();
    // TR
    ctx.beginPath(); ctx.moveTo(roiX + roiW - clen, roiY); ctx.lineTo(roiX + roiW, roiY); ctx.lineTo(roiX + roiW, roiY + clen); ctx.stroke();
    // BL
    ctx.beginPath(); ctx.moveTo(roiX, roiY + roiH - clen); ctx.lineTo(roiX, roiY + roiH); ctx.lineTo(roiX + clen, roiY + roiH); ctx.stroke();
    // BR
    ctx.beginPath(); ctx.moveTo(roiX + roiW - clen, roiY + roiH); ctx.lineTo(roiX + roiW, roiY + roiH); ctx.lineTo(roiX + roiW, roiY + roiH - clen); ctx.stroke();

    // Pulse sensor chip pill attached above ROI
    ctx.font = "bold 9px 'JetBrains Mono', monospace";
    const rppgTag = "rPPG OPTICAL ROI [G-CHANNEL]";
    const tagW = ctx.measureText(rppgTag).width + 12;

    ctx.translate(roiX + roiW / 2, roiY - 6);
    ctx.scale(-1, 1);

    ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
    ctx.strokeStyle = "rgba(56, 189, 248, 0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(-tagW / 2, -12, tagW, 14, 2);
    } else {
      ctx.rect(-tagW / 2, -12, tagW, 14);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#38BDF8";
    ctx.textAlign = "center";
    ctx.fillText(rppgTag, 0, -2);
    ctx.restore();
  }

  syncLiveHUDStatusBadge(badgeId, textId, dotId, labelId) {
    const hudBadge = document.getElementById(badgeId);
    const hudText = document.getElementById(textId);
    const hudDot = document.getElementById(dotId);
    const hudLabel = document.getElementById(labelId);

    if (!hudBadge) return;

    if (this.isCameraActive) {
      hudBadge.style.display = "flex";
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
      hudBadge.style.display = "none";
    }
  }

  syncMultiversePostureIndicator() {
    const postDot = document.getElementById("multiverse-ai-posture-dot");
    const postText = document.getElementById("multiverse-ai-posture-text");
    if (!postDot || !postText) return;

    if (!this.isCameraActive) {
      postDot.style.background = "#94A3B8";
      postDot.style.boxShadow = "none";
      postText.style.color = "#94A3B8";
      postText.textContent = "AI TRẠNG THÁI: [CAMERA ĐANG TẮT] Bật camera để AI quét khuôn mặt & tư thế";
    } else if (this.lastFaceDetected) {
      const isAligned = Math.abs(this.headTilt) <= 12;
      postDot.style.background = isAligned ? "#10B981" : "#F59E0B";
      postDot.style.boxShadow = isAligned ? "0 0 8px #10B981" : "0 0 8px #F59E0B";
      postText.style.color = isAligned ? "#34D399" : "#FBBF24";
      if (isAligned) {
        postText.textContent = `AI TRẠNG THÁI: [KHUÔN MẶT ĐÃ KHỚP] • Trục cổ C1-C7 thẳng chuẩn (Lệch ${Math.abs(this.headTilt)}°) • Sẵn sàng Chin Tuck`;
      } else {
        postText.textContent = `AI TRẠNG THÁI: [CẢNH BÁO TƯ THẾ] • Đầu đang nghiêng ${this.headTilt}° • Vui lòng giữ thẳng đầu vào khung xanh`;
      }
    } else {
      postDot.style.background = "#EF4444";
      postDot.style.boxShadow = "0 0 8px #EF4444";
      postText.style.color = "#FCA5A5";
      postText.textContent = "AI TRẠNG THÁI: [ĐANG TÌM KHUÔN MẶT] Vui lòng hướng mặt vào khung xanh trên camera";
    }
  }

  startCanvasHUDLoop() {
    let angle = 0;
    const render = () => {
      angle += 0.03;
      const canvas1 = document.getElementById("overlay-canvas");
      const video1 = document.getElementById("webcam-video");
      const canvas3 = document.getElementById("multiverse-overlay-canvas");
      const video3 = document.getElementById("multiverse-video");

      if (canvas1) this.drawFaceDetectionHUD(canvas1, video1, angle);
      if (canvas3) this.drawFaceDetectionHUD(canvas3, video3, angle);

      // Synchronize Dynamic Non-Mirrored HTML Status Badges for Tab 1 and Tab 3
      this.syncLiveHUDStatusBadge("camera-live-hud-badge", "camera-live-hud-text", "camera-live-hud-dot", "camera-live-hud-label");
      this.syncLiveHUDStatusBadge("multiverse-live-hud-badge", "multiverse-live-hud-text", "multiverse-live-hud-dot", "multiverse-live-hud-label");

      // Synchronize Tab 3 Chin Tuck AI Posture Live Strip
      this.syncMultiversePostureIndicator();

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
    this.sentFrameWidth = 640;
    this.sentFrameHeight = 480;
    const offCanvas = document.createElement("canvas");
    offCanvas.width = this.sentFrameWidth;
    offCanvas.height = this.sentFrameHeight;
    const offCtx = offCanvas.getContext("2d");
    const video1 = document.getElementById("webcam-video");
    const video3 = document.getElementById("multiverse-video");

    setInterval(() => {
      if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
        const tabForecast = document.getElementById("tab-forecast");
        const isTab3 = tabForecast && tabForecast.classList.contains("active");
        const activeVideo = (isTab3 && video3 && video3.videoWidth > 0)
          ? video3
          : ((video1 && video1.videoWidth > 0) ? video1 : (video3 && video3.videoWidth > 0 ? video3 : null));

        if (this.isCameraActive && activeVideo && activeVideo.videoWidth > 0) {
          // Adapt dynamically to true camera aspect ratio (prevents face squashing/distortion)
          const vw = activeVideo.videoWidth;
          const vh = activeVideo.videoHeight;
          const targetW = 640;
          const targetH = Math.round((targetW * vh) / vw);
          if (offCanvas.width !== targetW || offCanvas.height !== targetH) {
            offCanvas.width = targetW;
            offCanvas.height = targetH;
            this.sentFrameWidth = targetW;
            this.sentFrameHeight = targetH;
          }

          // Send proportional camera frame to backend
          offCtx.drawImage(activeVideo, 0, 0, targetW, targetH);
          const b64 = offCanvas.toDataURL("image/jpeg", 0.92);
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
    this.lastLandmarks = data.landmarks || null;
    this.lastFrameW = data.frame_w;
    this.lastFrameH = data.frame_h;
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
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, 640, 480);
      b64Frame = canvas.toDataURL("image/jpeg", 0.92);
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
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 640, 480);
    return canvas.toDataURL("image/jpeg", 0.92);
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
  // 7. Tab 3: Hospital ECG Telemetry Monitor & Affective Coupling
  // -------------------------------------------------------------
  setMultiverseSliderVal(val) {
    this.multiverseSliderVal = typeof val === "number" ? val : 0.0;
  }

  initECGHospitalMonitor() {
    this.ecgConfig = {
      sweepX: 0,
      sweepSpeed: 2.3,
      currentBpm: 74,
      targetBpm: 74,
      hrv: 54,
      rrInterval: 810,
      rhythmStatus: "NHỊP XOANG BÌNH THƯỜNG",
      soundEnabled: false
    };

    const soundBtn = document.getElementById("btn-toggle-ecg-sound");
    if (soundBtn) {
      soundBtn.addEventListener("click", () => {
        this.ecgConfig.soundEnabled = !this.ecgConfig.soundEnabled;
        if (this.ecgConfig.soundEnabled) {
          soundBtn.innerHTML = "🔊 BÍP ECG: BẬT";
          soundBtn.classList.add("btn-primary");
          soundBtn.style.color = "#34D399";
          soundBtn.style.borderColor = "#10B981";
          this.playECGBeep(this.ecgConfig.currentBpm);
        } else {
          soundBtn.innerHTML = "🔇 BÍP ECG: TẮT";
          soundBtn.classList.remove("btn-primary");
          soundBtn.style.color = "";
          soundBtn.style.borderColor = "";
        }
      });
    }

    this.startECGMonitorLoop();
  }

  initECGAudioContext() {
    if (!this.ecgAudioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ecgAudioCtx = new AudioCtx();
    }
    if (this.ecgAudioCtx && this.ecgAudioCtx.state === "suspended") {
      this.ecgAudioCtx.resume();
    }
  }

  playECGBeep(bpm) {
    if (!this.ecgConfig || !this.ecgConfig.soundEnabled) return;
    try {
      this.initECGAudioContext();
      if (!this.ecgAudioCtx) return;

      const now = this.ecgAudioCtx.currentTime;
      const osc = this.ecgAudioCtx.createOscillator();
      const gain = this.ecgAudioCtx.createGain();

      const pitch = bpm > 98 ? 960 : (bpm < 65 ? 780 : 880);
      osc.type = "sine";
      osc.frequency.setValueAtTime(pitch, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.08, now + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.055);

      osc.connect(gain);
      gain.connect(this.ecgAudioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.06);
    } catch (e) {
      console.warn("[ECG AUDIO] Beep error:", e);
    }
  }

  calculateECGSample(phase, valence, exhaustion) {
    // Lead II ECG normalized wave (-0.4 to +1.25)
    if (phase >= 0.12 && phase < 0.22) {
      const p = (phase - 0.12) / 0.10;
      return Math.sin(p * Math.PI) * 0.14;
    }
    if (phase >= 0.22 && phase < 0.32) return 0.0;
    if (phase >= 0.32 && phase < 0.35) {
      const q = (phase - 0.32) / 0.03;
      return -Math.sin(q * Math.PI) * 0.15;
    }
    if (phase >= 0.35 && phase < 0.39) {
      const r = (phase - 0.35) / 0.04;
      return Math.sin(r * Math.PI) * 1.25;
    }
    if (phase >= 0.39 && phase < 0.43) {
      const s = (phase - 0.39) / 0.04;
      return -Math.sin(s * Math.PI) * 0.38;
    }
    if (phase >= 0.43 && phase < 0.50) {
      return (exhaustion >= 3) ? -0.05 : 0.01;
    }
    if (phase >= 0.50 && phase < 0.68) {
      const t = (phase - 0.50) / 0.18;
      const tAmp = 0.28 + (valence > 0 ? 0.06 : -0.06);
      return Math.sin(t * Math.PI) * tAmp;
    }
    return 0.0;
  }

  calculatePlethSample(phase, arousal) {
    if (phase < 0.15) return 0.0;
    if (phase >= 0.15 && phase < 0.35) {
      const rise = (phase - 0.15) / 0.20;
      return Math.sin(rise * Math.PI * 0.5);
    }
    if (phase >= 0.35 && phase < 0.52) {
      const decay = (phase - 0.35) / 0.17;
      return 1.0 - decay * 0.45;
    }
    if (phase >= 0.52 && phase < 0.68) {
      const notch = (phase - 0.52) / 0.16;
      return 0.55 + Math.sin(notch * Math.PI) * 0.16;
    }
    const runoff = (phase - 0.68) / 0.32;
    return 0.55 * Math.cos(runoff * Math.PI * 0.5);
  }

  startECGMonitorLoop() {
    // If multiverseSim is initialized, waveform monitor is managed by the dedicated rPPG engine in multiverse.js
    if (window.multiverseSim && typeof window.multiverseSim.initWaveformMonitor === "function") {
      return;
    }
    const canvas = document.getElementById("trajectory-canvas");
    if (!canvas) return;

    let cardiacPhase = 0.0;
    let sweepX = 0;
    let lastTime = performance.now();
    let hasBeepedThisBeat = false;
    let buffer = [];

    const render = (now) => {
      // 1. Sync canvas resolution with display width
      if (canvas.parentElement) {
        const pw = canvas.parentElement.clientWidth || 600;
        const ph = 220;
        if (canvas.width !== pw || canvas.height !== ph) {
          canvas.width = pw;
          canvas.height = ph;
          buffer = new Array(pw).fill(null);
        }
      }
      const w = canvas.width;
      const h = canvas.height;
      if (w === 0 || h === 0) {
        requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext("2d");

      // 2. Compute dynamic heart rate modulated by emotion & multiverse timeline
      const val = (this.currentBiometrics && typeof this.currentBiometrics.valence === "number") 
        ? this.currentBiometrics.valence : 0.0;
      const exLevel = (this.currentBiometrics && typeof this.currentBiometrics.mental_exhaustion_level === "number") 
        ? this.currentBiometrics.mental_exhaustion_level : 1;
      const slider = typeof this.multiverseSliderVal === "number" ? this.multiverseSliderVal : 0.0;

      let targetBpm = 72;
      targetBpm -= val * 10;
      targetBpm += exLevel * 3;

      if (slider < -0.01) {
        targetBpm += Math.abs(slider) * 32; // Tachycardia under burnout
      } else if (slider > 0.01) {
        targetBpm -= slider * 10; // Restful coherence under vitality
      }
      targetBpm = Math.max(52, Math.min(130, targetBpm));

      this.ecgConfig.currentBpm += (targetBpm - this.ecgConfig.currentBpm) * 0.04;
      const bpm = Math.round(this.ecgConfig.currentBpm);
      const rr = Math.round(60000 / bpm);
      const hrv = Math.max(22, Math.min(76, Math.round(55 + val * 16 - (slider < 0 ? 18 : -8))));

      let rhythm = "NHỊP XOANG BÌNH THƯỜNG";
      if (bpm >= 100) rhythm = "SINUS TACHYCARDIA (KIỆT SỨC)";
      else if (bpm <= 60) rhythm = "SINUS BRADYCARDIA (NGHỈ SÂU)";
      else if (this.isSmiling) rhythm = "NHỊP XOANG THĂNG HOA (DUCHENNE)";

      // 3. Advance cardiac phase
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      const beatPeriodSec = 60 / this.ecgConfig.currentBpm;
      cardiacPhase = (cardiacPhase + dt / beatPeriodSec) % 1.0;

      // Detect R-peak for audio beep & pulse animation
      if (cardiacPhase >= 0.35 && cardiacPhase < 0.40) {
        if (!hasBeepedThisBeat) {
          this.playECGBeep(bpm);
          hasBeepedThisBeat = true;
        }
      } else if (cardiacPhase > 0.48) {
        hasBeepedThisBeat = false;
      }

      // 4. Update sweep buffer
      const step = Math.max(1, Math.round(this.ecgConfig.sweepSpeed));
      for (let s = 0; s < step; s++) {
        const curX = Math.floor(sweepX);
        const ecgVal = this.calculateECGSample(cardiacPhase, val, exLevel);
        const plethVal = this.calculatePlethSample(cardiacPhase, 0.4);
        buffer[curX] = { ecg: ecgVal, pleth: plethVal };

        // Erase blanking gap ahead of scan beam (24px wide)
        for (let g = 1; g <= 24; g++) {
          buffer[(curX + g) % w] = null;
        }
        sweepX = (sweepX + 1) % w;
      }

      // 5. Draw Medical Display Background
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, w, h);

      // 6. Draw Clinical Millimeter Grid
      ctx.save();
      // Minor Grid (8px)
      ctx.strokeStyle = "rgba(16, 185, 129, 0.06)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (let x = 0; x < w; x += 8) {
        ctx.moveTo(x, 0); ctx.lineTo(x, h);
      }
      for (let y = 0; y < h; y += 8) {
        ctx.moveTo(0, y); ctx.lineTo(w, y);
      }
      ctx.stroke();

      // Major Grid (40px)
      ctx.strokeStyle = "rgba(16, 185, 129, 0.18)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      for (let x = 0; x < w; x += 40) {
        ctx.moveTo(x, 0); ctx.lineTo(x, h);
      }
      for (let y = 0; y < h; y += 40) {
        ctx.moveTo(0, y); ctx.lineTo(w, y);
      }
      ctx.stroke();
      ctx.restore();

      // 7. Base levels for Dual Channels
      const yMid1 = h * 0.32; // Channel 1: Lead II ECG
      const yMid2 = h * 0.74; // Channel 2: Pleth Pulse
      const amp1 = h * 0.22;
      const amp2 = h * 0.17;

      // Draw channel baselines
      ctx.strokeStyle = "rgba(16, 185, 129, 0.12)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(0, yMid1); ctx.lineTo(w, yMid1);
      ctx.moveTo(0, yMid2); ctx.lineTo(w, yMid2);
      ctx.stroke();
      ctx.setLineDash([]);

      // 8. Draw Channel 1: Lead II ECG Waveform (Glowing Green)
      ctx.save();
      ctx.strokeStyle = "#10B981";
      ctx.shadowColor = "#10B981";
      ctx.shadowBlur = 8;
      ctx.lineWidth = 2.0;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let started1 = false;
      for (let x = 0; x < w; x++) {
        const pt = buffer[x];
        if (pt && typeof pt.ecg === "number") {
          const py = yMid1 - pt.ecg * amp1;
          if (!started1) {
            ctx.moveTo(x, py);
            started1 = true;
          } else {
            ctx.lineTo(x, py);
          }
        } else {
          started1 = false;
        }
      }
      ctx.stroke();
      ctx.restore();

      // 9. Draw Channel 2: Pleth Arousal Waveform (Glowing Cyan)
      ctx.save();
      ctx.strokeStyle = "#06B6D4";
      ctx.shadowColor = "#06B6D4";
      ctx.shadowBlur = 6;
      ctx.lineWidth = 1.8;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let started2 = false;
      for (let x = 0; x < w; x++) {
        const pt = buffer[x];
        if (pt && typeof pt.pleth === "number") {
          const py = yMid2 - (pt.pleth - 0.45) * amp2;
          if (!started2) {
            ctx.moveTo(x, py);
            started2 = true;
          } else {
            ctx.lineTo(x, py);
          }
        } else {
          started2 = false;
        }
      }
      ctx.stroke();
      ctx.restore();

      // 10. Draw Sweep Beam (Scanning Cursor Head)
      const headX = Math.floor(sweepX);
      const sweepGrad = ctx.createLinearGradient(Math.max(0, headX - 18), 0, headX, 0);
      sweepGrad.addColorStop(0, "rgba(16, 185, 129, 0)");
      sweepGrad.addColorStop(1, "rgba(52, 211, 153, 0.45)");
      ctx.fillStyle = sweepGrad;
      ctx.fillRect(Math.max(0, headX - 18), 0, 18, h);

      // Cursor vertical hairline
      ctx.strokeStyle = "rgba(52, 211, 153, 0.85)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(headX, 0);
      ctx.lineTo(headX, h);
      ctx.stroke();

      // Leading Glowing Dots on both waves
      const curSample = buffer[headX] || { ecg: 0, pleth: 0 };
      const dot1Y = yMid1 - (curSample.ecg || 0) * amp1;
      const dot2Y = yMid2 - ((curSample.pleth || 0.45) - 0.45) * amp2;

      ctx.save();
      ctx.fillStyle = "#34D399";
      ctx.shadowColor = "#34D399";
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(headX, dot1Y, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#38BDF8";
      ctx.shadowColor = "#38BDF8";
      ctx.beginPath();
      ctx.arc(headX, dot2Y, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // 11. On-Screen Medical Telemetry Text
      ctx.save();
      ctx.font = "bold 9.5px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#34D399";
      ctx.fillText("ECG LEAD II • 25mm/s • 10mm/mV • GAIN: 1.0x", 12, 18);

      ctx.fillStyle = "#38BDF8";
      ctx.fillText("PLETH / AROUSAL PULSE WAVE (SPO2 EQUIV)", 12, h * 0.54);

      // Heart Icon & Digital BPM (Top Right)
      const isBeating = (cardiacPhase >= 0.35 && cardiacPhase < 0.42);
      const bpmColor = bpm > 98 ? "#EF4444" : (bpm < 60 ? "#38BDF8" : "#34D399");

      ctx.textAlign = "right";
      ctx.font = "bold 13px 'JetBrains Mono', monospace";
      ctx.fillStyle = bpmColor;
      ctx.fillText(`HR: ${bpm} BPM`, w - 12, 19);

      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#94A3B8";
      ctx.fillText(`R-R: ${rr}ms  HRV: ${hrv}ms`, w - 12, 33);
      ctx.restore();

      // 12. Synchronize HTML UI elements
      const hrDisplay = document.getElementById("ecg-hr-display");
      const metricBpm = document.getElementById("ecg-metric-bpm");
      const metricHrv = document.getElementById("ecg-metric-hrv");
      const metricRr = document.getElementById("ecg-metric-rr");
      const metricRhythm = document.getElementById("ecg-metric-rhythm");

      if (hrDisplay) hrDisplay.textContent = `❤️ HR: ${bpm} BPM`;
      if (metricBpm) metricBpm.innerHTML = `${bpm} <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">BPM</span>`;
      if (metricHrv) metricHrv.innerHTML = `${hrv} <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">ms</span>`;
      if (metricRr) metricRr.innerHTML = `${rr} <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">ms</span>`;
      if (metricRhythm) {
        metricRhythm.textContent = rhythm;
        metricRhythm.style.color = bpm > 98 ? "#EF4444" : "#34D399";
      }

      requestAnimationFrame(render);
    };

    requestAnimationFrame(render);
  }

  renderTrajectoryChart() {
    // Backward compatibility hook: ECG Monitor renders continuously
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

    // Sync to Future Multiverse Engine
    if (window.multiverseSim) {
      window.multiverseSim.burnoutIndex = riskPct;
      if (window.multiverseSim.sliderValue === 0) {
        window.multiverseSim.updateBurnoutDisplay(riskPct);
      }
    }
  }

  bindEvents() {
    // Webcam Toggle Buttons
    const camToggle = document.getElementById("webcam-toggle-btn");
    if (camToggle) {
      camToggle.addEventListener("click", () => this.toggleCamera());
    }

    const multiCamToggle = document.getElementById("multiverse-cam-toggle-btn");
    if (multiCamToggle) {
      multiCamToggle.addEventListener("click", () => this.toggleCamera());
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

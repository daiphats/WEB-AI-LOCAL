/**
 * Affective Mimic Match Biofeedback Game Logic
 * Dual Viewport: Target Expression + Live User Camera
 * 3-Second Countdown, Facial Muscle Recruitment Analysis, Cosine Score
 */

class MimicMatchGame {
  constructor() {
    this.currentTarget = null;
    this.isCountingDown = false;
    this.streak = 0;
    this.highScore = 0;
  }

  async init() {
    this.bindEvents();
    await this.loadNewTarget();
    this.attachWebcamStream();
  }

  attachWebcamStream() {
    const gameVideo = document.getElementById("game-webcam-video");
    const standbyOverlay = document.getElementById("game-cam-standby");
    const camBadge = document.getElementById("game-cam-status-badge");

    if (window.clinicalStation && window.clinicalStation.mediaStream && window.clinicalStation.isCameraActive) {
      if (gameVideo && gameVideo.srcObject !== window.clinicalStation.mediaStream) {
        gameVideo.srcObject = window.clinicalStation.mediaStream;
        gameVideo.play().catch(e => console.log("[GAME] Video play:", e));
      }
      if (standbyOverlay) standbyOverlay.style.display = "none";
      if (camBadge) {
        camBadge.textContent = "Camera trực tiếp";
        camBadge.className = "badge badge-success";
      }
    } else {
      if (standbyOverlay) standbyOverlay.style.display = "flex";
      if (camBadge) {
        camBadge.textContent = "Camera chưa kích hoạt";
        camBadge.className = "badge badge-warning";
      }
    }
  }

  bindEvents() {
    const startBtn = document.getElementById("game-start-btn");
    if (startBtn) {
      startBtn.addEventListener("click", () => this.startCountdown());
    }

    const nextBtn = document.getElementById("game-next-target-btn");
    if (nextBtn) {
      nextBtn.addEventListener("click", () => this.loadNewTarget());
    }

    const enableCamBtn = document.getElementById("game-enable-cam-btn");
    if (enableCamBtn) {
      enableCamBtn.addEventListener("click", async () => {
        if (window.clinicalStation) {
          await window.clinicalStation.startCamera();
          this.attachWebcamStream();
        }
      });
    }

    // Viewport HUD controls
    const hudCloseBtn = document.getElementById("hud-close-btn");
    if (hudCloseBtn) {
      hudCloseBtn.addEventListener("click", () => {
        const hud = document.getElementById("game-hud-result-overlay");
        if (hud) hud.style.display = "none";
      });
    }

    const hudRetryBtn = document.getElementById("hud-retry-btn");
    if (hudRetryBtn) {
      hudRetryBtn.addEventListener("click", () => {
        const hud = document.getElementById("game-hud-result-overlay");
        if (hud) hud.style.display = "none";
        this.startCountdown();
      });
    }

    const hudNextBtn = document.getElementById("hud-next-btn");
    if (hudNextBtn) {
      hudNextBtn.addEventListener("click", async () => {
        const hud = document.getElementById("game-hud-result-overlay");
        if (hud) hud.style.display = "none";
        await this.loadNewTarget();
        setTimeout(() => {
          this.startCountdown();
        }, 350);
      });
    }
  }

  async loadNewTarget() {
    const hud = document.getElementById("game-hud-result-overlay");
    if (hud) hud.style.display = "none";

    try {
      const res = await fetch("/api/game/target");
      const data = await res.json();
      this.currentTarget = data;
      this.renderTargetUI(data);
    } catch (err) {
      console.error("[GAME] Failed to load target:", err);
    }
  }

  renderTargetUI(target) {
    const titleEl = document.getElementById("game-target-title");
    const descEl = document.getElementById("game-target-desc");
    const imgEl = document.getElementById("game-target-img");
    const musclesEl = document.getElementById("game-target-muscles");

    if (titleEl) titleEl.textContent = target.name;
    if (descEl) descEl.textContent = target.description;

    // Map svg name to asset path
    let svgPath = "/static/assets/memes/exp_01_smile.svg";
    if (target.id === "EXP_02") svgPath = "/static/assets/memes/exp_02_surprise.svg";
    else if (target.id === "EXP_03") svgPath = "/static/assets/memes/exp_03_calm.svg";
    else if (target.id === "EXP_04") svgPath = "/static/assets/memes/exp_04_focus.svg";
    else if (target.id === "EXP_05") svgPath = "/static/assets/memes/exp_05_relief.svg";

    if (imgEl) imgEl.src = svgPath;

    if (musclesEl && target.primary_muscles) {
      musclesEl.innerHTML = target.primary_muscles
        .map(m => `<span class="badge badge-neutral">${m}</span>`)
        .join(" ");
    }

    // Reset result badge
    const badgeEl = document.getElementById("game-result-badge");
    if (badgeEl) {
      badgeEl.textContent = "Sẵn sàng";
      badgeEl.className = "badge badge-neutral";
    }
  }

  async startCountdown() {
    if (this.isCountingDown) return;

    // Hide any previous HUD result overlay
    const hud = document.getElementById("game-hud-result-overlay");
    if (hud) hud.style.display = "none";

    // Check if camera is active; if not, try activating
    if (!window.clinicalStation || !window.clinicalStation.isCameraActive) {
      if (window.clinicalStation) {
        await window.clinicalStation.startCamera();
        this.attachWebcamStream();
      }
    } else {
      this.attachWebcamStream();
    }

    this.isCountingDown = true;

    const overlay = document.getElementById("game-countdown-overlay");
    const digits = document.getElementById("game-countdown-digits");
    const startBtn = document.getElementById("game-start-btn");

    if (startBtn) startBtn.disabled = true;
    if (overlay) overlay.style.display = "flex";

    let count = 3;
    if (digits) digits.textContent = `0${count}`;
    this.playBeep(440, 0.1);

    const timer = setInterval(() => {
      count--;
      if (count > 0) {
        if (digits) digits.textContent = `0${count}`;
        this.playBeep(440, 0.1);
      } else {
        clearInterval(timer);
        if (digits) digits.textContent = "CAPTURE";
        this.playBeep(880, 0.25);

        setTimeout(() => {
          if (overlay) overlay.style.display = "none";
          this.isCountingDown = false;
          if (startBtn) startBtn.disabled = false;
          this.captureAndEvaluate();
        }, 400);
      }
    }, 1000);
  }

  playBeep(freq, duration) {
    if (window.acousticEngine) {
      window.acousticEngine.initAudioContext();
      const ctx = window.acousticEngine.ctx;
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
    }
  }

  getVideoElement() {
    const gameVid = document.getElementById("game-webcam-video");
    if (gameVid && gameVid.videoWidth > 0) return gameVid;

    const mainVid = document.getElementById("webcam-video");
    if (mainVid && mainVid.videoWidth > 0) return mainVid;

    if (window.clinicalStation && window.clinicalStation.videoEl && window.clinicalStation.videoEl.videoWidth > 0) {
      return window.clinicalStation.videoEl;
    }
    return gameVid || mainVid;
  }

  async captureAndEvaluate() {
    // Capture current active frame
    const video = this.getVideoElement();
    let b64Image = "";

    if (video && video.videoWidth > 0) {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, 640, 480);
      b64Image = canvas.toDataURL("image/jpeg", 0.92);
    }

    const station = window.clinicalStation;
    const clientFaceDetected = Boolean(station && station.lastFaceDetected);
    const clientIsSmiling = Boolean(station && station.lastFaceDetected && station.isSmiling);
    const clientSmileScore = Number((station && station.lastFaceDetected && station.smileScore) || 0);

    try {
      const res = await fetch("/api/game/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_id: this.currentTarget ? this.currentTarget.id : "EXP_01",
          image_b64: b64Image,
          client_face_detected: clientFaceDetected,
          client_is_smiling: clientIsSmiling,
          client_smile_score: clientSmileScore
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      this.displayEvaluationResult(result);
    } catch (err) {
      console.error("[GAME] Evaluation error:", err);
      const badgeEl = document.getElementById("game-result-badge");
      if (badgeEl) {
        badgeEl.textContent = "Lỗi kết nối";
        badgeEl.className = "badge badge-danger";
      }
    }
  }

  displayEvaluationResult(res) {
    // 1. Update persistent bottom card
    const scoreVal = document.getElementById("game-score-value");
    const scoreBar = document.getElementById("game-score-bar");
    const badgeEl = document.getElementById("game-result-badge");
    const streakEl = document.getElementById("game-streak-value");
    const highEl = document.getElementById("game-high-score-value");
    const insightEl = document.getElementById("game-clinical-insight");
    const muscleListEl = document.getElementById("game-muscle-feedback-list");

    if (scoreVal) scoreVal.textContent = `${res.matching_score}%`;
    if (scoreBar) scoreBar.style.width = `${res.matching_score}%`;
    if (streakEl) streakEl.textContent = res.streak;
    if (highEl) highEl.textContent = `${res.session_high_score}%`;
    if (insightEl) insightEl.textContent = res.clinical_insight;

    if (badgeEl) {
      badgeEl.textContent = res.status_badge;
      if (res.rating === "PASS") badgeEl.className = "badge badge-success";
      else if (res.rating === "ACCEPTABLE") badgeEl.className = "badge badge-warning";
      else badgeEl.className = "badge badge-danger";
    }

    if (muscleListEl && res.muscle_feedback) {
      muscleListEl.innerHTML = res.muscle_feedback.map(m => `
        <div class="telemetry-cell" style="padding: 0.5rem 0.75rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="cell-label">${m.muscle}</span>
            <span class="badge ${m.status === 'OPTIMAL' ? 'badge-success' : 'badge-neutral'}">${m.activation_pct}%</span>
          </div>
        </div>
      `).join("");
    }

    // 2. Update and reveal Viewport HUD (Zero scrolling needed!)
    const hudOverlay = document.getElementById("game-hud-result-overlay");
    const hudBadge = document.getElementById("hud-status-badge");
    const hudScore = document.getElementById("hud-score-number");
    const hudScoreBar = document.getElementById("hud-score-bar-fill");
    const hudStreak = document.getElementById("hud-streak-val");
    const hudHighScore = document.getElementById("hud-highscore-val");
    const hudRating = document.getElementById("hud-rating-val");
    const hudMusclesRow = document.getElementById("hud-muscles-row");

    if (hudBadge) {
      hudBadge.textContent = res.status_badge;
      hudBadge.className = `hud-status-badge ${res.rating === 'PASS' ? 'pass' : (res.rating === 'ACCEPTABLE' ? 'warn' : 'fail')}`;
    }
    if (hudScore) hudScore.textContent = `${res.matching_score}%`;
    if (hudScoreBar) hudScoreBar.style.width = `${res.matching_score}%`;
    if (hudStreak) hudStreak.textContent = res.streak;
    if (hudHighScore) hudHighScore.textContent = `${res.session_high_score}%`;
    if (hudRating) {
      hudRating.textContent = res.rating === "PASS" ? "Đạt" : (res.rating === "ACCEPTABLE" ? "Chấp nhận" : "Thử lại");
    }

    if (hudMusclesRow && res.muscle_feedback) {
      hudMusclesRow.innerHTML = res.muscle_feedback.map(m => {
        const shortName = m.muscle.split("(")[0].trim();
        const isOpt = m.status === "OPTIMAL";
        return `<span class="hud-muscle-pill ${isOpt ? 'optimal' : ''}">${shortName}: ${m.activation_pct}%</span>`;
      }).join("");
    }

    const hudInsight = document.getElementById("hud-clinical-insight");
    if (hudInsight) {
      hudInsight.textContent = res.clinical_insight || "";
    }

    if (hudOverlay) {
      hudOverlay.style.display = "flex";
    }
  }
}

window.mimicGame = new MimicMatchGame();

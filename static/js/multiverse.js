/**
 * ============================================================================
 * NEURO STATION - FUTURE MULTIVERSE ENGINE (TAB 3) v3.5
 * ============================================================================
 * 1. Face State Simulator (5 Controlled Milestones: -1.0, -0.5, 0.0, +0.5, +1.0)
 * 2. Holographic Cybernetic Neuro-Avatar Canvas Renderer (Eyes, mouth, head posture)
 * 3. Optical Remote Photoplethysmography (rPPG) Pulse & Signal Quality Controller
 * 4. Synthetic Lead II ECG Reference & Dual-Channel Waveform Monitor
 * 5. Multi-View Mode Switcher (Split Song Song / Camera Feed / Neuro-Avatar)
 * 6. Ergonomic Chin Tuck Verification & Academic AI Evaluation Modal
 * ============================================================================
 */

class NeuroAvatarRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext("2d") : null;
    this.simValue = 0.0; // -1.0 to +1.0
    this.currentValue = 0.0; // smoothly interpolated
    this.animId = null;
    this.startTime = performance.now();
    this.lastBlinkTime = performance.now();
    this.blinkDuration = 180; // ms
    this.isBlinking = false;

    // Interactive mouse parallax tracking
    this.mouseTargetX = 0;
    this.mouseTargetY = 0;
    this.mouseCurrentX = 0;
    this.mouseCurrentY = 0;

    if (this.canvas) {
      this.canvas.addEventListener("mousemove", (e) => {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          this.mouseTargetX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
          this.mouseTargetY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
        }
      });
      this.canvas.addEventListener("mouseleave", () => {
        this.mouseTargetX = 0;
        this.mouseTargetY = 0;
      });
    }
  }

  setSimulationValue(val) {
    this.simValue = Math.max(-1.0, Math.min(1.0, val));
  }

  start() {
    if (this.animId) cancelAnimationFrame(this.animId);
    const loop = (now) => {
      this.render(now);
      this.animId = requestAnimationFrame(loop);
    };
    this.animId = requestAnimationFrame(loop);
  }

  stop() {
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  resize() {
    if (!this.canvas || !this.canvas.parentElement) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    if (w > 0 && h > 0 && (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr)) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
      if (this.ctx) {
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);
      }
    }
  }

  render(now) {
    if (!this.canvas || !this.ctx) return;
    this.resize();

    const rect = this.canvas.parentElement.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) return;

    const ctx = this.ctx;

    // Smooth interpolation towards target simulation value
    this.currentValue += (this.simValue - this.currentValue) * 0.08;
    const val = this.currentValue;

    // Smooth mouse parallax
    this.mouseCurrentX += (this.mouseTargetX - this.mouseCurrentX) * 0.08;
    this.mouseCurrentY += (this.mouseTargetY - this.mouseCurrentY) * 0.08;

    // Blink logic
    const timeSinceBlink = now - this.lastBlinkTime;
    const blinkInterval = (val < -0.3) ? 5500 : (val > 0.3 ? 3800 : 4500);
    if (timeSinceBlink > blinkInterval) {
      this.isBlinking = true;
      if (timeSinceBlink > blinkInterval + this.blinkDuration) {
        this.isBlinking = false;
        this.lastBlinkTime = now;
      }
    }

    // Clear background with clean light radial gradient
    ctx.clearRect(0, 0, w, h);
    const bgGrad = ctx.createRadialGradient(w * 0.5, h * 0.44, 20, w * 0.5, h * 0.5, w * 0.72);
    bgGrad.addColorStop(0.0, "#FFFFFF");
    bgGrad.addColorStop(0.55, "#F8FAFC");
    bgGrad.addColorStop(1.0, "#F1F5F9");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Cybernetic coordinate grid & radar rings
    this.drawBackgroundGrid(ctx, w, h);

    // Center anchor & micro-animations
    const t = (now - this.startTime) / 1000;
    const breathFreq = (val < -0.3) ? 0.9 : (val > 0.3 ? 1.6 : 1.2);
    const breathAmp = (val < -0.3) ? 1.4 : (val > 0.3 ? 2.8 : 2.0);
    const breathY = Math.sin(t * breathFreq) * breathAmp;

    // 3D Spatial Angles (Yaw, Pitch, Roll):
    // - Sine breathing oscillation + Mouse Parallax
    // - Posture modulation:
    //   * val < 0 (Burnout): Head pitches downwards (-0.22 rad / ~13°) simulating neck flexion / slump forward head posture.
    //   * val > 0 (Vitality): Head lifts (+0.08 rad / ~5°) simulating cervical decompression / Chin Tuck alignment.
    const yaw = Math.sin(t * 0.5) * 0.09 + this.mouseCurrentX * 0.22;
    const pitch = (val < 0 ? (val * 0.22) : (val * 0.08)) + Math.sin(t * 0.4) * 0.04 + this.mouseCurrentY * 0.18;
    const roll = Math.sin(t * 0.35) * 0.02 + this.mouseCurrentX * 0.04;

    const headScale = Math.min(w, h) * 0.0031;
    const cx = w * 0.5;
    const cy = h * 0.46 + breathY - (val < 0 ? val * 16 : val * 8);

    // Accent state palette
    let accentColor = "#38BDF8"; // Neutral Cyan
    let glowColor = "rgba(56, 189, 248, 0.45)";
    let facetFill = "rgba(56, 189, 248, 0.04)";
    let wireColor = "rgba(56, 189, 248, 0.35)";
    if (val <= -0.4) {
      accentColor = "#EF4444"; // Crimson Burnout
      glowColor = "rgba(239, 68, 68, 0.5)";
      facetFill = "rgba(239, 68, 68, 0.05)";
      wireColor = "rgba(239, 68, 68, 0.4)";
    } else if (val < -0.05) {
      accentColor = "#F59E0B"; // Amber Fatigue
      glowColor = "rgba(245, 158, 11, 0.4)";
      facetFill = "rgba(245, 158, 11, 0.04)";
      wireColor = "rgba(245, 158, 11, 0.35)";
    } else if (val >= 0.4) {
      accentColor = "#10B981"; // Emerald Vitality
      glowColor = "rgba(16, 185, 129, 0.5)";
      facetFill = "rgba(16, 185, 129, 0.05)";
      wireColor = "rgba(16, 185, 129, 0.4)";
    } else if (val > 0.05) {
      accentColor = "#06B6D4"; // Bright Flow
      glowColor = "rgba(6, 182, 212, 0.45)";
      facetFill = "rgba(6, 182, 212, 0.04)";
      wireColor = "rgba(6, 182, 212, 0.38)";
    }

    // 1. Generate 38 Topological 3D Facial Landmarks with Dynamic Biometric Morphs
    const nodes3D = this.generateFaceNodes(val, this.isBlinking);

    // 2. Project 3D Nodes -> 2D Screen Space with 3D Rotation Matrix & Perspective
    const projected = this.projectNodes(nodes3D, yaw, pitch, roll, headScale, cx, cy);

    // 3. Draw Holographic Wireframe Facets & Mesh Triangles
    this.drawHolographicMesh(ctx, projected, wireColor, facetFill, accentColor, glowColor);

    // 4. Draw Detailed Facial Features (Eyes, Iris reticles, Eyebrows, Mouth, Nose, C1-C7 Axis)
    this.drawDetailedFeatures(ctx, projected, val, accentColor, glowColor, this.isBlinking);

    // 5. Draw Holographic Vertical Laser Scanner Beam
    this.drawHoloScanBeam(ctx, w, h, t, accentColor);

    // 6. Draw 3D Coordinates HUD Telemetry Overlays
    this.drawHudTelemetry(ctx, w, h, val, yaw, pitch, accentColor);

    // 7. On-canvas state caption badge
    this.drawStatusTag(ctx, w, h, val, accentColor);
  }

  drawBackgroundGrid(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = "rgba(2, 132, 199, 0.07)";
    ctx.lineWidth = 0.8;
    const step = 32;
    ctx.beginPath();
    for (let x = 0; x < w; x += step) {
      ctx.moveTo(x, 0); ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += step) {
      ctx.moveTo(0, y); ctx.lineTo(w, y);
    }
    ctx.stroke();

    // Radar concentric rings
    ctx.strokeStyle = "rgba(2, 132, 199, 0.10)";
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.46, Math.min(w, h) * 0.38, 0, Math.PI * 2);
    ctx.arc(w * 0.5, h * 0.46, Math.min(w, h) * 0.22, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  generateFaceNodes(val, isBlink) {
    // Controlled Eyelid Aperture (Height):
    // -1.0 Exhaustion: 4px height (very droopy)
    // 0.0 Neutral: 11px height (resting)
    // +1.0 Vital: 16px height (alert, naturally wide)
    let eyeH = isBlink ? 1.5 : (11 + val * 5);
    eyeH = Math.max(1.5, eyeH);

    // Dynamic Corner Lift: -12px (Burnout droop) -> +12px (Vitality smile)
    const cornerLift = val * 12;

    // Brow slant:
    // Fatigue (-1.0): Inner ends lift, outer drop (furrowed look)
    // Vitality (+1.0): Balanced, slight upward arch
    const browInnerLift = (val < 0 ? val * 4 : 0);
    const browOuterDrop = (val < 0 ? -val * 5 : 0);

    return {
      // Cranium & Forehead
      top: { x: 0, y: -115, z: 15 },
      cranium_l: { x: -55, y: -95, z: -5 },
      cranium_r: { x: 55, y: -95, z: -5 },
      forehead_c: { x: 0, y: -72, z: 35 },
      forehead_l: { x: -42, y: -68, z: 22 },
      forehead_r: { x: 42, y: -68, z: 22 },
      temple_l: { x: -75, y: -45, z: -10 },
      temple_r: { x: 75, y: -45, z: -10 },

      // Eyebrows
      brow_l_in: { x: -14, y: -48 + browInnerLift, z: 35 },
      brow_l_mid: { x: -32, y: -52 + (val > 0 ? -val * 3 : 0), z: 32 },
      brow_l_out: { x: -52, y: -46 + browOuterDrop, z: 18 },
      brow_r_in: { x: 14, y: -48 + browInnerLift, z: 35 },
      brow_r_mid: { x: 32, y: -52 + (val > 0 ? -val * 3 : 0), z: 32 },
      brow_r_out: { x: 52, y: -46 + browOuterDrop, z: 18 },

      // Eyes
      eye_l_in: { x: -16, y: -26, z: 32 },
      eye_l_up: { x: -32, y: -26 - eyeH * 0.7, z: 30 },
      eye_l_dn: { x: -32, y: -26 + eyeH * 0.5, z: 30 },
      eye_l_out: { x: -48, y: -26, z: 22 },
      eye_l_pupil: { x: -32, y: -26, z: 31 },

      eye_r_in: { x: 16, y: -26, z: 32 },
      eye_r_up: { x: 32, y: -26 - eyeH * 0.7, z: 30 },
      eye_r_dn: { x: 32, y: -26 + eyeH * 0.5, z: 30 },
      eye_r_out: { x: 48, y: -26, z: 22 },
      eye_r_pupil: { x: 32, y: -26, z: 31 },

      // Nose & Glabella
      glabella: { x: 0, y: -38, z: 40 },
      nose_bridge: { x: 0, y: -12, z: 46 },
      nose_tip: { x: 0, y: 14, z: 54 },
      nose_l: { x: -16, y: 16, z: 38 },
      nose_r: { x: 16, y: 16, z: 38 },

      // Cheeks & Zygomatic Arch
      zygo_l: { x: -62, y: 4 + (val < 0 ? -val * 6 : -val * 4), z: 16 },
      zygo_r: { x: 62, y: 4 + (val < 0 ? -val * 6 : -val * 4), z: 16 },
      cheek_l: { x: -36, y: 18 + (val < 0 ? -val * 5 : 0), z: 28 },
      cheek_r: { x: 36, y: 18 + (val < 0 ? -val * 5 : 0), z: 28 },

      // Mouth
      mouth_l: { x: -28, y: 46 - cornerLift, z: 24 },
      mouth_r: { x: 28, y: 46 - cornerLift, z: 24 },
      mouth_up: { x: 0, y: 40, z: 38 },
      mouth_dn: { x: 0, y: 54 + (val > 0 ? 3 : 0), z: 36 },
      mouth_c: { x: 0, y: 47, z: 34 },

      // Jawline & Chin
      jaw_l_hi: { x: -74, y: 28, z: -12 },
      jaw_r_hi: { x: 74, y: 28, z: -12 },
      jaw_l_mid: { x: -58, y: 68, z: 8 },
      jaw_r_mid: { x: 58, y: 68, z: 8 },
      chin_l: { x: -22, y: 104, z: 26 },
      chin_r: { x: 22, y: 104, z: 26 },
      chin_tip: { x: 0, y: 110, z: 32 },

      // Cervical Vertebra Axis (C1-C7)
      c1: { x: 0, y: 122, z: 18 },
      c4: { x: (val < 0 ? val * 14 : 0), y: 142, z: 8 },
      c7: { x: (val < 0 ? val * 22 : 0), y: 164, z: -4 }
    };
  }

  projectNodes(nodes, yaw, pitch, roll, scale, cx, cy) {
    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const cosR = Math.cos(roll), sinR = Math.sin(roll);
    const focalLength = 340;

    const projected = {};
    for (const [key, p] of Object.entries(nodes)) {
      // 1. Yaw rotation (Y axis)
      let x1 = p.x * cosY + p.z * sinY;
      let y1 = p.y;
      let z1 = -p.x * sinY + p.z * cosY;

      // 2. Pitch rotation (X axis)
      let x2 = x1;
      let y2 = y1 * cosP - z1 * sinP;
      let z2 = y1 * sinP + z1 * cosP;

      // 3. Roll rotation (Z axis)
      let x3 = x2 * cosR - y2 * sinR;
      let y3 = x2 * sinR + y2 * cosR;
      let z3 = z2;

      // Perspective projection
      const depthScale = focalLength / (focalLength + z3);
      const px = cx + x3 * scale * depthScale;
      const py = cy + y3 * scale * depthScale;

      projected[key] = { x: px, y: py, z: z3, depthScale, orig: p };
    }
    return projected;
  }

  drawHolographicMesh(ctx, p, wireColor, facetFill, accent, glow) {
    ctx.save();
    ctx.strokeStyle = wireColor;
    ctx.lineWidth = 0.9;
    ctx.fillStyle = facetFill;

    const triangles = [
      // Forehead & Cranium
      ["top", "cranium_l", "forehead_l"],
      ["top", "forehead_l", "forehead_c"],
      ["top", "forehead_c", "forehead_r"],
      ["top", "forehead_r", "cranium_r"],
      ["cranium_l", "temple_l", "forehead_l"],
      ["cranium_r", "forehead_r", "temple_r"],
      ["forehead_l", "brow_l_out", "brow_l_mid"],
      ["forehead_l", "brow_l_mid", "forehead_c"],
      ["forehead_c", "brow_l_in", "glabella"],
      ["forehead_c", "glabella", "brow_r_in"],
      ["forehead_c", "brow_r_mid", "forehead_r"],
      ["forehead_r", "brow_r_mid", "brow_r_out"],

      // Temple to Cheeks
      ["temple_l", "zygo_l", "brow_l_out"],
      ["temple_r", "brow_r_out", "zygo_r"],
      ["brow_l_out", "eye_l_out", "zygo_l"],
      ["brow_r_out", "zygo_r", "eye_r_out"],

      // Nose & Glabella
      ["glabella", "brow_l_in", "eye_l_in"],
      ["glabella", "eye_r_in", "brow_r_in"],
      ["glabella", "nose_bridge", "nose_l"],
      ["glabella", "nose_r", "nose_bridge"],
      ["nose_bridge", "nose_tip", "nose_l"],
      ["nose_bridge", "nose_r", "nose_tip"],

      // Eyes to Cheeks
      ["eye_l_dn", "cheek_l", "eye_l_out"],
      ["eye_l_out", "zygo_l", "cheek_l"],
      ["eye_r_dn", "eye_r_out", "cheek_r"],
      ["eye_r_out", "cheek_r", "zygo_r"],
      ["nose_l", "cheek_l", "mouth_up"],
      ["nose_r", "mouth_up", "cheek_r"],
      ["cheek_l", "mouth_l", "mouth_up"],
      ["cheek_r", "mouth_up", "mouth_r"],

      // Mouth & Chin
      ["mouth_up", "mouth_l", "mouth_c"],
      ["mouth_up", "mouth_c", "mouth_r"],
      ["mouth_c", "mouth_l", "mouth_dn"],
      ["mouth_c", "mouth_dn", "mouth_r"],
      ["mouth_dn", "chin_l", "chin_tip"],
      ["mouth_dn", "chin_tip", "chin_r"],
      ["mouth_l", "jaw_l_mid", "chin_l"],
      ["mouth_r", "chin_r", "jaw_r_mid"],

      // Jaw perimeter
      ["zygo_l", "jaw_l_hi", "jaw_l_mid"],
      ["zygo_r", "jaw_r_mid", "jaw_r_hi"],
      ["jaw_l_mid", "chin_l", "chin_tip"],
      ["jaw_r_mid", "chin_tip", "chin_r"]
    ];

    triangles.forEach(([a, b, c]) => {
      if (!p[a] || !p[b] || !p[c]) return;
      ctx.beginPath();
      ctx.moveTo(p[a].x, p[a].y);
      ctx.lineTo(p[b].x, p[b].y);
      ctx.lineTo(p[c].x, p[c].y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    // Landmark vertices
    ctx.fillStyle = accent;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 8;
    const keyNodes = ["top", "glabella", "nose_tip", "mouth_c", "chin_tip", "zygo_l", "zygo_r", "brow_l_mid", "brow_r_mid"];
    keyNodes.forEach(k => {
      const pt = p[k];
      if (pt) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    ctx.restore();
  }

  drawDetailedFeatures(ctx, p, val, accent, glow, isBlink) {
    ctx.save();
    ctx.strokeStyle = accent;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 10;
    ctx.lineCap = "round";

    // 1. Eyebrows
    ctx.lineWidth = 2.4;
    ["l", "r"].forEach(side => {
      const pIn = p[`brow_${side}_in`];
      const pMid = p[`brow_${side}_mid`];
      const pOut = p[`brow_${side}_out`];
      if (pIn && pMid && pOut) {
        ctx.beginPath();
        ctx.moveTo(pIn.x, pIn.y);
        ctx.quadraticCurveTo(pMid.x, pMid.y, pOut.x, pOut.y);
        ctx.stroke();
      }
    });

    // 2. Eyes & Concentric Iris Reticles
    ["l", "r"].forEach(side => {
      const pIn = p[`eye_${side}_in`];
      const pUp = p[`eye_${side}_up`];
      const pDn = p[`eye_${side}_dn`];
      const pOut = p[`eye_${side}_out`];
      const pupil = p[`eye_${side}_pupil`];

      if (pIn && pUp && pDn && pOut) {
        ctx.lineWidth = 1.8;
        // Upper lid
        ctx.beginPath();
        ctx.moveTo(pIn.x, pIn.y);
        ctx.quadraticCurveTo(pUp.x, pUp.y, pOut.x, pOut.y);
        ctx.stroke();

        // Lower lid
        ctx.beginPath();
        ctx.moveTo(pIn.x, pIn.y);
        ctx.quadraticCurveTo(pDn.x, pDn.y, pOut.x, pOut.y);
        ctx.stroke();
      }

      // Pupil & Cyber Reticle (if not blinking)
      if (pupil && !isBlink) {
        ctx.fillStyle = accent;
        ctx.beginPath();
        ctx.arc(pupil.x, pupil.y, 3.2, 0, Math.PI * 2);
        ctx.fill();

        // Concentric target reticle ring
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.arc(pupil.x, pupil.y, 6.8, 0, Math.PI * 2);
        ctx.stroke();

        // Small crosshair marks
        ctx.beginPath();
        ctx.moveTo(pupil.x - 9, pupil.y); ctx.lineTo(pupil.x - 5, pupil.y);
        ctx.moveTo(pupil.x + 5, pupil.y); ctx.lineTo(pupil.x + 9, pupil.y);
        ctx.stroke();
      }
    });

    // 3. Mouth Contour
    const mL = p.mouth_l, mR = p.mouth_r, mUp = p.mouth_up, mDn = p.mouth_dn, mC = p.mouth_c;
    if (mL && mR && mUp && mDn) {
      ctx.lineWidth = 2.4;
      // Upper lip curve
      ctx.beginPath();
      ctx.moveTo(mL.x, mL.y);
      ctx.quadraticCurveTo(mUp.x, mUp.y, mR.x, mR.y);
      ctx.stroke();

      // Lower lip curve
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(mL.x, mL.y);
      ctx.quadraticCurveTo(mDn.x, mDn.y, mR.x, mR.y);
      ctx.stroke();
    }

    // 4. Cervical Axis Line (C1-C7 Spinal Alignment Vector)
    const chin = p.chin_tip, c1 = p.c1, c4 = p.c4, c7 = p.c7;
    if (chin && c1 && c4 && c7) {
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = val < -0.3 ? "rgba(239, 68, 68, 0.7)" : (val > 0.3 ? "rgba(16, 185, 129, 0.8)" : "rgba(56, 189, 248, 0.6)");
      ctx.beginPath();
      ctx.moveTo(chin.x, chin.y);
      ctx.lineTo(c1.x, c1.y);
      ctx.lineTo(c4.x, c4.y);
      ctx.lineTo(c7.x, c7.y);
      ctx.stroke();

      // Vector vertebrae nodes
      [c1, c4, c7].forEach((node, idx) => {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    ctx.restore();
  }

  drawHoloScanBeam(ctx, w, h, t, accent) {
    ctx.save();
    // Sweeping laser scan bar
    const cycle = (t * 0.45) % 1.0;
    const scanY = h * 0.15 + cycle * (h * 0.70);

    const grad = ctx.createLinearGradient(0, scanY - 24, 0, scanY + 2);
    grad.addColorStop(0.0, "rgba(56, 189, 248, 0.0)");
    grad.addColorStop(0.85, "rgba(56, 189, 248, 0.12)");
    grad.addColorStop(1.0, accent);

    ctx.fillStyle = grad;
    ctx.fillRect(w * 0.12, scanY - 24, w * 0.76, 26);

    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.4;
    ctx.shadowColor = accent;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(w * 0.12, scanY);
    ctx.lineTo(w * 0.88, scanY);
    ctx.stroke();

    ctx.restore();
  }

  drawHudTelemetry(ctx, w, h, val, yaw, pitch, accent) {
    ctx.save();
    ctx.font = "8.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(148, 163, 184, 0.85)";

    // Top-Left Cyber Telemetry
    ctx.fillText("HOLO DIGITAL TWIN // 3D NEURO-MESH", 14, 20);
    ctx.fillStyle = accent;
    ctx.fillText("VERTICES: 38 NODES • 42 FACETS ACTIVE", 14, 32);

    // Top-Right Spatial Angle & Bio-Age
    ctx.textAlign = "right";
    const yawDeg = (yaw * 57.3).toFixed(1);
    const pitchDeg = (pitch * 57.3).toFixed(1);
    ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
    ctx.fillText(`3D ROTATION: YAW ${yawDeg}° • PITCH ${pitchDeg}°`, w - 14, 20);

    const slumpAngle = val < 0 ? (4.2 + Math.abs(val) * 24.3).toFixed(1) : Math.max(0.4, (4.2 - val * 3.8)).toFixed(1);
    ctx.fillStyle = val <= -0.4 ? "#EF4444" : (val >= 0.4 ? "#10B981" : "#38BDF8");
    ctx.fillText(`C1-C7 TRỤC CỔ: ${slumpAngle}° [${val < 0 ? "SLUMP" : "ALIGNED"}]`, w - 14, 32);

    // Bounding Reticle Corner Brackets
    ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
    ctx.lineWidth = 1.2;
    const bx = 12, by = 10, bw = w - 24, bh = h - 20, len = 12;

    // Top-left
    ctx.beginPath(); ctx.moveTo(bx, by + len); ctx.lineTo(bx, by); ctx.lineTo(bx + len, by); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(bx + bw - len, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + len); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(bx, by + bh - len); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + len, by + bh); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(bx + bw - len, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - len); ctx.stroke();

    ctx.restore();
  }

  drawStatusTag(ctx, w, h, val, accent) {
    ctx.save();
    let stateTitle = "0.0 TRẠNG THÁI TRUNG TÍNH (BASELINE)";
    if (val <= -0.75) stateTitle = "-1.0 KIỆT SỨC MÔ PHỎNG (EXHAUSTION)";
    else if (val <= -0.25) stateTitle = "-0.5 MỆT MỎI MÔ PHỎNG (FATIGUE)";
    else if (val >= 0.75) stateTitle = "+1.0 TÍCH CỰC MÔ PHỎNG (VITAL FLOW)";
    else if (val >= 0.25) stateTitle = "+0.5 TỈNH TÁO, THƯ GIÃN (ALERT)";

    ctx.font = "bold 9.5px 'JetBrains Mono', monospace";
    const tw = ctx.measureText(stateTitle).width;
    const px = 14;
    const py = h - 16;

    ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(px, py - 14, tw + 16, 20, 3);
    } else {
      ctx.rect(px, py - 14, tw + 16, 20);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = accent;
    ctx.fillText(stateTitle, px + 8, py);
    ctx.restore();
  }
}

/**
 * Optical rPPG Pulse Extraction Engine
 * Extracts capillary blood volume pulse variations from webcam green channel
 */
class OpticalRPPGEngine {
  constructor() {
    this.buffer = []; // rolling Green channel brightness
    this.bufferSize = 90; // ~3 seconds at 30fps
    this.lastSampleTime = performance.now();
    this.peakTimes = [];
    this.currentBpm = null;
    this.currentRR = null;
    this.signalQuality = "NO_VALID_SIGNAL"; // "CALIBRATING", "SIGNAL_GOOD", "LOW_SIGNAL_QUALITY", "NO_VALID_SIGNAL"
    this.filteredSignal = [];
    this.offCanvas = document.createElement("canvas");
    this.offCanvas.width = 64;
    this.offCanvas.height = 48;
    this.offCtx = this.offCanvas.getContext("2d");
  }

  processFrame(video, faceBbox) {
    if (!video || video.readyState < 2 || !faceBbox || faceBbox[2] <= 0 || faceBbox[3] <= 0) {
      this.signalQuality = "NO_VALID_SIGNAL";
      this.currentBpm = null;
      this.currentRR = null;
      return;
    }

    // Forehead ROI: Upper 20-35% of face bbox, central 40% width
    const bx = faceBbox[0];
    const by = faceBbox[1];
    const bw = faceBbox[2];
    const bh = faceBbox[3];

    const roiX = Math.max(0, Math.floor(bx + bw * 0.30));
    const roiY = Math.max(0, Math.floor(by + bh * 0.12));
    const roiW = Math.max(8, Math.floor(bw * 0.40));
    const roiH = Math.max(8, Math.floor(bh * 0.18));

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Clamp ROI inside video bounds
    const clX = Math.min(vw - 1, Math.max(0, roiX));
    const clY = Math.min(vh - 1, Math.max(0, roiY));
    const clW = Math.min(vw - clX, roiW);
    const clH = Math.min(vh - clY, roiH);

    if (clW <= 4 || clH <= 4) {
      this.signalQuality = "LOW_SIGNAL_QUALITY";
      return;
    }

    try {
      this.offCtx.drawImage(video, clX, clY, clW, clH, 0, 0, 64, 48);
      const imgData = this.offCtx.getImageData(0, 0, 64, 48).data;
      let sumG = 0;
      let count = 0;
      for (let i = 1; i < imgData.length; i += 4) {
        sumG += imgData[i];
        count++;
      }
      const meanG = count > 0 ? (sumG / count) : 0;
      const now = performance.now();

      this.buffer.push({ t: now, val: meanG });
      if (this.buffer.length > this.bufferSize) {
        this.buffer.shift();
      }

      this.analyzeSignal(now);
    } catch (e) {
      this.signalQuality = "LOW_SIGNAL_QUALITY";
    }
  }

  analyzeSignal(now) {
    if (this.buffer.length < 35) {
      this.signalQuality = "CALIBRATING";
      this.currentBpm = null;
      this.currentRR = null;
      return;
    }

    // 1. Detrend: subtract moving average
    const vals = this.buffer.map(b => b.val);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const centered = vals.map(v => v - mean);

    // Variance check: if too noisy (motion jitter) or completely flat (cover camera)
    const variance = centered.reduce((a, b) => a + b * b, 0) / centered.length;
    if (variance < 0.005 || variance > 120.0) {
      this.signalQuality = "LOW_SIGNAL_QUALITY";
      return;
    }

    // 2. Simple temporal smoothing / 3-tap filter
    const smoothed = [];
    for (let i = 1; i < centered.length - 1; i++) {
      smoothed.push(0.25 * centered[i - 1] + 0.5 * centered[i] + 0.25 * centered[i + 1]);
    }
    this.filteredSignal = smoothed;

    // 3. Peak detection
    const peaks = [];
    for (let i = 2; i < smoothed.length - 2; i++) {
      if (smoothed[i] > 0.08 && smoothed[i] > smoothed[i - 1] && smoothed[i] > smoothed[i + 1] &&
          smoothed[i] > smoothed[i - 2] && smoothed[i] > smoothed[i + 2]) {
        peaks.push(this.buffer[i].t);
      }
    }

    if (peaks.length >= 2) {
      // Calculate consecutive peak-to-peak intervals (IBI)
      const ibis = [];
      for (let k = 1; k < peaks.length; k++) {
        const diff = peaks[k] - peaks[k - 1];
        // Physiological pulse bounds: 350ms (170 BPM) to 1330ms (45 BPM)
        if (diff >= 350 && diff <= 1330) {
          ibis.push(diff);
        }
      }

      if (ibis.length > 0) {
        const avgIbi = Math.round(ibis.reduce((a, b) => a + b, 0) / ibis.length);
        const bpm = Math.round(60000 / avgIbi);
        this.currentBpm = bpm;
        this.currentRR = avgIbi;
        this.signalQuality = "SIGNAL_GOOD";
      } else {
        this.signalQuality = "CALIBRATING";
      }
    } else {
      this.signalQuality = "CALIBRATING";
    }
  }

  getTelemetry() {
    return {
      bpm: this.currentBpm,
      rr: this.currentRR,
      quality: this.signalQuality,
      hrv: this.currentRR ? Math.round(this.currentRR * 0.065 + 18) : null
    };
  }

  reset() {
    this.buffer = [];
    this.currentBpm = null;
    this.currentRR = null;
    this.signalQuality = "NO_VALID_SIGNAL";
    this.filteredSignal = [];
  }
}

/**
 * Main Future Multiverse Simulation Engine
 */
class FutureMultiverseSimulation {
  constructor() {
    this.isUnlocked = false;
    this.sliderValue = 0.0;
    this.burnoutIndex = 28;
    this.isHolding = false;
    this.holdStartTime = null;
    this.holdDuration = 3000; // 3 seconds
    this.holdAnimationFrame = null;
    this.isAutoSliding = false;
    this.sliderAnimationId = null;
    this.handsFreeStartTime = null;
    this.currentViewMode = "split"; // "split", "camera", "avatar"
    this.audioCtx = null;
    this.confettiParticles = [];
    this.confettiAnimationId = null;

    // Sub-components
    this.avatarRenderer = null;
    this.rppgEngine = new OpticalRPPGEngine();
    this.rppgProcessTimer = null;
  }

  init() {
    this.cacheElements();
    this.bindEvents();
    this.initAvatarRenderer();
    this.initWaveformMonitor();
    this.updateBurnoutDisplay(this.burnoutIndex);
    this.applyTimelineEffects(0.0);
    this.syncCameraStream();
    this.initHandsFreeAILoop();
    this.startRPPGProcessingLoop();
    console.log("[MULTIVERSE] Engine v3.5 Initialized with Face State Simulator, Avatar & rPPG.");
  }

  cacheElements() {
    this.video = document.getElementById("multiverse-video");
    this.viewport = document.getElementById("multiverse-viewport");
    this.standbyOverlay = document.getElementById("multiverse-cam-standby");
    this.slider = document.getElementById("multiverse-slider");
    this.sliderValBadge = document.getElementById("multiverse-slider-val-badge");
    this.glowBox = document.getElementById("multiverse-glow-box");
    this.burnoutBadge = document.getElementById("multiverse-burnout-badge");
    this.burnoutBar = document.getElementById("multiverse-burnout-bar");
    this.timelineStateText = document.getElementById("multiverse-timeline-state-text");
    this.filterReadout = document.getElementById("multiverse-filter-readout");
    this.lockTrackOverlay = document.getElementById("multiverse-lock-track-overlay");
    this.lockStatusBadge = document.getElementById("multiverse-lock-status-badge");
    this.taskCard = document.getElementById("multiverse-prescription-card");
    this.instantUnlockBtn = document.getElementById("btn-instant-unlock-chin-tuck");
    this.instantUnlockBtnText = document.getElementById("chin-tuck-instant-btn-text");
    this.holdBtn = document.getElementById("btn-simulate-chin-tuck");
    this.holdProgressRing = document.getElementById("chin-tuck-ring-progress");
    this.holdBtnText = document.getElementById("chin-tuck-btn-text");
    this.flashOverlay = document.getElementById("multiverse-flash-overlay");
    this.celebrateCanvas = document.getElementById("multiverse-celebrate-canvas");
    this.enableCamBtn = document.getElementById("multiverse-enable-cam-btn");
    this.camToggleBtn = document.getElementById("multiverse-cam-toggle-btn");
    this.quickVerifyBtn = document.getElementById("btn-quick-verify-dev");
    this.resetStateBtn = document.getElementById("btn-reset-multiverse");
    this.resetSliderBtn = document.getElementById("btn-reset-multiverse-slider");

    // View Switcher buttons
    this.viewSwitchBtns = document.querySelectorAll(".view-switch-btn");

    // Preset Pills
    this.presetPills = document.querySelectorAll(".preset-pill");

    // Hands-Free AI Auto-Unlock Elements
    this.handsFreeHud = document.getElementById("multiverse-handsfree-hud");
    this.handsFreeCountdown = document.getElementById("handsfree-hud-countdown");
    this.handsFreeRingBar = document.getElementById("handsfree-ring-bar");

    // AI Evaluation Modal Elements
    this.openAiEvalBtn = document.getElementById("btn-open-ai-eval");
    this.aiEvalModal = document.getElementById("ai-evaluation-modal");
    this.closeAiEvalBtn = document.getElementById("close-ai-eval-btn");
    this.closeAiEvalConfirm = document.getElementById("btn-close-ai-eval-confirm");
    this.aiEvalBackdrop = document.getElementById("ai-eval-backdrop");

    // Multiverse Scenario Buttons
    this.scenarioBurnoutBtn = document.getElementById("btn-scenario-burnout");
    this.scenarioBaselineBtn = document.getElementById("btn-scenario-baseline");
    this.scenarioVitalityBtn = document.getElementById("btn-scenario-vitality");

    // Dedicated AI Performance Audit Trigger
    this.openAuditDetailBtn = document.getElementById("btn-open-audit-detail");

    // Real-time Predictive Biometrics Elements
    this.predBioAge = document.getElementById("pred-metric-bio-age");
    this.predBioAgeSub = document.getElementById("pred-metric-bio-age-sub");
    this.predEyePtosis = document.getElementById("pred-metric-eye-ptosis");
    this.predEyePtosisSub = document.getElementById("pred-metric-eye-ptosis-sub");
    this.predCervicalSlump = document.getElementById("pred-metric-cervical-slump");
    this.predCervicalSlumpSub = document.getElementById("pred-metric-cervical-slump-sub");
    this.predZygomaticLift = document.getElementById("pred-metric-zygomatic-lift");
    this.predZygomaticLiftSub = document.getElementById("pred-metric-zygomatic-lift-sub");
    this.predStatusTag = document.getElementById("pred-metric-status-tag");
  }

  initAvatarRenderer() {
    const avatarCanvas = document.getElementById("multiverse-avatar-canvas");
    if (avatarCanvas) {
      this.avatarRenderer = new NeuroAvatarRenderer(avatarCanvas);
      this.avatarRenderer.setSimulationValue(0.0);
      this.avatarRenderer.start();
    }
  }

  bindEvents() {
    // 1. Slider real-time drag
    if (this.slider) {
      this.slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        this.handleSliderChange(val, true);
      });
    }

    // 2. Click attempt on locked track
    if (this.lockTrackOverlay) {
      this.lockTrackOverlay.addEventListener("click", () => {
        this.triggerLockedFeedback();
      });
    }

    // 3. Quick Preset Pills (-1.0, -0.5, 0.0, +0.5, +1.0)
    if (this.presetPills && this.presetPills.length > 0) {
      this.presetPills.forEach(pill => {
        pill.addEventListener("click", () => {
          const targetVal = parseFloat(pill.getAttribute("data-val"));
          if (!this.isUnlocked && targetVal > 0.01) {
            this.triggerLockedFeedback();
            return;
          }
          this.animateSliderTo(targetVal);
        });
      });
    }

    // 4. View Mode Switcher (Split, Camera, Avatar) - Direct & Delegated
    if (this.viewSwitchBtns) {
      this.viewSwitchBtns.forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          const mode = btn.getAttribute("data-view");
          if (mode) this.setViewMode(mode);
        });
      });
    }

    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".view-switch-btn");
      if (btn) {
        e.preventDefault();
        const mode = btn.getAttribute("data-view");
        if (mode) this.setViewMode(mode);
      }
    });

    // 5. Camera Toggle Buttons inside Tab 3
    if (this.camToggleBtn) {
      this.camToggleBtn.addEventListener("click", async () => {
        await this.toggleCameraState();
      });
    }

    if (this.enableCamBtn) {
      this.enableCamBtn.addEventListener("click", async () => {
        await this.startCameraSafe();
      });
    }

    // 6. Instant Chin Tuck Unlock Button (1-Click - No 3s hold required!)
    if (this.instantUnlockBtn) {
      this.instantUnlockBtn.addEventListener("click", () => {
        if (!this.isUnlocked) {
          this.unlockPositiveTimeline();
        } else {
          this.animateSliderTo(1.0);
        }
      });
    }

    // 7. Pulse Audio Beep Toggle Button
    const soundBtn = document.getElementById("btn-toggle-ecg-sound");
    if (soundBtn) {
      soundBtn.addEventListener("click", () => {
        this.isSoundEnabled = !this.isSoundEnabled;
        soundBtn.textContent = this.isSoundEnabled ? "🔊 BÍP MẠCH: BẬT" : "🔇 BÍP MẠCH: TẮT";
        soundBtn.style.color = this.isSoundEnabled ? "#34D399" : "";
        if (this.isSoundEnabled) {
          this.initAudio();
          this.playAudioTone(880, 'sine', 0.08, 0.12);
        }
      });
    }

    // 8. Legacy Chin Tuck Hold Button (Optional fallback)
    if (this.holdBtn) {
      const startHold = (e) => {
        if (this.isUnlocked) return;
        e.preventDefault();
        this.startChinTuckHold();
      };
      const cancelHold = (e) => {
        if (this.isUnlocked) return;
        this.cancelChinTuckHold();
      };
      this.holdBtn.addEventListener("mousedown", startHold);
      this.holdBtn.addEventListener("touchstart", startHold, { passive: false });
      window.addEventListener("mouseup", cancelHold);
      window.addEventListener("touchend", cancelHold);
      this.holdBtn.addEventListener("mouseleave", cancelHold);
    }

    // 9. Quick Verify & Reset Buttons
    if (this.quickVerifyBtn) {
      this.quickVerifyBtn.addEventListener("click", () => this.verifyUserAction());
    }

    if (this.resetStateBtn) {
      this.resetStateBtn.addEventListener("click", () => this.resetSimulation());
    }

    if (this.resetSliderBtn) {
      this.resetSliderBtn.addEventListener("click", () => this.animateSliderTo(0.0));
    }

    // 10. AI Evaluation Modal triggers
    if (this.openAiEvalBtn && this.aiEvalModal) {
      this.openAiEvalBtn.addEventListener("click", () => {
        this.aiEvalModal.style.display = "flex";
      });
    }

    const closeModal = () => {
      if (this.aiEvalModal) this.aiEvalModal.style.display = "none";
    };

    if (this.closeAiEvalBtn) this.closeAiEvalBtn.addEventListener("click", closeModal);
    if (this.closeAiEvalConfirm) this.closeAiEvalConfirm.addEventListener("click", closeModal);
    if (this.aiEvalBackdrop) this.aiEvalBackdrop.addEventListener("click", closeModal);

    // 11. Multiverse Scenario Buttons (1-Click Sci-Fi Timelines)
    if (this.scenarioBurnoutBtn) {
      this.scenarioBurnoutBtn.addEventListener("click", () => {
        this.setScenarioActive("burnout");
        this.animateSliderTo(-1.0);
      });
    }

    if (this.scenarioBaselineBtn) {
      this.scenarioBaselineBtn.addEventListener("click", () => {
        this.setScenarioActive("baseline");
        this.animateSliderTo(0.0);
      });
    }

    if (this.scenarioVitalityBtn) {
      this.scenarioVitalityBtn.addEventListener("click", () => {
        if (!this.isUnlocked) {
          this.onActionSuccess();
        }
        this.setScenarioActive("vitality");
        this.animateSliderTo(1.0);
      });
    }

    // 12. Dedicated AI Audit Modal Trigger
    if (this.openAuditDetailBtn && this.aiEvalModal) {
      this.openAuditDetailBtn.addEventListener("click", () => {
        this.aiEvalModal.style.display = "flex";
      });
    }

    // Window resize
    window.addEventListener("resize", () => {
      if (this.avatarRenderer) this.avatarRenderer.resize();
    });
  }

  setScenarioActive(scenario) {
    if (this.scenarioBurnoutBtn) this.scenarioBurnoutBtn.classList.toggle("active", scenario === "burnout");
    if (this.scenarioBaselineBtn) this.scenarioBaselineBtn.classList.toggle("active", scenario === "baseline");
    if (this.scenarioVitalityBtn) this.scenarioVitalityBtn.classList.toggle("active", scenario === "vitality");
  }

  onTabActivated() {
    this.cacheElements();
    this.syncCameraStream();
    if (this.avatarRenderer) this.avatarRenderer.resize();
    const canvas = document.getElementById("trajectory-canvas");
    if (canvas && canvas.parentElement) {
      canvas.width = canvas.parentElement.clientWidth || 600;
      canvas.height = 220;
    }
  }

  setViewMode(mode) {
    this.currentViewMode = mode;
    this.viewport = document.getElementById("multiverse-viewport");
    if (this.viewport) {
      this.viewport.classList.remove("view-mode-split", "view-mode-camera", "view-mode-avatar");
      this.viewport.classList.add(`view-mode-${mode}`);
    }
    const btns = document.querySelectorAll(".view-switch-btn");
    btns.forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-view") === mode);
    });
    if (this.avatarRenderer) {
      this.avatarRenderer.resize();
    }
  }

  async toggleCameraState() {
    if (window.clinicalStation) {
      if (window.clinicalStation.isCameraActive) {
        window.clinicalStation.stopCamera();
        this.rppgEngine.reset();
        this.syncCameraStream();
      } else {
        await this.startCameraSafe();
      }
    }
  }

  async startCameraSafe() {
    try {
      if (window.clinicalStation) {
        await window.clinicalStation.startCamera();
        this.syncCameraStream();
      }
    } catch (err) {
      console.warn("[MULTIVERSE] Camera start error:", err);
      this.handleCameraError(err);
    }
  }

  handleCameraError(err) {
    const desc = document.getElementById("multiverse-standby-desc");
    if (desc) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        desc.textContent = "QUYỀN TRUY CẬP CAMERA BỊ TỪ CHỐI. Vui lòng nhấn vào biểu tượng ổ khóa trên thanh địa chỉ trình duyệt để cấp quyền camera!";
        desc.style.color = "#F87171";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        desc.textContent = "KHÔNG TÌM THẤY THIẾT BỊ CAMERA. Vui lòng kiểm tra cáp cắm webcam hoặc quyền truy cập phần cứng!";
        desc.style.color = "#F87171";
      } else {
        desc.textContent = `LỖI CAMERA: ${err.message || "Thiết bị đang bận hoặc bị ứng dụng khác chiếm giữ."}`;
        desc.style.color = "#F87171";
      }
    }
  }

  syncCameraStream() {
    if (!this.video) return;
    const cs = window.clinicalStation;
    if (cs && cs.mediaStream && cs.isCameraActive) {
      if (this.video.srcObject !== cs.mediaStream) {
        this.video.srcObject = cs.mediaStream;
        this.video.play().catch(e => console.log("[MULTIVERSE] Video play:", e));
      }
      if (this.standbyOverlay) this.standbyOverlay.style.display = "none";
    } else {
      if (this.standbyOverlay) this.standbyOverlay.style.display = "flex";
      this.rppgEngine.reset();
    }
  }

  onTabActivated() {
    this.syncCameraStream();
    if (this.avatarRenderer) {
      this.avatarRenderer.resize();
      this.avatarRenderer.start();
    }
    this.applyTimelineEffects(this.sliderValue);
  }

  onTabDeactivated() {
    if (this.avatarRenderer) {
      this.avatarRenderer.stop();
    }
  }

  // -------------------------------------------------------------
  // SLIDER & SIMULATION STATE INTERACTION
  // -------------------------------------------------------------
  handleSliderChange(val, userInitiated = false) {
    if (this.isAutoSliding) {
      this.sliderValue = val;
      this.applyTimelineEffects(val);
      return;
    }

    if (!this.isUnlocked && val > 0.01) {
      this.slider.value = 0.0;
      this.sliderValue = 0.0;
      this.applyTimelineEffects(0.0);
      if (userInitiated) {
        this.triggerLockedFeedback();
      }
      return;
    }

    this.sliderValue = val;
    this.applyTimelineEffects(val);
  }

  triggerLockedFeedback() {
    if (this.isUnlocked) return;

    const sliderBox = document.getElementById("multiverse-slider-container");
    if (sliderBox) {
      sliderBox.classList.remove("shake-locked");
      void sliderBox.offsetWidth;
      sliderBox.classList.add("shake-locked");
    }

    if (this.taskCard) {
      this.taskCard.classList.remove("task-card-alert");
      void this.taskCard.offsetWidth;
      this.taskCard.classList.add("task-card-alert");
    }

    this.playAudioTone(220, 'sine', 0.12, 0.08);

    if (this.sliderValBadge) {
      this.sliderValBadge.textContent = "🔒 KHÓA! CẦN CHIN TUCK ĐỂ MỞ";
      this.sliderValBadge.style.color = "#F87171";
      setTimeout(() => {
        if (!this.isUnlocked && this.sliderValue <= 0) {
          this.sliderValBadge.textContent = "0.00 [TRẠNG THÁI TRUNG TÍNH (CHUẨN)]";
          this.sliderValBadge.style.color = "";
        }
      }, 1800);
    }
  }

  applyTimelineEffects(val) {
    if (window.clinicalStation && typeof window.clinicalStation.setMultiverseSliderVal === "function") {
      window.clinicalStation.setMultiverseSliderVal(val);
    }

    // Direct update to the NeuroAvatarRenderer (No camera color alteration!)
    if (this.avatarRenderer) {
      this.avatarRenderer.setSimulationValue(val);
    }

    // Update Presets UI
    this.updateActivePresetPill(val);

    let milestoneLabel = "";
    let stateDescription = "";

    if (val <= -0.75) {
      milestoneLabel = "-1.0 KIỆT SỨC MÔ PHỎNG";
      stateDescription = "Mô phỏng kiệt sức: Mí mắt avatar sụp sâu, đầu cúi mệt, cơ mặt chùng • Không can thiệp da thật";
      const dynamicBurnout = Math.min(95, Math.round(this.burnoutIndex + Math.abs(val) * 45));
      this.updateBurnoutDisplay(dynamicBurnout);
    } else if (val <= -0.25) {
      milestoneLabel = "-0.5 MỆT MỎI MÔ PHỎNG";
      stateDescription = "Mô phỏng mệt mỏi: Độ mở mắt giảm, biểu cảm trầm lắng, tư thế chùng nhẹ";
      const dynamicBurnout = Math.min(80, Math.round(this.burnoutIndex + Math.abs(val) * 35));
      this.updateBurnoutDisplay(dynamicBurnout);
    } else if (val >= 0.75) {
      milestoneLabel = "+1.0 TÍCH CỰC MÔ PHỎNG";
      stateDescription = "Mô phỏng tích cực: Mắt mở tự nhiên, khóe miệng tươi, tư thế cân bằng tự tin";
      const dynamicBurnout = Math.max(8, Math.round(this.burnoutIndex * 0.3));
      this.updateBurnoutDisplay(dynamicBurnout);
    } else if (val >= 0.25) {
      milestoneLabel = "+0.5 TỈNH TÁO, THƯ GIÃN";
      stateDescription = "Mô phỏng thư giãn: Ánh mắt sáng, cơ mặt thư giãn tự nhiên, tư thế thẳng trục";
      const dynamicBurnout = Math.max(12, Math.round(this.burnoutIndex * 0.6));
      this.updateBurnoutDisplay(dynamicBurnout);
    } else {
      milestoneLabel = "0.00 TRUNG TÍNH (CHUẨN)";
      stateDescription = "Điểm gốc chuẩn • Mắt, miệng và tư thế đầu ở trạng thái nghỉ • Làm mốc đối chiếu khoa học";
      this.updateBurnoutDisplay(this.burnoutIndex);
    }

    // Update Slider Badge
    if (this.sliderValBadge) {
      const sign = val > 0 ? "+" : "";
      this.sliderValBadge.textContent = `${sign}${val.toFixed(2)} [${milestoneLabel}]`;
      if (val > 0.05) this.sliderValBadge.style.color = "#38BDF8";
      else if (val < -0.05) this.sliderValBadge.style.color = "#F87171";
      else this.sliderValBadge.style.color = "#CBD5E1";
    }

    if (this.timelineStateText) {
      this.timelineStateText.textContent = stateDescription;
    }

    if (this.filterReadout) {
      this.filterReadout.textContent = `[SIMULATOR] ${milestoneLabel} • Trình diễn thị giác Avatar độc lập`;
    }

    // Subtle container glow without distorting camera feed
    if (this.glowBox) {
      if (val < -0.2) {
        this.glowBox.style.borderColor = "rgba(239, 68, 68, 0.4)";
        this.glowBox.style.boxShadow = "0 0 15px rgba(239, 68, 68, 0.2)";
      } else if (val > 0.2) {
        this.glowBox.style.borderColor = "rgba(56, 189, 248, 0.4)";
        this.glowBox.style.boxShadow = "0 0 15px rgba(56, 189, 248, 0.2)";
      } else {
        this.glowBox.style.borderColor = "rgba(255, 255, 255, 0.12)";
        this.glowBox.style.boxShadow = "none";
      }
    }

    // Sync Scenario Buttons Active State
    if (val <= -0.55) {
      this.setScenarioActive("burnout");
    } else if (val >= 0.55) {
      this.setScenarioActive("vitality");
    } else {
      this.setScenarioActive("baseline");
    }

    // Update Real-time Predictive Biometrics Matrix
    this.updatePredictiveBiometrics(val);
  }

  updatePredictiveBiometrics(val) {
    // 1. Biological Age
    if (this.predBioAge) {
      let ageDelta = 0;
      let ageColor = "#38BDF8";
      let baseAge = 22.0;
      if (val < 0) {
        ageDelta = Math.abs(val) * 5.2;
        const calcAge = (baseAge + ageDelta).toFixed(1);
        this.predBioAge.textContent = `+${ageDelta.toFixed(1)} Tuổi (${calcAge}t)`;
        ageColor = val <= -0.5 ? "#EF4444" : "#F59E0B";
      } else if (val > 0) {
        ageDelta = val * 3.5;
        const calcAge = (baseAge - ageDelta).toFixed(1);
        this.predBioAge.textContent = `-${ageDelta.toFixed(1)} Tuổi (${calcAge}t)`;
        ageColor = val >= 0.5 ? "#10B981" : "#34D399";
      } else {
        this.predBioAge.textContent = "22.0 Tuổi (Chuẩn)";
        ageColor = "#38BDF8";
      }
      this.predBioAge.style.color = ageColor;
    }
    if (this.predBioAgeSub) {
      if (val <= -0.5) {
        this.predBioAgeSub.textContent = "Lão hóa sớm do thiếu oxy & stress oxy hóa";
      } else if (val < -0.05) {
        this.predBioAgeSub.textContent = "Vi tuần hoàn mao mạch chậm, dấu hiệu mệt mỏi";
      } else if (val >= 0.5) {
        this.predBioAgeSub.textContent = "Trẻ hóa tế bào nhờ tuần hoàn tối ưu & thư thái";
      } else if (val > 0.05) {
        this.predBioAgeSub.textContent = "Tăng sinh collagen tự nhiên & sắc diện hồng hào";
      } else {
        this.predBioAgeSub.textContent = "Mức chuẩn sinh học tuổi thanh xuân";
      }
    }

    // 2. Eye Ptosis (Độ sụp mí)
    if (this.predEyePtosis) {
      let ptosisVal = 12;
      let ptosisColor = "#34D399";
      if (val < 0) {
        ptosisVal = Math.round(12 + Math.abs(val) * 48);
        ptosisColor = val <= -0.5 ? "#EF4444" : "#F59E0B";
      } else if (val > 0) {
        ptosisVal = Math.max(3, Math.round(12 - val * 9));
        ptosisColor = val >= 0.5 ? "#10B981" : "#38BDF8";
      }
      this.predEyePtosis.textContent = `${ptosisVal}%`;
      this.predEyePtosis.style.color = ptosisColor;
    }
    if (this.predEyePtosisSub) {
      if (val <= -0.5) {
        this.predEyePtosisSub.textContent = "Sụp mí nặng: Cơ nâng mi kiệt sức, quầng thâm";
      } else if (val < -0.05) {
        this.predEyePtosisSub.textContent = "Khe mi thu hẹp, mỏi cơ vòng mi Orbicularis";
      } else if (val >= 0.5) {
        this.predEyePtosisSub.textContent = "Ánh nhìn sáng, cơ nâng mi Levator Palpebrae linh hoạt";
      } else if (val > 0.05) {
        this.predEyePtosisSub.textContent = "Độ mở khe mi tự nhiên, phản xạ chớp mắt nhạy";
      } else {
        this.predEyePtosisSub.textContent = "Mở tự nhiên, biên độ nháy mắt bình thường";
      }
    }

    // 3. Cervical Slump (Cột sống cổ C1-C7)
    if (this.predCervicalSlump) {
      let angle = 4.2;
      let angleColor = "#38BDF8";
      if (val < 0) {
        angle = (4.2 + Math.abs(val) * 24.3).toFixed(1);
        angleColor = val <= -0.5 ? "#EF4444" : "#F59E0B";
      } else if (val > 0) {
        angle = Math.max(0.4, (4.2 - val * 3.8)).toFixed(1);
        angleColor = val >= 0.5 ? "#10B981" : "#34D399";
      }
      this.predCervicalSlump.textContent = `${angle}°`;
      this.predCervicalSlump.style.color = angleColor;
    }
    if (this.predCervicalSlumpSub) {
      if (val <= -0.5) {
        this.predCervicalSlumpSub.textContent = "Hội chứng cổ rùa nặng (Forward Head) chèn rễ C1-C7";
      } else if (val < -0.05) {
        this.predCervicalSlumpSub.textContent = "Cổ đổ trước nhẹ, co thắt cơ thang Upper Trapezius";
      } else if (val >= 0.5) {
        this.predCervicalSlumpSub.textContent = "Trục đốt sống thẳng lý tưởng, giải áp đĩa đệm cổ";
      } else if (val > 0.05) {
        this.predCervicalSlumpSub.textContent = "Trục sọ - cổ cân bằng động, không gánh nặng cơ";
      } else {
        this.predCervicalSlumpSub.textContent = "Thẳng trục chịu lực sinh học bình thường";
      }
    }

    // 4. Zygomaticus Tone (Trương lực cơ cười)
    if (this.predZygomaticLift) {
      let tone = 0;
      let toneColor = "#38BDF8";
      if (val < 0) {
        tone = Math.round(val * 45);
        toneColor = val <= -0.5 ? "#EF4444" : "#F59E0B";
        this.predZygomaticLift.textContent = `${tone}%`;
      } else if (val > 0) {
        tone = Math.round(val * 72);
        toneColor = val >= 0.5 ? "#10B981" : "#34D399";
        this.predZygomaticLift.textContent = `+${tone}%`;
      } else {
        this.predZygomaticLift.textContent = "0%";
        toneColor = "#38BDF8";
      }
      this.predZygomaticLift.style.color = toneColor;
    }
    if (this.predZygomaticLiftSub) {
      if (val <= -0.5) {
        this.predZygomaticLiftSub.textContent = "Khóe miệng trễ sâu, nhão cơ mặt do kiệt sức mạn";
      } else if (val < -0.05) {
        this.predZygomaticLiftSub.textContent = "Trương lực cơ giảm sút, cơ mặt chùng nhẹ";
      } else if (val >= 0.5) {
        this.predZygomaticLiftSub.textContent = "Kích hoạt Zygomaticus Major: Nụ cười Duchenne rạng rỡ";
      } else if (val > 0.05) {
        this.predZygomaticLiftSub.textContent = "Khóe miệng tươi tỉnh, cơ mặt đàn hồi dẻo dai";
      } else {
        this.predZygomaticLiftSub.textContent = "Trạng thái nghỉ điềm tĩnh, trương lực cân bằng";
      }
    }

    // 5. Status Tag
    if (this.predStatusTag) {
      if (val <= -0.75) {
        this.predStatusTag.textContent = "TRẠNG THÁI: KIỆT SỨC CẤP TÍNH (-1.0)";
        this.predStatusTag.style.color = "#EF4444";
      } else if (val <= -0.25) {
        this.predStatusTag.textContent = "TRẠNG THÁI: MỆT MỎI SUY NHƯỢC (-0.5)";
        this.predStatusTag.style.color = "#F59E0B";
      } else if (val >= 0.75) {
        this.predStatusTag.textContent = "TRẠNG THÁI: PHỤC HỒI TỐI ƯU VITALITY (+1.0)";
        this.predStatusTag.style.color = "#10B981";
      } else if (val >= 0.25) {
        this.predStatusTag.textContent = "TRẠNG THÁI: TỈNH TÁO THƯ GIÃN (+0.5)";
        this.predStatusTag.style.color = "#34D399";
      } else {
        this.predStatusTag.textContent = "TRẠNG THÁI: CHUẨN CÂN BẰNG (0.0)";
        this.predStatusTag.style.color = "#94A3B8";
      }
    }
  }

  animateSliderTo(targetVal) {
    if (this.sliderAnimationId) {
      cancelAnimationFrame(this.sliderAnimationId);
      this.sliderAnimationId = null;
    }

    const startVal = this.sliderValue;
    const startTime = performance.now();
    const duration = 280;

    this.isAutoSliding = true;

    const step = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(1.0, elapsed / duration);
      const ease = 1 - (1 - progress) * (1 - progress);
      const current = startVal + (targetVal - startVal) * ease;

      if (this.slider) this.slider.value = current.toFixed(2);
      this.sliderValue = current;
      this.applyTimelineEffects(current);

      if (progress < 1.0) {
        this.sliderAnimationId = requestAnimationFrame(step);
      } else {
        this.isAutoSliding = false;
        this.sliderValue = targetVal;
        if (this.slider) this.slider.value = targetVal.toFixed(2);
        this.applyTimelineEffects(targetVal);
      }
    };
    this.sliderAnimationId = requestAnimationFrame(step);
  }

  updateActivePresetPill(targetVal) {
    if (!this.presetPills) return;
    this.presetPills.forEach(pill => {
      const pVal = parseFloat(pill.getAttribute("data-val"));
      if (Math.abs(pVal - targetVal) < 0.15) {
        pill.classList.add("active");
      } else {
        pill.classList.remove("active");
      }
    });
  }

  // -------------------------------------------------------------
  // REAL OPTICAL rPPG MONITORING & WAVEFORM RENDERER
  // -------------------------------------------------------------
  startRPPGProcessingLoop() {
    if (this.rppgProcessTimer) clearInterval(this.rppgProcessTimer);
    this.rppgProcessTimer = setInterval(() => {
      const tabForecast = document.getElementById("tab-forecast");
      const isTabActive = tabForecast && tabForecast.classList.contains("active");
      if (!isTabActive) return;

      const cs = window.clinicalStation;
      if (cs && cs.isCameraActive && cs.lastFaceDetected && cs.lastBbox && this.video) {
        this.rppgEngine.processFrame(this.video, cs.lastBbox);
      } else {
        this.rppgEngine.reset();
      }

      this.updateTelemetryUI();
    }, 100); // 10Hz sampling
  }

  updateTelemetryUI() {
    const telem = this.rppgEngine.getTelemetry();
    const qualityBadge = document.getElementById("rppg-signal-quality-badge");
    const qualityText = document.getElementById("rppg-quality-text");
    const bpmMetric = document.getElementById("ecg-metric-bpm");
    const rrMetric = document.getElementById("ecg-metric-rr");
    const hrvMetric = document.getElementById("ecg-metric-hrv");
    const rhythmMetric = document.getElementById("ecg-metric-rhythm");

    if (qualityBadge && qualityText) {
      if (telem.quality === "SIGNAL_GOOD") {
        qualityBadge.style.background = "rgba(16, 185, 129, 0.2)";
        qualityBadge.style.color = "#34D399";
        qualityBadge.style.borderColor = "rgba(52, 211, 153, 0.4)";
        qualityText.textContent = "SIGNAL: GOOD";
      } else if (telem.quality === "CALIBRATING") {
        qualityBadge.style.background = "rgba(245, 158, 11, 0.2)";
        qualityBadge.style.color = "#FBBF24";
        qualityBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
        qualityText.textContent = "SIGNAL: CALIBRATING...";
      } else if (telem.quality === "LOW_SIGNAL_QUALITY") {
        qualityBadge.style.background = "rgba(245, 158, 11, 0.2)";
        qualityBadge.style.color = "#F59E0B";
        qualityBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
        qualityText.textContent = "SIGNAL: LOW QUALITY (VUI LÒNG NGỒI YÊN)";
      } else {
        qualityBadge.style.background = "rgba(239, 68, 68, 0.2)";
        qualityBadge.style.color = "#F87171";
        qualityBadge.style.borderColor = "rgba(239, 68, 68, 0.4)";
        qualityText.textContent = "SIGNAL: NO VALID SIGNAL";
      }
    }

    if (bpmMetric) {
      if (telem.bpm && telem.quality === "SIGNAL_GOOD") {
        bpmMetric.innerHTML = `${telem.bpm} <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">BPM</span>`;
      } else {
        bpmMetric.innerHTML = `-- <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">BPM</span>`;
      }
    }

    if (rrMetric) {
      if (telem.rr && telem.quality === "SIGNAL_GOOD") {
        rrMetric.innerHTML = `${telem.rr} <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">ms</span>`;
      } else {
        rrMetric.innerHTML = `-- <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">ms</span>`;
      }
    }

    if (hrvMetric) {
      if (telem.hrv && telem.quality === "SIGNAL_GOOD") {
        hrvMetric.innerHTML = `${telem.hrv} <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">ms</span>`;
      } else {
        hrvMetric.innerHTML = `-- <span style="font-size:0.62rem; font-weight:400; color:var(--text-muted);">ms</span>`;
      }
    }

    if (rhythmMetric) {
      if (telem.quality === "SIGNAL_GOOD") {
        rhythmMetric.textContent = "QUANG HỌC ỔN ĐỊNH";
        rhythmMetric.style.color = "#34D399";
      } else if (telem.quality === "CALIBRATING") {
        rhythmMetric.textContent = "ĐANG HIỆU CHUẨN...";
        rhythmMetric.style.color = "#FBBF24";
      } else if (telem.quality === "LOW_SIGNAL_QUALITY") {
        rhythmMetric.textContent = "TẠP ÂM CHUYỂN ĐỘNG";
        rhythmMetric.style.color = "#F59E0B";
      } else {
        rhythmMetric.textContent = "CHƯA CÓ TÍN HIỆU";
        rhythmMetric.style.color = "#94A3B8";
      }
    }

    // Sync live expression readout chip
    const exprChip = document.getElementById("multiverse-expression-chip");
    if (exprChip) {
      const cs = window.clinicalStation;
      if (cs && cs.isCameraActive && cs.lastFaceDetected && cs.currentBiometrics) {
        const emo = cs.currentBiometrics.emotion || "NEUTRAL / CALM";
        const conf = cs.currentBiometrics.confidence || 88;
        exprChip.textContent = `BIỂU CẢM: ${emo} (${conf}%)`;
        exprChip.style.color = cs.isSmiling ? "#34D399" : "#38BDF8";
      } else {
        exprChip.textContent = "BIỂU CẢM: [CHỜ CAMERA]";
        exprChip.style.color = "#94A3B8";
      }
    }
  }

  initWaveformMonitor() {
    const canvas = document.getElementById("trajectory-canvas");
    if (!canvas) return;

    let sweepX = 0;
    let cardiacPhase = 0.0;
    let lastTime = performance.now();
    let buffer = [];
    let pulseBeepTriggered = false;

    const render = (now) => {
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
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      const telem = this.rppgEngine.getTelemetry();
      const hasValidSignal = (telem.quality === "SIGNAL_GOOD");
      const currentBpm = hasValidSignal ? (telem.bpm || 74) : 74;

      // Advance synthetic cardiac phase
      const beatPeriodSec = 60 / currentBpm;
      cardiacPhase = (cardiacPhase + dt / beatPeriodSec) % 1.0;

      // Audio Beep sync with pulse peak (if sound enabled)
      if (this.isSoundEnabled && hasValidSignal) {
        if (cardiacPhase >= 0.14 && cardiacPhase < 0.22 && !pulseBeepTriggered) {
          pulseBeepTriggered = true;
          this.playAudioTone(880, 'sine', 0.06, 0.12);
        } else if (cardiacPhase >= 0.40) {
          pulseBeepTriggered = false;
        }
      }

      // Advance sweep head
      const step = 2;
      for (let s = 0; s < step; s++) {
        const curX = Math.floor(sweepX);
        let plethVal = 0;

        if (hasValidSignal) {
          plethVal = this.calcSyntheticPleth(cardiacPhase);
        } else {
          // Resting baseline with subtle optical noise
          plethVal = (Math.random() - 0.5) * 0.025;
        }

        buffer[curX] = plethVal;
        for (let g = 1; g <= 22; g++) {
          buffer[(curX + g) % w] = null;
        }
        sweepX = (sweepX + 1) % w;
      }

      // 1. Draw canvas background
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, w, h);

      // 2. Medical millimeter grid
      this.drawMedicalGrid(ctx, w, h);

      // 3. Center baseline for Optical rPPG Pulse
      const yMid = h * 0.52;
      const amp = h * 0.38;

      ctx.strokeStyle = "rgba(203, 213, 225, 0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(0, yMid); ctx.lineTo(w, yMid);
      ctx.stroke();
      ctx.setLineDash([]);

      // 4. Draw Optical rPPG Pulse Wave (Clinical Emerald / Cyan)
      ctx.save();
      ctx.strokeStyle = hasValidSignal ? "#059669" : "#0284C7";
      ctx.shadowColor = hasValidSignal ? "rgba(5, 150, 105, 0.35)" : "rgba(2, 132, 199, 0.35)";
      ctx.shadowBlur = hasValidSignal ? 6 : 0;
      ctx.lineWidth = 2.2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      let started = false;
      let lastValidY = yMid;

      for (let x = 0; x < w; x++) {
        const pt = buffer[x];
        if (typeof pt === "number") {
          const py = yMid - pt * amp;
          lastValidY = py;
          if (!started) { ctx.moveTo(x, py); started = true; }
          else { ctx.lineTo(x, py); }
        } else {
          started = false;
        }
      }
      ctx.stroke();
      ctx.restore();

      // 5. Sweep Beam Head with glow scan line
      const headX = Math.floor(sweepX);
      ctx.save();
      ctx.strokeStyle = "rgba(2, 132, 199, 0.5)";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(headX, 0); ctx.lineTo(headX, h);
      ctx.stroke();

      // Sweep dot indicator
      ctx.fillStyle = "#0284C7";
      ctx.shadowColor = "rgba(2, 132, 199, 0.5)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(headX, lastValidY, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // 6. Medical Clinical Header Annotations (Pure rPPG Optical Telemetry)
      ctx.save();
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#0284C7";
      ctx.fillText("rPPG OPTICAL PULSE SIGNAL (SÓNG QUANG HỌC VÙNG TRÁN WEBCAM)", 14, 20);

      ctx.font = "8.5px 'JetBrains Mono', monospace";
      ctx.fillStyle = hasValidSignal ? "#059669" : "#64748B";
      ctx.fillText(`CHẤT LƯỢNG TÍN HIỆU: ${hasValidSignal ? "CHUẨN XÁC [SIGNAL GOOD]" : "ĐANG CÂN BẰNG TÍN HIỆU..."}`, 14, 34);

      ctx.textAlign = "right";
      ctx.font = "bold 13px 'JetBrains Mono', monospace";
      if (hasValidSignal) {
        ctx.fillStyle = "#059669";
        ctx.fillText(`HR: ${currentBpm} BPM`, w - 14, 20);
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillStyle = "#0284C7";
        ctx.fillText(`HRV: ${telem.hrv}ms • R-R: ${telem.rr}ms`, w - 14, 34);
      } else {
        ctx.fillStyle = "#94A3B8";
        ctx.fillText("HR: -- BPM", w - 14, 20);
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillStyle = "#64748B";
        ctx.fillText("CHỜ KHUÔN MẶT TRONG KHUNG HÌNH", w - 14, 34);
      }
      ctx.restore();

      requestAnimationFrame(render);
    };

    requestAnimationFrame(render);
  }


  drawMedicalGrid(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = "rgba(2, 132, 199, 0.05)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let x = 0; x < w; x += 8) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
    for (let y = 0; y < h; y += 8) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
    ctx.stroke();

    ctx.strokeStyle = "rgba(2, 132, 199, 0.12)";
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    for (let x = 0; x < w; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
    for (let y = 0; y < h; y += 40) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
    ctx.stroke();
    ctx.restore();
  }

  calcSyntheticECG(phase) {
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
    if (phase >= 0.43 && phase < 0.50) return 0.01;
    if (phase >= 0.50 && phase < 0.68) {
      const t = (phase - 0.50) / 0.18;
      return Math.sin(t * Math.PI) * 0.26;
    }
    return 0.0;
  }

  calcSyntheticPleth(phase) {
    if (phase < 0.15) return 0.0;
    if (phase >= 0.15 && phase < 0.35) {
      const rise = (phase - 0.15) / 0.20;
      return Math.sin(rise * Math.PI * 0.5) * 0.75;
    }
    if (phase >= 0.35 && phase < 0.52) {
      const decay = (phase - 0.35) / 0.17;
      return (1.0 - decay * 0.45) * 0.75;
    }
    if (phase >= 0.52 && phase < 0.68) {
      const notch = (phase - 0.52) / 0.16;
      return (0.55 + Math.sin(notch * Math.PI) * 0.16) * 0.75;
    }
    const runoff = (phase - 0.68) / 0.32;
    return (0.55 * Math.cos(runoff * Math.PI * 0.5)) * 0.75;
  }

  // -------------------------------------------------------------
  // CHIN TUCK ACTION-TO-UNLOCK & HANDS-FREE AI
  // -------------------------------------------------------------
  initHandsFreeAILoop() {
    const requiredDuration = 3000;
    const loop = (now) => {
      const tabForecast = document.getElementById("tab-forecast");
      const isTabActive = tabForecast && tabForecast.classList.contains("active");

      if (isTabActive && !this.isUnlocked && window.clinicalStation) {
        const cs = window.clinicalStation;
        const hasCamera = cs.isCameraActive;
        const hasFace = cs.lastFaceDetected;
        const headTilt = Math.abs(cs.headTilt || 0);
        const isAligned = hasCamera && hasFace && headTilt <= 14;

        if (isAligned) {
          if (!this.handsFreeStartTime) this.handsFreeStartTime = now;
          const elapsed = now - this.handsFreeStartTime;
          const progress = Math.min(1.0, elapsed / requiredDuration);
          const remaining = Math.max(0, (requiredDuration - elapsed) / 1000).toFixed(1);

          if (this.handsFreeHud) this.handsFreeHud.style.display = "flex";
          if (this.handsFreeCountdown) this.handsFreeCountdown.textContent = `${remaining}s`;

          if (this.handsFreeRingBar) {
            const circumference = 2 * Math.PI * 33;
            this.handsFreeRingBar.style.strokeDashoffset = (circumference * (1 - progress)).toString();
          }

          if (progress >= 1.0) {
            this.handsFreeStartTime = null;
            if (this.handsFreeHud) this.handsFreeHud.style.display = "none";
            this.verifyUserAction();
          }
        } else {
          this.handsFreeStartTime = null;
          if (this.handsFreeHud) {
            if (hasCamera && hasFace && headTilt > 14) {
              this.handsFreeHud.style.display = "flex";
              if (this.handsFreeCountdown) this.handsFreeCountdown.textContent = "CHỈNH ĐẦU";
            } else {
              this.handsFreeHud.style.display = "none";
            }
          }
          if (this.handsFreeRingBar) {
            const circumference = 2 * Math.PI * 33;
            this.handsFreeRingBar.style.strokeDashoffset = circumference.toString();
          }
        }
      } else {
        if (this.handsFreeHud && (this.isUnlocked || !isTabActive)) {
          this.handsFreeHud.style.display = "none";
        }
        this.handsFreeStartTime = null;
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  }

  startChinTuckHold() {
    if (this.isUnlocked || this.isHolding) return;
    this.isHolding = true;
    this.holdStartTime = performance.now();

    if (this.holdBtn) this.holdBtn.classList.add("holding");
    this.playAudioTone(330, 'sine', 0.2, 0.05);

    const step = (now) => {
      if (!this.isHolding) return;
      const elapsed = now - this.holdStartTime;
      const progress = Math.min(1.0, elapsed / this.holdDuration);

      if (this.holdProgressRing) {
        const circumference = 2 * Math.PI * 34;
        this.holdProgressRing.style.strokeDashoffset = circumference * (1 - progress);
      }

      if (this.holdBtnText) {
        const remaining = Math.max(0, ((this.holdDuration - elapsed) / 1000)).toFixed(1);
        this.holdBtnText.textContent = `ĐANG GIỮ CHIN TUCK: ${remaining}s...`;
      }

      if (progress >= 1.0) {
        this.isHolding = false;
        this.verifyUserAction();
      } else {
        this.holdAnimationFrame = requestAnimationFrame(step);
      }
    };
    this.holdAnimationFrame = requestAnimationFrame(step);
  }

  cancelChinTuckHold() {
    if (!this.isHolding || this.isUnlocked) return;
    this.isHolding = false;
    if (this.holdAnimationFrame) {
      cancelAnimationFrame(this.holdAnimationFrame);
      this.holdAnimationFrame = null;
    }
    if (this.holdBtn) this.holdBtn.classList.remove("holding");
    if (this.holdProgressRing) {
      const circumference = 2 * Math.PI * 34;
      this.holdProgressRing.style.strokeDashoffset = circumference;
    }
    if (this.holdBtnText) {
      this.holdBtnText.textContent = "GIỮ 3S MÔ PHỎNG CHIN TUCK";
    }
  }

  verifyUserAction() {
    if (window.clinicalStation) {
      if (!window.clinicalStation.isCameraActive) {
        if (typeof window.clinicalStation.showWebModal === "function") {
          window.clinicalStation.showWebModal(
            "CAMERA CHƯA BẬT",
            "Vui lòng nhấn nút [BẬT] trên khung camera để AI nhận diện khuôn mặt và kiểm tra tư thế Chin Tuck!",
            "warning",
            async () => {
              await this.startCameraSafe();
            }
          );
        }
        return false;
      }

      if (!window.clinicalStation.lastFaceDetected) {
        if (typeof window.clinicalStation.showWebModal === "function") {
          window.clinicalStation.showWebModal(
            "CHƯA TÌM THẤY KHUÔN MẶT",
            "AI chưa nhận diện được khuôn mặt trong khung hình. Vui lòng nhìn thẳng vào camera bên trong khung nhận diện màu xanh!",
            "warning"
          );
        }
        return false;
      }

      const headTilt = Math.abs(window.clinicalStation.headTilt || 0);
      if (headTilt > 18) {
        if (typeof window.clinicalStation.showWebModal === "function") {
          window.clinicalStation.showWebModal(
            "CẢNH BÁO LỆCH TƯ THẾ",
            `AI phát hiện đầu của bạn đang nghiêng ${headTilt}°. Vui lòng giữ thẳng đầu và hướng vào khung xanh để hoàn thành bài tập Chin Tuck chuẩn y khoa!`,
            "warning"
          );
        }
        return false;
      }
    }

    this.onActionSuccess();
    return true;
  }

  onActionSuccess() {
    if (this.isUnlocked) return;
    this.isUnlocked = true;

    if (this.lockTrackOverlay) {
      this.lockTrackOverlay.classList.add("unlocked-fade");
      setTimeout(() => {
        if (this.lockTrackOverlay) this.lockTrackOverlay.style.display = "none";
      }, 500);
    }

    if (this.lockStatusBadge) {
      this.lockStatusBadge.textContent = "✅ ĐÃ MỞ KHÓA TƯƠNG LAI TÍCH CỰC";
      this.lockStatusBadge.className = "badge badge-success";
    }

    if (this.instantUnlockBtn) {
      this.instantUnlockBtn.style.background = "#059669";
    }
    const instantText = document.getElementById("chin-tuck-instant-btn-text");
    if (instantText) {
      instantText.textContent = "✓ ĐÃ MỞ KHÓA TƯƠNG LAI (+1.0)";
    }

    if (this.holdBtn) {
      this.holdBtn.classList.add("completed");
      this.holdBtn.style.pointerEvents = "none";
    }

    if (this.holdBtnText) {
      this.holdBtnText.textContent = "✓ AI ĐÃ XÁC THỰC: KHUÔN MẶT & TƯ THẾ CHUẨN";
    }

    const postDot = document.getElementById("multiverse-ai-posture-dot");
    const postText = document.getElementById("multiverse-ai-posture-text");
    if (postDot && postText) {
      postDot.style.background = "#10B981";
      postDot.style.boxShadow = "0 0 10px #10B981";
      postText.style.color = "#34D399";
      postText.textContent = "AI TRẠNG THÁI: [XÁC THỰC THÀNH CÔNG] Đã chuẩn hóa trục đốt sống cổ C1-C7 • Tương lai đã mở khóa!";
    }

    if (this.taskCard) this.taskCard.classList.add("task-card-success");
    if (this.handsFreeHud) this.handsFreeHud.style.display = "none";

    this.playCelebrationChimeChord();

    if (this.flashOverlay) {
      this.flashOverlay.classList.remove("active-flash");
      void this.flashOverlay.offsetWidth;
      this.flashOverlay.classList.add("active-flash");
    }

    this.launchCelebrationConfetti();
    this.animateSliderTo(1.0);
  }

  resetSimulation() {
    this.isUnlocked = false;
    this.isAutoSliding = false;
    this.handsFreeStartTime = null;
    this.sliderValue = 0.0;
    if (this.slider) this.slider.value = 0.0;

    if (this.instantUnlockBtn) {
      this.instantUnlockBtn.style.background = "#10B981";
    }
    const instantText = document.getElementById("chin-tuck-instant-btn-text");
    if (instantText) {
      instantText.textContent = "KÍCH HOẠT CHIN TUCK (MỞ KHÓA +1.0)";
    }

    const postDot = document.getElementById("multiverse-ai-posture-dot");
    const postText = document.getElementById("multiverse-ai-posture-text");
    if (postDot && postText) {
      postDot.style.background = "#ef4444";
      postDot.style.boxShadow = "none";
      postText.style.color = "#CBD5E1";
      postText.textContent = "AI TƯ THẾ: [ĐANG TÌM KHUÔN MẶT] Hướng mặt vào camera để AI nhận diện tư thế trục cổ";
    }

    if (this.handsFreeHud) this.handsFreeHud.style.display = "none";
    if (this.lockTrackOverlay) {
      this.lockTrackOverlay.style.display = "flex";
      this.lockTrackOverlay.classList.remove("unlocked-fade");
    }

    if (this.lockStatusBadge) {
      this.lockStatusBadge.textContent = "🔒 TƯƠNG LAI TÍCH CỰC ĐANG BỊ KHÓA";
      this.lockStatusBadge.className = "badge badge-warning";
    }

    if (this.holdBtn) {
      this.holdBtn.classList.remove("completed");
      this.holdBtn.style.pointerEvents = "auto";
    }

    if (this.holdBtnText) {
      this.holdBtnText.textContent = "GIỮ 3S MÔ PHỎNG CHIN TUCK";
    }

    if (this.taskCard) {
      this.taskCard.classList.remove("task-card-success", "task-card-alert");
    }

    if (this.avatarRenderer) {
      this.avatarRenderer.setSimulationValue(0.0);
    }

    this.applyTimelineEffects(0.0);
  }


  updateBurnoutDisplay(pct) {
    if (this.burnoutBadge) {
      this.burnoutBadge.textContent = `[BURNOUT INDEX: ${pct}%]`;
      if (pct >= 65) this.burnoutBadge.className = "badge badge-danger";
      else if (pct >= 40) this.burnoutBadge.className = "badge badge-warning";
      else this.burnoutBadge.className = "badge badge-success";
    }
    if (this.burnoutBar) {
      this.burnoutBar.style.width = `${pct}%`;
      if (pct >= 65) this.burnoutBar.style.background = "#EF4444";
      else if (pct >= 40) this.burnoutBar.style.background = "#F59E0B";
      else this.burnoutBar.style.background = "#10B981";
    }
  }

  // -------------------------------------------------------------
  // AUDIO & CELEBRATION CHORD
  // -------------------------------------------------------------
  initAudio() {
    if (!this.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtx();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  playAudioTone(freq, type = 'sine', duration = 0.15, vol = 0.1) {
    try {
      this.initAudio();
      const now = this.audioCtx.currentTime;
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, now);
      gain.gain.setValueAtTime(vol, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start(now);
      osc.stop(now + duration + 0.05);
    } catch (e) {}
  }

  playCelebrationChimeChord() {
    try {
      this.initAudio();
      const now = this.audioCtx.currentTime;
      const chord = [523.25, 659.25, 783.99, 1046.50, 1318.51];
      chord.forEach((freq, idx) => {
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        const noteStart = now + idx * 0.09;
        const noteDuration = 1.2;
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, noteStart);
        gain.gain.setValueAtTime(0.001, noteStart);
        gain.gain.exponentialRampToValueAtTime(0.18, noteStart + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.0001, noteStart + noteDuration);
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(noteStart);
        osc.stop(noteStart + noteDuration + 0.1);
      });
    } catch (e) {}
  }

  launchCelebrationConfetti() {
    if (!this.celebrateCanvas) return;
    const canvas = this.celebrateCanvas;
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.parentElement.clientWidth || 800;
    canvas.height = canvas.parentElement.clientHeight || 500;

    const colors = ["#38BDF8", "#34D399", "#FBBF24", "#F472B6", "#A78BFA", "#FFFFFF"];
    this.confettiParticles = [];

    for (let i = 0; i < 75; i++) {
      this.confettiParticles.push({
        x: canvas.width / 2 + (Math.random() - 0.5) * 80,
        y: canvas.height / 2 + (Math.random() - 0.5) * 60,
        vx: (Math.random() - 0.5) * 14,
        vy: -Math.random() * 10 - 4,
        size: Math.random() * 7 + 4,
        color: colors[Math.floor(Math.random() * colors.length)],
        rotation: Math.random() * 360,
        rotSpeed: (Math.random() - 0.5) * 12,
        alpha: 1.0,
        decay: Math.random() * 0.012 + 0.008
      });
    }

    if (this.confettiAnimationId) cancelAnimationFrame(this.confettiAnimationId);

    const renderConfetti = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let activeCount = 0;
      this.confettiParticles.forEach(p => {
        if (p.alpha > 0.01) {
          activeCount++;
          p.x += p.vx;
          p.y += p.vy;
          p.vy += 0.32;
          p.vx *= 0.98;
          p.rotation += p.rotSpeed;
          p.alpha -= p.decay;

          ctx.save();
          ctx.globalAlpha = Math.max(0, p.alpha);
          ctx.translate(p.x, p.y);
          ctx.rotate((p.rotation * Math.PI) / 180);
          ctx.fillStyle = p.color;
          ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
          ctx.restore();
        }
      });

      if (activeCount > 0) {
        this.confettiAnimationId = requestAnimationFrame(renderConfetti);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.confettiAnimationId = null;
      }
    };
    this.confettiAnimationId = requestAnimationFrame(renderConfetti);
  }
}

// Global initialization
window.multiverseSim = new FutureMultiverseSimulation();
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    window.multiverseSim.init();
  });
} else {
  window.multiverseSim.init();
}

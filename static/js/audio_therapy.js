/**
 * Acoustic Therapy Engine (100% Offline Web Audio API Synthesizer)
 * Implements 432Hz Harmonic Sine, Tibetan Singing Bowl,
 * Gentle Rain Pink Noise, and Generative Alpha Binaural Drone.
 */

class AcousticTherapyManager {
  constructor() {
    this.ctx = null;
    this.isAmbientPlaying = false;
    this.ambientNodes = [];
    this.lastInterventionTime = 0;
    this.interventionCooldown = 15000; // 15 seconds cooldown between auto interventions
  }

  initAudioContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  // 1. Play 5-Second 432Hz Harmonic Sine Tone
  play432HzChime() {
    this.initAudioContext();
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(432, now);

    // ADSR Envelope: Soft attack, gentle 5s exponential decay
    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(0.25, now + 0.3);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 5.0);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + 5.1);
  }

  // 2. Play Tibetan Singing Bowl (Additive Harmonics + Exponential Decay)
  playTibetanBowl() {
    this.initAudioContext();
    const now = this.ctx.currentTime;
    const fundamental = 216; // 216Hz base
    const harmonicFreqs = [fundamental, fundamental * 2, fundamental * 3, fundamental * 4.1];
    const harmonicWeights = [0.35, 0.20, 0.10, 0.05];

    // Master gain for the bowl
    const masterGain = this.ctx.createGain();
    masterGain.gain.setValueAtTime(0.001, now);
    masterGain.gain.exponentialRampToValueAtTime(0.3, now + 0.15);
    masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 5.5);
    masterGain.connect(this.ctx.destination);

    // LFO for slow vibrato / acoustic beating
    const lfo = this.ctx.createOscillator();
    const lfoGain = this.ctx.createGain();
    lfo.frequency.setValueAtTime(3.2, now); // 3.2 Hz beating
    lfoGain.gain.setValueAtTime(3.0, now);
    lfo.connect(lfoGain);

    harmonicFreqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const hGain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now);
      lfoGain.connect(osc.frequency);

      hGain.gain.setValueAtTime(harmonicWeights[idx], now);
      osc.connect(hGain);
      hGain.connect(masterGain);

      osc.start(now);
      osc.stop(now + 5.6);
    });

    lfo.start(now);
    lfo.stop(now + 5.6);
  }

  // 3. Play Gentle Rain Noise Burst (5s)
  playGentleRain(duration = 5.0) {
    this.initAudioContext();
    const bufferSize = this.ctx.sampleRate * duration;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);

    // Pink noise generation
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.035;
      b6 = white * 0.115926;
    }

    const noiseSource = this.ctx.createBufferSource();
    noiseSource.buffer = buffer;

    // Gentle lowpass filter at 1100Hz
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(1100, this.ctx.currentTime);

    const gain = this.ctx.createGain();
    const now = this.ctx.currentTime;
    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.4);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    noiseSource.connect(filter);
    filter.connect(gain);
    gain.connect(this.ctx.destination);

    noiseSource.start(now);
    noiseSource.stop(now + duration + 0.1);
  }

  // 4. Toggle Continuous Ambient Alpha Wave Drone (Binaural Beats: 432Hz vs 442Hz -> 10Hz Alpha)
  toggleAmbientAlphaDrone(enable) {
    this.initAudioContext();
    if (enable) {
      if (this.isAmbientPlaying) return;
      const now = this.ctx.currentTime;

      // Master ambient gain
      const masterGain = this.ctx.createGain();
      masterGain.gain.setValueAtTime(0.001, now);
      masterGain.gain.exponentialRampToValueAtTime(0.09, now + 2.0); // Gentle fade in
      masterGain.connect(this.ctx.destination);

      // Left Channel: 432Hz
      const oscL = this.ctx.createOscillator();
      const pannerL = this.ctx.createStereoPanner ? this.ctx.createStereoPanner() : null;
      oscL.type = 'sine';
      oscL.frequency.setValueAtTime(432.0, now);
      if (pannerL) {
        pannerL.pan.setValueAtTime(-0.8, now);
        oscL.connect(pannerL);
        pannerL.connect(masterGain);
      } else {
        oscL.connect(masterGain);
      }

      // Right Channel: 442Hz (Creates 10Hz Alpha entrainment)
      const oscR = this.ctx.createOscillator();
      const pannerR = this.ctx.createStereoPanner ? this.ctx.createStereoPanner() : null;
      oscR.type = 'sine';
      oscR.frequency.setValueAtTime(442.0, now);
      if (pannerR) {
        pannerR.pan.setValueAtTime(0.8, now);
        oscR.connect(pannerR);
        pannerR.connect(masterGain);
      } else {
        oscR.connect(masterGain);
      }

      // Subtle sub-bass grounding 108Hz
      const oscSub = this.ctx.createOscillator();
      const subGain = this.ctx.createGain();
      oscSub.type = 'triangle';
      oscSub.frequency.setValueAtTime(108.0, now);
      subGain.gain.setValueAtTime(0.03, now);
      oscSub.connect(subGain);
      subGain.connect(masterGain);

      oscL.start(now);
      oscR.start(now);
      oscSub.start(now);

      this.ambientNodes = [oscL, oscR, oscSub, masterGain];
      this.isAmbientPlaying = true;
    } else {
      if (!this.isAmbientPlaying) return;
      const now = this.ctx.currentTime;
      const masterGain = this.ambientNodes[3];
      if (masterGain) {
        masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.5);
      }
      setTimeout(() => {
        this.ambientNodes.forEach(node => {
          try {
            if (node.stop) node.stop();
            if (node.disconnect) node.disconnect();
          } catch (e) { }
        });
        this.ambientNodes = [];
        this.isAmbientPlaying = false;
      }, 1600);
    }
  }

  // Automatic trigger check from live metrics
  checkAutoIntervention(valence, arousal, exhaustion) {
    const now = Date.now();
    if (now - this.lastInterventionTime < this.interventionCooldown) return;

    if (exhaustion >= 3) {
      this.lastInterventionTime = now;
      this.playTibetanBowl();
      this.notifyIntervention("TIBETAN_BOWL", "KIỆT SỨC CẤP 3/5: TỰ ĐỘNG KÍCH HOẠT CHUÔNG XOAY TÂY TẠNG");
    } else if (arousal > 0.6 && valence < -0.2) {
      this.lastInterventionTime = now;
      this.play432HzChime();
      this.notifyIntervention("432HZ_HEALING", "AROUSAL CAO / CĂNG THẲNG: TỰ ĐỘNG PHÁT SÓNG HÀI 432HZ");
    } else if (valence < -0.5) {
      this.lastInterventionTime = now;
      this.playGentleRain(5.0);
      this.notifyIntervention("GENTLE_RAIN", "SUY GIẢM VALENCE: TỰ ĐỘNG PHÁT TIẾNG MƯA LÀM DỊU HỆ THẦN KINH");
    }
  }

  notifyIntervention(type, message) {
    const badge = document.getElementById("acoustic-intervention-badge");
    if (badge) {
      badge.textContent = `Âm học: ${type}`;
      badge.className = "badge badge-clinical";
      setTimeout(() => {
        badge.textContent = "Âm học: Chờ kích hoạt";
        badge.className = "badge badge-neutral";
      }, 6000);
    }
  }
}

window.acousticEngine = new AcousticTherapyManager();

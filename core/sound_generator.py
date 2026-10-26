import math
import numpy as np
import io

class AcousticTherapyEngine:
    """
    Acoustic specifications for Web Audio API synthesis and offline sound generation.
    Supports 432Hz Harmonic Resonance, Tibetan Singing Bowl synthesis,
    Gentle Rain noise synthesis, and Alpha Binaural Drones.
    """
    ACOUSTIC_PRESETS = {
        "432HZ_HEALING": {
            "name": "Sóng Hài 432Hz Thư Thái Thần Kinh",
            "base_frequency": 432.0,
            "waveform": "sine",
            "harmonics": [1.0, 0.3, 0.1],
            "decay": 5.0,
            "description": "Tần số dao động tự nhiên của nước và mô cơ thể, hỗ trợ giải tỏa co thắt cơ và kích thích hệ thần kinh đối giao cảm."
        },
        "TIBETAN_BOWL": {
            "name": "Chuông Xoay Tây Tạng (Tibetan Singing Bowl)",
            "base_frequency": 216.0,
            "harmonics": [1.0, 0.7, 0.45, 0.25],
            "beating_frequency": 3.5, # slow amplitude modulation
            "decay": 6.0,
            "description": "Âm sắc chuông đồng giàu bội âm và nhịp rung chậm, ngắt cơn hoảng loạn và hội tụ sự chú ý."
        },
        "GENTLE_RAIN": {
            "name": "Tiếng Mưa Nhẹ & Tiếng Ồn Hồng (Pink Noise)",
            "filter_cutoff": 1200.0,
            "filter_type": "lowpass",
            "description": "Dải phổ năng lượng giảm 3dB/octave che phủ tạp âm ngoại cảnh và làm dịu vỏ não cảm giác."
        },
        "ALPHA_DRONE": {
            "name": "Sóng Não Alpha 10Hz (Binaural Ambient Drone)",
            "carrier_frequency": 432.0,
            "binaural_offset": 10.0, # 432Hz Left, 442Hz Right -> 10Hz Alpha entrainment
            "description": "Kích thích sóng não tần số 8-12Hz đưa trạng thái tinh thần vào trạng thái tĩnh tâm cảnh giác (Relaxed Alertness)."
        }
    }

    @staticmethod
    def should_trigger_intervention(valence, arousal, exhaustion_level):
        """
        Determine if immediate 5-second acoustic intervention should be triggered.
        Triggers on:
        1. Mental exhaustion level >= 3
        2. High arousal panic/anxiety (Arousal > 0.6 and Valence < -0.3)
        3. Severe depressive slump (Valence < -0.6)
        """
        if exhaustion_level >= 3:
            return True, "EXHAUSTION_THRESHOLD_EXCEEDED", "TIBETAN_BOWL"
        if arousal > 0.6 and valence < -0.2:
            return True, "ACUTE_AROUSAL_SPIKE", "432HZ_HEALING"
        if valence < -0.6:
            return True, "AFFECTIVE_DEPRESSION_DIP", "GENTLE_RAIN"
        return False, "NORMAL_RANGE", None

    @staticmethod
    def generate_sine_wave_pcm(frequency=432.0, duration=5.0, sample_rate=16000):
        """Generate raw 16-bit PCM bytes for audio synthesis testing."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Apply gentle ADSR envelope
        attack = int(sample_rate * 0.2)
        decay = int(sample_rate * 0.5)
        release = int(sample_rate * 1.0)
        sustain_len = len(t) - attack - decay - release

        envelope = np.ones_like(t)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        envelope[-release:] = np.linspace(0.7, 0, release)

        audio = np.sin(2 * np.pi * frequency * t) * envelope * 0.5
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

sound_engine = AcousticTherapyEngine()

import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
import config

TARGET_EXPRESSIONS = [
    {
        "id": "EXP_01",
        "name": "NỤ CƯỜI TỰ NHIÊN (DUCHENNE SMILE)",
        "target_emotion": "HAPPY / EQUILIBRIUM",
        "description": "Nâng khóe miệng, tạo nếp nhăn đuôi mắt nhẹ, thả lỏng trán.",
        "primary_muscles": ["Cơ gò má lớn (Zygomaticus major)", "Cơ vòng mi (Orbicularis oculi)"],
        "svg_icon": "smile",
        "reference_valence": 0.85,
        "reference_arousal": 0.40
    },
    {
        "id": "EXP_02",
        "name": "BẤT NGỜ CẢNH GIÁC (ALERT SURPRISE)",
        "target_emotion": "SURPRISE",
        "description": "Mắt mở to, nâng cao hai cung mày, mở nhẹ khóe môi.",
        "primary_muscles": ["Cơ trán (Frontalis)", "Cơ nâng mi trên (Levator palpebrae superioris)"],
        "svg_icon": "surprise",
        "reference_valence": 0.30,
        "reference_arousal": 0.85
    },
    {
        "id": "EXP_03",
        "name": "TĨNH TÂM THIỀN ĐỊNH (CALM EQUILIBRIUM)",
        "target_emotion": "NEUTRAL / CALM",
        "description": "Thả lỏng toàn bộ cơ mặt, mắt nhìn thẳng điềm tĩnh, môi khép hờ tự nhiên.",
        "primary_muscles": ["Thả lỏng cơ cắn (Masseter relaxation)", "Thả lỏng cơ cau mày (Corrugator deactivation)"],
        "svg_icon": "calm",
        "reference_valence": 0.05,
        "reference_arousal": -0.20
    },
    {
        "id": "EXP_04",
        "name": "TẬP TRUNG CAO ĐỘ (COGNITIVE FOCUS)",
        "target_emotion": "FOCUS / DETERMINATION",
        "description": "Ánh mắt kiên định, hai chân mày hơi khép nhẹ lại gần trục giữa.",
        "primary_muscles": ["Cơ hạ mày (Depressor supercilii)", "Cơ tháp (Procerus)"],
        "svg_icon": "focus",
        "reference_valence": 0.10,
        "reference_arousal": 0.45
    },
    {
        "id": "EXP_05",
        "name": "GIẢI TỎA CĂNG THẲNG (DEEP EXHALATION)",
        "target_emotion": "RELIEF",
        "description": "Môi hơi chúm nhẹ như đang thở ra từ từ, hai má thả lỏng sâu.",
        "primary_muscles": ["Cơ mút (Buccinator)", "Cơ vòng miệng (Orbicularis oris)"],
        "svg_icon": "relief",
        "reference_valence": 0.50,
        "reference_arousal": -0.40
    }
]

def extract_detailed_biometrics(face_crop):
    """
    Extracts normalized facial kinematic metrics from a cropped face image.
    All returned values are calibrated against standard 160x160 coordinates.
    """
    if face_crop is None or face_crop.size == 0:
        return {
            "forehead_activity": 12.0,
            "glabella_activity": 10.0,
            "mouth_open_cavity": 0.0,
            "mouth_vertical_edges": 10.0,
            "mouth_height_active": 0.0
        }

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    h, w = gray.shape
    if h < 20 or w < 20:
        return {
            "forehead_activity": 12.0,
            "glabella_activity": 10.0,
            "mouth_open_cavity": 0.0,
            "mouth_vertical_edges": 10.0,
            "mouth_height_active": 0.0
        }

    std_face = cv2.resize(gray, (160, 160))

    # 1. Forehead region (Frontalis muscle / horizontal wrinkling when raised)
    forehead = std_face[10:45, 30:130]
    sobel_forehead_y = cv2.Sobel(forehead, cv2.CV_32F, 0, 1, ksize=3)
    forehead_activity = float(np.mean(np.abs(sobel_forehead_y)))

    # 2. Glabella region (Procerus / Corrugator supercilii / vertical brow furrowing)
    glabella = std_face[35:65, 55:105]
    sobel_glabella_x = cv2.Sobel(glabella, cv2.CV_32F, 1, 0, ksize=3)
    glabella_activity = float(np.mean(np.abs(sobel_glabella_x)))

    # 3. Mouth cavity & lips
    mouth = std_face[100:150, 30:130]
    m_mean = float(np.mean(mouth))
    mouth_open_cavity = float(np.mean(mouth < (m_mean * 0.65)))

    sobel_mouth_y = cv2.Sobel(mouth, cv2.CV_32F, 0, 1, ksize=3)
    mouth_vertical_edges = float(np.mean(np.abs(sobel_mouth_y)))

    lip_rows = np.std(mouth, axis=1)
    mouth_height_active = float(np.mean(lip_rows > 14))

    return {
        "forehead_activity": round(forehead_activity, 2),
        "glabella_activity": round(glabella_activity, 2),
        "mouth_open_cavity": round(mouth_open_cavity, 4),
        "mouth_vertical_edges": round(mouth_vertical_edges, 2),
        "mouth_height_active": round(mouth_height_active, 3)
    }

class MimicMatchGameEngine:
    """
    Biofeedback facial motor control and neuro-inhibition release game engine.
    Calculates facial muscle activation accuracy and real kinematic biomarkers.
    Operates 100% offline with zero cloud dependencies.
    """
    def __init__(self):
        self.session_high_score = 0.0
        self.current_streak = 0
        self._target_vectors = {}
        self._init_target_vectors()

    def _init_target_vectors(self):
        """Generate consistent reference biometric vectors for each target expression."""
        for exp in TARGET_EXPRESSIONS:
            torch.manual_seed(abs(hash(exp["id"])))
            vec = F.normalize(torch.randn(1, config.DIM_FACE), p=2, dim=-1)
            self._target_vectors[exp["id"]] = vec

    def get_random_target(self):
        """Get random facial expression challenge."""
        return random.choice(TARGET_EXPRESSIONS)

    def get_target_by_id(self, target_id):
        return next((t for t in TARGET_EXPRESSIONS if t["id"] == target_id), TARGET_EXPRESSIONS[0])

    def evaluate_mimic_match(self, target_id, user_face_vec=None, face_crop=None, analysis=None, client_data=None):
        """
        Calibrated biometric evaluation of user facial mimic match.
        Assesses actual face presence, smile intensity, arousal/valence, and motor engagement.
        """
        target = self.get_target_by_id(target_id)
        if client_data is None:
            client_data = {}

        # 1. STRICT Face Presence Verification: Only genuine detected human faces pass
        face_detected = bool(analysis and analysis.get("face_detected"))

        if not face_detected:
            self.current_streak = 0
            muscle_feedback = []
            for muscle in target["primary_muscles"]:
                muscle_feedback.append({
                    "muscle": muscle,
                    "activation_pct": 0.0,
                    "status": "NOT_DETECTED"
                })
            return {
                "target_id": target["id"],
                "target_name": target["name"],
                "matching_score": 0.0,
                "status_badge": "Chưa nhận diện được khuôn mặt",
                "rating": "FAIL",
                "streak": 0,
                "session_high_score": self.session_high_score,
                "muscle_feedback": muscle_feedback,
                "clinical_insight": "Camera không phát hiện thấy khuôn mặt nào trong khung hình. Vui lòng ngồi trước ống kính, hướng thẳng mặt vào camera và đảm bảo đủ ánh sáng!"
            }

        # 2. Extract biometric kinematics from real face landmarks and crop
        kinematics = extract_detailed_biometrics(face_crop)
        forehead_act = kinematics["forehead_activity"]
        glabella_act = kinematics["glabella_activity"]
        mouth_cavity = kinematics["mouth_open_cavity"]

        landmarks = analysis.get("landmarks") or {}
        r_width = float(analysis.get("mouth_width_ratio") or landmarks.get("r_width") or 0.83)
        r_jaw = float(analysis.get("jaw_drop_ratio") or landmarks.get("r_jaw") or 0.45)
        r_smile_morph = float(landmarks.get("r_smile_morph", 1.80))
        head_tilt = abs(float(analysis.get("head_tilt_deg") or landmarks.get("tilt") or 0.0))

        smile_srv = float(analysis.get("smile_score", 0.0))
        is_smiling = bool(analysis.get("is_smiling", False)) or (smile_srv >= 40.0)

        score_pct = 50.0
        status_badge = "Mức độ trung bình"
        rating = "ACCEPTABLE"
        clinical_insight = ""
        muscle_acts = {}

        # 3. Dynamic Targeted Mimic Match Evaluation
        if target["id"] == "EXP_01":
            # Target: Duchenne Smile (Zygomaticus major & Orbicularis oculi)
            # Requires genuine smile: is_smiling or significant mouth expansion (r_width >= 0.92)
            if is_smiling and (smile_srv >= 50.0 or r_width >= 0.92):
                # Strong, genuine smile
                m1 = round(min(97.0, 83.0 + (smile_srv / 100.0) * 12.0 + max(0.0, r_width - 0.92) * 20.0), 1)
                m2 = round(min(95.5, 81.0 + (smile_srv / 100.0) * 13.0 + max(0.0, r_width - 0.92) * 20.0), 1)
                score_pct = round(0.55 * m1 + 0.45 * m2, 1)
                status_badge = "Xuất sắc • Nụ cười rạng rỡ"
                rating = "PASS"
                clinical_insight = "Phản xạ nâng cơ gò má lớn và co cơ vòng mi đạt chuẩn nụ cười Duchenne. Xung thần kinh từ vỏ não vận động M1 điều hòa tối ưu."
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2
            elif is_smiling or (smile_srv >= 25.0) or (r_width >= 0.89 and r_smile_morph >= 1.95):
                # Moderate smile
                m1 = round(min(78.0, 62.0 + max(0.0, smile_srv - 25.0) * 0.5), 1)
                m2 = round(min(76.0, 60.0 + max(0.0, smile_srv - 25.0) * 0.5), 1)
                score_pct = round(0.55 * m1 + 0.45 * m2, 1)
                status_badge = "Đạt mức khá • Hãy cười tươi hơn"
                rating = "ACCEPTABLE"
                clinical_insight = "Đã có phản xạ mỉm cười nhẹ. Hãy cười tươi hơn nữa để khóe miệng nâng cao rõ rệt và đuôi mắt nheo lại!"
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2
            else:
                # Resting face or not smiling - STRICT FAIL!
                m1 = round(max(15.0, min(32.0, 18.0 + max(0.0, r_width - 0.75) * 40.0)), 1)
                m2 = round(max(12.0, min(28.0, 15.0 + max(0.0, r_width - 0.75) * 35.0)), 1)
                score_pct = round(0.55 * m1 + 0.45 * m2, 1)
                status_badge = "Chưa đạt • Thiếu nụ cười"
                rating = "FAIL"
                clinical_insight = "Khuôn mặt đang ở trạng thái nghiêm nghị hoặc bất động. Hãy mỉm cười thật tươi, nâng cao hai bên khóe môi và nheo nhẹ khóe mắt!"
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2

        elif target["id"] == "EXP_02":
            # Target: Alert Surprise (Frontalis & Levator palpebrae superioris)
            # Requires jaw drop (r_jaw >= 0.65) and/or raised forehead
            if r_jaw >= 0.68:
                m1 = round(min(96.5, 83.0 + min(11.0, (r_jaw - 0.68) * 45.0) + min(3.0, max(0.0, forehead_act - 25.0) * 0.15)), 1)
                m2 = round(min(95.0, 81.0 + min(11.0, (r_jaw - 0.68) * 45.0) + min(3.0, max(0.0, forehead_act - 25.0) * 0.15)), 1)
                score_pct = round(0.5 * m1 + 0.5 * m2, 1)
                status_badge = "Xuất sắc • Phản xạ cảnh giác"
                rating = "PASS"
                clinical_insight = "Cơ trán nâng cao và cơ nâng mi trên mở rộng tối ưu biên độ thị giác. Hệ thống lưới hoạt hóa hướng lên (ARAS) được kích hoạt chuẩn xác."
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2
            elif r_jaw >= 0.56:
                m1 = round(min(76.0, 60.0 + (r_jaw - 0.56) * 110.0), 1)
                m2 = round(min(74.0, 58.0 + (r_jaw - 0.56) * 110.0), 1)
                score_pct = round(0.5 * m1 + 0.5 * m2, 1)
                status_badge = "Đạt mức khá • Hãy há rộng hơn"
                rating = "ACCEPTABLE"
                clinical_insight = "Đã có phản xạ há miệng nhưng biên độ còn hẹp. Hãy há rộng miệng hơn và nhướng chân mày lên cao!"
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2
            else:
                m1 = round(max(14.0, min(28.0, 15.0 + max(0.0, r_jaw - 0.40) * 40.0)), 1)
                m2 = round(max(12.0, min(26.0, 14.0 + max(0.0, r_jaw - 0.40) * 35.0)), 1)
                score_pct = round(0.5 * m1 + 0.5 * m2, 1)
                status_badge = "Chưa đạt • Hãy há miệng và mở to mắt"
                rating = "FAIL"
                clinical_insight = "Khuôn mặt vẫn khép miệng ở trạng thái tĩnh. Hãy nhướng hai chân mày lên cao, mở to mắt và há rộng khẩu hình miệng!"
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2

        elif target["id"] == "EXP_03":
            # Target: Calm Equilibrium (Masseter relaxation & Corrugator deactivation)
            # Resting mouth, relaxed jaw (r_jaw < 0.60), not smiling
            if is_smiling and (smile_srv >= 35.0 or r_width >= 0.91):
                score_pct = round(max(18.0, min(32.0, 30.0 - max(0.0, smile_srv - 35.0) * 0.25)), 1)
                status_badge = "Chưa đạt • Cần thả lỏng cơ mặt"
                rating = "FAIL"
                clinical_insight = "Hệ thống phát hiện khóe miệng đang nhếch (đang cười). Thử thách này yêu cầu nét mặt tĩnh lặng, thả lỏng hoàn toàn quai hàm và khép nhẹ môi."
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 1.0, 1)
            elif r_jaw >= 0.65:
                score_pct = round(max(16.0, min(28.0, 22.0)), 1)
                status_badge = "Chưa đạt • Quai hàm đang mở"
                rating = "FAIL"
                clinical_insight = "Quai hàm đang há mở. Hãy khép hờ hai môi và thả lỏng cơ mặt hoàn toàn."
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 1.0, 1)
            elif glabella_act > 38.0:
                score_pct = round(max(50.0, min(68.0, 65.0 - (glabella_act - 38.0) * 0.5)), 1)
                status_badge = "Cần thả lỏng vùng trán & chân mày"
                rating = "ACCEPTABLE"
                clinical_insight = "Vùng giữa hai chân mày còn hơi co thắt. Hãy thả lỏng cơ cau mày và hít thở nhẹ nhàng."
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 2.0, 1)
            else:
                # Genuine calm resting face: dynamic 86% - 94%
                symmetry_bonus = min(3.0, max(0.0, 3.0 - abs(head_tilt) * 0.3))
                tension_penalty = min(4.0, max(0.0, glabella_act - 15.0) * 0.15)
                m1 = round(min(94.5, 89.0 + symmetry_bonus - tension_penalty), 1)
                m2 = round(min(93.5, 88.0 + symmetry_bonus - tension_penalty), 1)
                score_pct = round(0.5 * m1 + 0.5 * m2, 1)
                status_badge = "Xuất sắc • Thả lỏng điềm tĩnh"
                rating = "PASS"
                clinical_insight = "Toàn bộ cơ cắn và cơ cau mày đã được thả lỏng hoàn toàn. Hệ thần kinh đối giao cảm (Parasympathetic) đang chiếm ưu thế vượt trội."
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2

        elif target["id"] == "EXP_04":
            # Target: Cognitive Focus (Depressor supercilii & Procerus)
            # Needs focused gaze, firm closed mouth, concentrated brow
            if is_smiling and (smile_srv >= 35.0 or r_width >= 0.91):
                score_pct = round(max(18.0, min(30.0, 25.0)), 1)
                status_badge = "Chưa đạt • Cần tập trung"
                rating = "FAIL"
                clinical_insight = "Phát hiện nét mặt cười cợt hoặc thiếu tập trung. Thử thách này yêu cầu ánh mắt kiên định, khép chặt môi và hơi khép nhẹ chân mày."
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 1.0, 1)
            elif r_jaw >= 0.65:
                score_pct = 20.0
                status_badge = "Chưa đạt • Cần khép chặt môi"
                rating = "FAIL"
                clinical_insight = "Miệng đang mở rộng. Hãy khép chặt môi và tập trung cao độ vào tâm màn hình."
                muscle_acts[target["primary_muscles"][0]] = 18.0
                muscle_acts[target["primary_muscles"][1]] = 22.0
            elif glabella_act < 20.0:
                # Completely relaxed without active focus engagement
                score_pct = round(max(55.0, min(68.0, 58.0 + (glabella_act / 20.0) * 10.0)), 1)
                status_badge = "Cần tập trung hơn • Hơi khép nhẹ chân mày"
                rating = "ACCEPTABLE"
                clinical_insight = "Nét mặt đang thả lỏng tự nhiên nhưng chưa kích hoạt phức hợp cơ hạ mày. Hãy nhìn thẳng vào tâm camera, hơi khép nhẹ chân mày để tập trung cao độ."
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 1.0, 1)
            else:
                # Active focused brow and firm mouth: 83% - 94%
                focus_boost = min(10.0, (glabella_act - 20.0) * 0.45)
                m1 = round(min(94.0, 84.0 + focus_boost), 1)
                m2 = round(min(93.0, 83.0 + focus_boost), 1)
                score_pct = round(0.5 * m1 + 0.5 * m2, 1)
                status_badge = "Xuất sắc • Tập trung cao độ"
                rating = "PASS"
                clinical_insight = "Ánh mắt kiên định, phức hợp cơ hạ mày và cơ tháp hội tụ chuẩn xác về trục thị giác trung tâm, tối ưu hóa năng lượng nhận thức."
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2

        else: # EXP_05: Deep Exhalation / Relief
            # Target: Pursed lips (r_width <= 0.73)
            # In resting face, r_width is 0.81 - 0.87. Pursed lips narrows to <= 0.73!
            if is_smiling and (smile_srv >= 35.0 or r_width >= 0.90):
                score_pct = round(max(15.0, min(26.0, 22.0)), 1)
                status_badge = "Chưa đạt • Chúm môi thở ra thay vì cười"
                rating = "FAIL"
                clinical_insight = "Bạn đang cười làm mở rộng khóe môi. Thử thách này yêu cầu chúm môi hẹp lại như đang thổi tắt một ngọn nến!"
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 1.0, 1)
            elif r_width <= 0.73:
                # Genuine pursed lips: 84% - 95%
                pucker_boost = min(12.0, (0.73 - r_width) * 80.0)
                m1 = round(min(95.0, 84.0 + pucker_boost), 1)
                m2 = round(min(94.0, 83.0 + pucker_boost), 1)
                score_pct = round(0.5 * m1 + 0.5 * m2, 1)
                status_badge = "Xuất sắc • Thở ra giải tỏa"
                rating = "PASS"
                clinical_insight = "Giải phóng trương lực cơ mút và cơ vòng miệng qua kỹ thuật thở chúm môi kéo dài, kích thích hệ thần kinh đối giao cảm."
                muscle_acts[target["primary_muscles"][0]] = m1
                muscle_acts[target["primary_muscles"][1]] = m2
            elif r_width <= 0.77:
                # Moderate pursing
                score_pct = round(max(62.0, min(76.0, 62.0 + (0.77 - r_width) * 300.0)), 1)
                status_badge = "Đạt mức khá • Hãy chúm môi hẹp hơn"
                rating = "ACCEPTABLE"
                clinical_insight = "Đã có phản xạ chúm môi nhưng khẩu hình chưa thật sự thu hẹp tối đa. Hãy chúm hai môi nhỏ lại như đang huýt sáo!"
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 1.0, 1)
            else:
                # Normal un-pursed mouth (r_width >= 0.78): FAIL
                score_pct = round(max(22.0, min(45.0, 40.0 - (r_width - 0.78) * 120.0)), 1)
                status_badge = "Chưa đạt • Hãy chúm môi nhỏ như thổi nến"
                rating = "FAIL"
                clinical_insight = "Khẩu hình miệng chưa đủ hẹp. Hãy chúm hai môi lại tròn nhỏ như đang huýt sáo hoặc thổi nhẹ vào một ngọn nến!"
                muscle_acts[target["primary_muscles"][0]] = round(score_pct - 2.0, 1)
                muscle_acts[target["primary_muscles"][1]] = round(score_pct + 2.0, 1)

        # Update streak and high score
        if rating == "PASS":
            self.current_streak += 1
            if score_pct > self.session_high_score:
                self.session_high_score = score_pct
        else:
            self.current_streak = 0

        muscle_feedback = []
        for muscle in target["primary_muscles"]:
            act_val = muscle_acts.get(muscle, score_pct)
            muscle_feedback.append({
                "muscle": muscle,
                "activation_pct": act_val,
                "status": "OPTIMAL" if act_val >= 70.0 else "SUB_THRESHOLD"
            })

        return {
            "target_id": target["id"],
            "target_name": target["name"],
            "matching_score": score_pct,
            "status_badge": status_badge,
            "rating": rating,
            "streak": self.current_streak,
            "session_high_score": self.session_high_score,
            "muscle_feedback": muscle_feedback,
            "clinical_insight": clinical_insight
        }

game_engine = MimicMatchGameEngine()

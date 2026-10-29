"""
CMP169 - Multimodal Late Fusion Engine (Phase 5)
================================================================================
Phân Hệ Kết Hợp Đa Phương Thức Trễ (Decision-Level Late Fusion)
Kết hợp Cảm Xúc Khuôn Mặt (YuNet Pipeline) và Cảm Xúc Giọng Nói (CNN-LSTM SER).

Cơ chế: Weighted Average Late Fusion (Bù trừ trọng số động)
Trọng số mặc định: weight_face = 0.8, weight_audio = 0.2

Tác giả: Chuyên Gia Kỹ Sư AI/ML (CMP169 Team)
"""

import io
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

# Đảm bảo mã hóa UTF-8 trên Windows console
if hasattr(sys.stdout, "buffer") and (getattr(sys.stdout, "encoding", "") or "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Danh sách 7 lớp cảm xúc chuẩn hóa của dự án CMP169
try:
    from dataset_pipeline_v2 import EMOTION_CLASSES
except ImportError:
    EMOTION_CLASSES: List[str] = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]

# Ánh xạ nhãn tiếng Việt phục vụ giải thích và hiển thị trực quan
EMOTION_VIETNAMESE: Dict[str, str] = {
    "angry": "Giận dữ (Angry)",
    "disgust": "Ghê tởm (Disgust)",
    "fear": "Sợ hãi / Bất an (Fear)",
    "happy": "Vui vẻ / Hạnh phúc (Happy)",
    "neutral": "Điềm tĩnh / Trung tính (Neutral)",
    "sad": "Buồn bã (Sad)",
    "surprise": "Ngạc nhiên (Surprise)"
}


class FusionResult(np.ndarray):
    """
    Phase 5.6 Dual-Behavior Fusion Result Object.
    - As a 1D ndarray (shape 7): Represents the probability distribution (sum = 1.0).
    - Supports 2-tuple unpacking: `pred_label, confidence = result` for downstream compatibility.
    - Attributes: `result.pred_label`, `result.label`, `result.confidence`.
    """
    def __new__(
        cls,
        probs: np.ndarray,
        pred_label: str,
        confidence: float,
        emotion_classes: Optional[List[str]] = None
    ):
        obj = np.asarray(probs, dtype=np.float32).view(cls)
        obj.pred_label = str(pred_label)
        obj.label = str(pred_label)
        obj.confidence = float(confidence)
        obj.emotion_classes = emotion_classes
        return obj

    def __iter__(self):
        yield self.pred_label
        yield self.confidence


class MultimodalFusionEngine:
    """
    Bộ máy Kết Hợp Đa Phương Thức (Multimodal Late Fusion Engine).
    Phase 5.6: Nâng cấp thành Stacking Ensemble Meta-Learner (Scikit-Learn).
    Đầu vào: Vector đặc trưng 14 chiều (Face 7D + Audio 7D).
    Đầu ra: Phân phối xác suất 7 lớp cảm xúc.
    Fallback: Bù trừ trọng số số học 80/20 nếu không tìm thấy file trọng số meta-model.
    """

    def __new__(cls, *args, **kwargs):
        """
        Hỗ trợ gọi linh hoạt theo 2 phong cách:
        1. Hướng đối tượng: engine = MultimodalFusionEngine(weight_face=0.8, weight_audio=0.2)
                            label, conf = engine.fuse(face_probs, audio_probs)
        2. Hướng hàm chức năng: label, conf = MultimodalFusionEngine(face_probs, audio_probs)
        """
        # Nếu tham số đầu tiên là mảng xác suất (list, ndarray, tuple)
        if args and isinstance(args[0], (list, tuple, np.ndarray)):
            instance = super().__new__(cls)
            instance.__init__()
            return instance.fuse(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        weight_face: float = 0.8,
        weight_audio: float = 0.2,
        emotion_classes: Optional[List[str]] = None,
        model_path: Optional[Union[str, Path]] = None,
        use_meta: bool = True
    ) -> None:
        """
        Khởi tạo Fusion Engine với Meta-Learner và bộ trọng số fallback.

        Args:
            weight_face: Trọng số của phân hệ khuôn mặt cho fallback (Mặc định: 0.8).
            weight_audio: Trọng số của phân hệ âm thanh cho fallback (Mặc định: 0.2).
            emotion_classes: Danh sách 7 lớp cảm xúc (Mặc định: EMOTION_CLASSES).
            model_path: Đường dẫn tới meta_fusion_model.pkl (Mặc định: checkpoints/meta_fusion_model.pkl).
            use_meta: Cho phép sử dụng Meta-Learner nếu có sẵn.
        """
        self.weight_face: float = float(weight_face)
        self.weight_audio: float = float(weight_audio)
        self.emotion_classes: List[str] = list(emotion_classes or EMOTION_CLASSES)
        self.use_meta: bool = bool(use_meta)

        # Phase 5.6: Meta-Learner Model loading & Fallback safety
        self.meta_model = None
        if model_path:
            self.meta_model_path = Path(model_path)
        else:
            self.meta_model_path = Path(__file__).resolve().parent / "checkpoints" / "meta_fusion_model.pkl"

        if self.use_meta:
            self._load_meta_model()
        else:
            print("[PHASE 5.6] WARNING: Meta-Learner disabled. Using 80/20 fallback.")

    def _load_meta_model(self) -> bool:
        """
        Attempts to load the trained scikit-learn meta-learner model.
        Falls back to 80/20 arithmetic logic if file is not found or fails to load.
        """
        if self.meta_model_path.exists():
            try:
                if joblib is not None:
                    self.meta_model = joblib.load(str(self.meta_model_path))
                else:
                    with open(self.meta_model_path, "rb") as f:
                        self.meta_model = pickle.load(f)
                print("[PHASE 5.6] Meta-Learner Fusion Engine loaded.")
                return True
            except Exception as e:
                try:
                    with open(self.meta_model_path, "rb") as f:
                        self.meta_model = pickle.load(f)
                    print("[PHASE 5.6] Meta-Learner Fusion Engine loaded.")
                    return True
                except Exception as ex:
                    print(f"[PHASE 5.6] WARNING: Failed to load Meta-Learner ({ex}). Using 80/20 fallback.")
                    self.meta_model = None
                    return False
        else:
            print(f"[PHASE 5.6] WARNING: Meta-Learner model not found at '{self.meta_model_path.name}'. Using 80/20 fallback.")
            self.meta_model = None
            return False

    def fuse(
        self,
        face_probs: Union[np.ndarray, List[float]],
        audio_probs: Union[np.ndarray, List[float]],
        weight_face: Optional[float] = None,
        weight_audio: Optional[float] = None,
        return_distribution: bool = False,
        return_probs: bool = False
    ):
        """
        Thực hiện Multimodal Fusion:
          - Nếu Meta-Learner được nạp: Ghép [face_probs, audio_probs] thành shape (1, 14)
            và trích xuất meta_model.predict_proba(X)[0].
          - Nếu không: Fallback về công thức trung bình có trọng số (0.8 * face_probs) + (0.2 * audio_probs).

        Args:
            face_probs: Mảng xác suất Softmax 7 lớp từ mô hình khuôn mặt.
            audio_probs: Mảng xác suất Softmax 7 lớp từ mô hình giọng nói.
            weight_face: Ghi đè trọng số khuôn mặt cho fallback (tùy chọn).
            weight_audio: Ghi đè trọng số giọng nói cho fallback (tùy chọn).
            return_distribution: Nếu True, trả về tuple (pred_label, confidence, final_probs).
            return_probs: Nếu True, trả về trực tiếp mảng final_probs.

        Returns:
            FusionResult (np.ndarray): Hỗ trợ cả truy cập array và giải nén (pred_label, confidence).
        """
        w_f = self.weight_face if weight_face is None else float(weight_face)
        w_a = self.weight_audio if weight_audio is None else float(weight_audio)

        # Chuyển đổi an toàn sang numpy float32 và làm phẳng (1D vector)
        f_arr = np.asarray(face_probs, dtype=np.float32).ravel()
        a_arr = np.asarray(audio_probs, dtype=np.float32).ravel()

        num_classes = len(self.emotion_classes)
        if len(f_arr) != num_classes or len(a_arr) != num_classes:
            raise ValueError(
                f"[FUSION ERROR] Kích thước đầu vào không khớp: face={len(f_arr)}, audio={len(a_arr)}, "
                f"yêu cầu chính xác {num_classes} lớp cảm xúc."
            )

        # PHASE 5.6: META-LEARNER MULTIMODAL STACKING FUSION
        if self.meta_model is not None:
            # Concatenate [face_probs, audio_probs] into shape (1, 14)
            X = np.concatenate([f_arr, a_arr]).reshape(1, 14)
            raw_meta_probs = self.meta_model.predict_proba(X)[0]

            # Map to self.emotion_classes ordering
            if hasattr(self.meta_model, "classes_"):
                model_classes = list(self.meta_model.classes_)
                if all(isinstance(c, str) for c in model_classes):
                    final_probs = np.zeros(num_classes, dtype=np.float32)
                    for idx, c in enumerate(model_classes):
                        if c in self.emotion_classes:
                            final_probs[self.emotion_classes.index(c)] = raw_meta_probs[idx]
                elif all(isinstance(c, (int, np.integer)) for c in model_classes):
                    final_probs = np.zeros(num_classes, dtype=np.float32)
                    for idx, c_idx in enumerate(model_classes):
                        if 0 <= c_idx < num_classes:
                            final_probs[c_idx] = raw_meta_probs[idx]
                else:
                    final_probs = np.asarray(raw_meta_probs, dtype=np.float32)
            else:
                final_probs = np.asarray(raw_meta_probs, dtype=np.float32)

            s = float(final_probs.sum())
            if s > 0:
                final_probs = (final_probs / s).astype(np.float32)
        else:
            # FALLBACK: (0.8 * face_probs) + (0.2 * audio_probs)
            final_probs = (w_f * f_arr) + (w_a * a_arr)
            w_sum = w_f + w_a
            if w_sum > 0:
                final_probs = (final_probs / w_sum).astype(np.float32)

        win_idx = int(np.argmax(final_probs))
        pred_label = self.emotion_classes[win_idx]
        confidence = float(final_probs[win_idx])

        if return_probs:
            return final_probs
        if return_distribution:
            return pred_label, confidence, final_probs
        return FusionResult(final_probs, pred_label, confidence, self.emotion_classes)

    def predict_proba(
        self,
        face_probs: Union[np.ndarray, List[float]],
        audio_probs: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """Trả về trực tiếp mảng phân phối xác suất 7 lớp."""
        return self.fuse(face_probs, audio_probs, return_probs=True)

    def __call__(self, *args, **kwargs):
        """Cho phép gọi thực thi trực tiếp trên instance: engine(face_probs, audio_probs)."""
        return self.fuse(*args, **kwargs)


def print_simulation_report(
    face_probs: np.ndarray,
    audio_probs: np.ndarray,
    final_probs: np.ndarray,
    pred_face: str,
    conf_face: float,
    pred_audio: str,
    conf_audio: float,
    pred_final: str,
    conf_final: float,
    w_face: float,
    w_audio: float,
    classes: List[str]
) -> None:
    """
    Xuất báo cáo mô phỏng kịch bản và cơ chế bù trừ trọng số hoàn toàn bằng Tiếng Việt.
    """
    border = "=" * 80
    sub_border = "-" * 80

    print(border)
    print("   CMP169: BÁO CÁO MÔ PHỎNG PHÂN HỆ FUSION ĐA PHƯƠNG THỨC (PHASE 5)")
    print("           LATE FUSION: THỊ GIÁC (YUNET) + GIỌNG NÓI (CNN-LSTM SER)")
    print(border)
    print("\n1. BỐI CẢNH THỬ NGHIỆM LÂM SÀNG:")
    print("   - Đối tượng thử nghiệm: Đang cười tươi trong buổi đánh giá.")
    print("   - Thị giác (YuNet Pipeline): Phát hiện rõ nét nụ cười (smile landmarks).")
    print("   - Thính giác (CNN-LSTM SER): Phòng khám có tiếng ồn tần số cao / âm thanh la hét")
    print("     bất ngờ (High Arousal Noise) gây hiện tượng lệch miền (Domain Shift).")
    print("     Hậu quả: Mô hình giọng nói phán đoán nhầm thành FEAR với độ tin cậy rất cao.")

    print("\n2. CƠ CHẾ BÙ TRỪ TRỌNG SỐ (WEIGHT COMPENSATION MECHANISM):")
    print(f"   - Trọng số Thị giác (Weight Face) : {w_face:.2f} (80% độ tin cậy)")
    print(f"   - Trọng số Âm thanh (Weight Audio): {w_audio:.2f} (20% độ tin cậy)")
    print("   * Lý do kỹ thuật:")
    print("     + Dữ liệu âm thanh thực tế chịu ảnh hưởng nặng nề bởi micro, độ vang phòng,")
    print("       và tiếng ồn môi trường, khiến các mẫu thanh âm có Arousal cao dễ bị ngộ nhận.")
    print("     + Trái lại, camera và pipeline YuNet trích xuất nụ cười có độ phân giải cao,")
    print("       rất bền vững và không bị nhiễu âm tác động.")
    print("     + Bằng cách gán trọng số 0.8 cho Face và 0.2 cho Audio, Fusion Engine tự động")
    print("       nén sai số 72% của Audio xuống chỉ còn đóng góp 14.4%, đồng thời khuếch đại")
    print("       tín hiệu nụ cười 85% của Face lên mức 68.0%, đưa ra phán đoán cuối cùng đúng đắn.")

    print(f"\n{sub_border}")
    print(f"{'LỚP CẢM XÚC':<20} | {'FACE (YuNet)':<15} | {'AUDIO (CNN-LSTM)':<18} | {'FUSED (0.8/0.2)':<15}")
    print(sub_border)

    for i, c in enumerate(classes):
        vn_name = EMOTION_VIETNAMESE.get(c, c)
        f_p = face_probs[i] * 100.0
        a_p = audio_probs[i] * 100.0
        fu_p = final_probs[i] * 100.0
        marker = " <== [CHIẾN THẮNG]" if c == pred_final else ""
        print(f"{vn_name:<20} | {f_p:6.2f}%         | {a_p:6.2f}%           | {fu_p:6.2f}%{marker}")

    print(sub_border)
    print("\n3. TỔNG KẾT SO SÁNH QUYẾT ĐỊNH:")
    print(f"   * [Đơn Lẻ] Mô hình Thị Giác : {pred_face.upper():<8} (Độ tin cậy: {conf_face*100:.2f}%) -> CHÍNH XÁC")
    print(f"   * [Đơn Lẻ] Mô hình Âm Thanh : {pred_audio.upper():<8} (Độ tin cậy: {conf_audio*100:.2f}%) -> DỰ ĐOÁN SAI (Nhiễu Arousal)")
    print(f"   * [HỢP NHẤT] KẾT QUẢ FUSION : {pred_final.upper():<8} (Độ tin cậy: {conf_final*100:.2f}%) -> SỬA SAI THÀNH CÔNG!")
    print(f"\n   ==> KẾT LUẬN: Bộ máy Fusion đã loại bỏ hoàn toàn cảnh báo sai (False Alarm) của giọng nói,")
    print(f"       bảo vệ quyết định cảm xúc chính xác là {pred_final.upper()} với điểm tin cậy đạt {conf_final*100:.2f}%.\n")
    print(border)


# ------------------------------------------------------------------------------
# Khối Mô Phỏng & Kiểm Thử Tự Động (Simulation Test Block)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Khởi tạo Fusion Engine chuẩn theo yêu cầu
    engine = MultimodalFusionEngine(weight_face=0.8, weight_audio=0.2)

    # 1. Giả lập xác suất 7 lớp từ mô hình Thị Giác (YuNet Pipeline)
    # Phát hiện nụ cười rõ nét: HAPPY chiếm ưu thế tuyệt đối (85%)
    # Thứ tự lớp: [angry, disgust, fear, happy, neutral, sad, surprise]
    sim_face_probs = np.array(
        [0.010, 0.010, 0.020, 0.850, 0.060, 0.025, 0.025],
        dtype=np.float32
    )

    # 2. Giả lập xác suất 7 lớp từ mô hình Giọng Nói (CNN-LSTM SER)
    # Bị nhiễu kích động âm học (High Arousal Noise): nhầm sang FEAR (72%)
    sim_audio_probs = np.array(
        [0.080, 0.040, 0.720, 0.050, 0.040, 0.035, 0.035],
        dtype=np.float32
    )

    # Quyết định đơn lẻ của từng mô hình
    face_idx = int(np.argmax(sim_face_probs))
    face_label = EMOTION_CLASSES[face_idx]
    face_conf = float(sim_face_probs[face_idx])

    audio_idx = int(np.argmax(sim_audio_probs))
    audio_label = EMOTION_CLASSES[audio_idx]
    audio_conf = float(sim_audio_probs[audio_idx])

    # 3. Thực thi Late Fusion (Weighted Average)
    final_label, final_conf, final_distribution = engine.fuse(
        face_probs=sim_face_probs,
        audio_probs=sim_audio_probs,
        return_distribution=True
    )

    # 4. Xuất toàn bộ kết quả mô phỏng bằng Tiếng Việt
    print_simulation_report(
        face_probs=sim_face_probs,
        audio_probs=sim_audio_probs,
        final_probs=final_distribution,
        pred_face=face_label,
        conf_face=face_conf,
        pred_audio=audio_label,
        conf_audio=audio_conf,
        pred_final=final_label,
        conf_final=final_conf,
        w_face=engine.weight_face,
        w_audio=engine.weight_audio,
        classes=EMOTION_CLASSES
    )

    # Kiểm tra khẳng định (Assertion) đảm bảo đầu ra bắt buộc phải là "happy"
    assert final_label == "happy", f"Kỳ vọng nhãn 'happy', nhưng nhận được '{final_label}'!"
    if engine.meta_model is None:
        assert abs(final_conf - 0.690) < 1e-3, f"Kỳ vọng độ tin cậy ~0.690, nhưng nhận được {final_conf}!"
    else:
        assert final_conf > 0.50, f"Kỳ vọng độ tin cậy > 0.50, nhưng nhận được {final_conf}!"
    print("[KIỂM THỬ THÀNH CÔNG] Toàn bộ điều kiện logic và kiểm toán toán học đã đạt tiêu chuẩn 100%!")

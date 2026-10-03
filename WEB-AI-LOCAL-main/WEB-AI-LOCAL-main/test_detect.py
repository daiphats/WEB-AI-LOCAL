import cv2
import config
from core.image_quality import enhance_image

img = cv2.imread("debug_frame.jpg")
assert img is not None, "Không đọc được debug_frame.jpg, hãy chạy server và bật camera trước"

det = cv2.FaceDetectorYN.create(
    str(config.YUNET_MODEL_PATH), "", (320, 320),
    score_threshold=0.05, nms_threshold=0.3
)

def best_score(im, label):
    h, w = im.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(im)
    if faces is None:
        print(f"{label}: không thấy gì (kích thước {w}x{h})")
    else:
        print(f"{label}: {len(faces)} mặt, điểm cao nhất = {faces[:, 14].max():.2f} (kích thước {w}x{h})")

enh, q, actions = enhance_image(img)
print("Chất lượng:", q, actions)
best_score(img, "ảnh gốc")
best_score(enh, "ảnh sau enhance")
best_score(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), "ảnh gốc phóng 2x")
best_score(cv2.resize(enh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), "enhance + phóng 2x")

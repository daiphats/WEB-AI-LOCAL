import cv2
import numpy as np
import config


def assess_quality(img_bgr):
    """Đo độ nét, độ sáng, tỉ lệ cháy sáng/quá tối."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "brightness": float(gray.mean()),
        "over_ratio": float((gray > 250).mean()),
        "under_ratio": float((gray < 30).mean()),
    }


def _apply_gamma(img, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def _clahe_on_lightness(img_bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _unsharp(img, amount=1.5, sigma=1.2):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def _auto_gamma(img_bgr, target=config.TARGET_BRIGHTNESS):
    """Chọn gamma sao cho độ sáng trung bình tiến gần target."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = max(gray.mean(), 1.0) / 255.0
    gamma = np.log(target / 255.0) / np.log(mean)   # <1 làm sáng, >1 làm tối
    return float(np.clip(gamma, 0.4, 2.2))


def enhance_image(img_bgr):
    """
    Tự thích ứng ánh sáng + độ nét.
    Trả về (ảnh đã xử lý, thông tin chất lượng, trạng thái).
    """
    q = assess_quality(img_bgr)
    out = img_bgr
    actions = []

    # 1) Ánh sáng
    if q["brightness"] < config.LOW_LIGHT_MEAN or q["under_ratio"] > config.UNDEREXPOSED_RATIO:
        out = _apply_gamma(out, _auto_gamma(out))
        out = _clahe_on_lightness(out, clip=3.0)
        actions.append("low_light_boost")
    elif q["brightness"] > config.HIGH_LIGHT_MEAN or q["over_ratio"] > config.OVEREXPOSED_RATIO:
        out = _apply_gamma(out, _auto_gamma(out))
        out = _clahe_on_lightness(out, clip=1.5)
        actions.append("glare_reduce")
    else:
        out = _clahe_on_lightness(out, clip=1.5)   # cân bằng nhẹ cho ánh sáng không đều
        actions.append("light_normalize")

    # 2) Độ nét
    if q["sharpness"] < config.BLUR_THRESHOLD:
        out = _unsharp(out, amount=1.5)
        actions.append("sharpen")

    # 3) Nếu quá tối thì khử nhiễu (tăng sáng làm nhiễu nổi lên)
    if "low_light_boost" in actions:
        out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)

    return out, q, actions


def is_usable(face_bgr):
    """Quality gate trên vùng mặt đã crop. Trả về (ok, lý do)."""
    h, w = face_bgr.shape[:2]
    if min(h, w) < config.MIN_FACE_SIZE:
        return False, "face_too_small"
    q = assess_quality(face_bgr)
    if q["sharpness"] < config.BLUR_REJECT_THRESHOLD:
        return False, "too_blurry"
    if q["over_ratio"] > 0.5:
        return False, "overexposed"
    if q["under_ratio"] > 0.7:
        return False, "too_dark"
    return True, "ok"
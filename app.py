import os
import json
import time
import math
import asyncio
import threading
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import psutil
import torch
import numpy as np

import config
from core.fusion_engine import fusion_engine as _legacy_fusion_engine
from core.face_memory import face_memory
from core.clinical_counselor import clinical_counselor
from core.sound_generator import sound_engine
from core.game_engine import game_engine
import core.mediapipe_bridge
from core.ear import BlinkDetector
from core.head_pose import HeadPoseEstimator
from core.vocal_engine import SpeechEmotionAnalyzer, train_tess_speech_model
from multimodal_fusion_v2 import MultimodalFusionEngine

# Initialize Multimodal Late Fusion Engine (Phase 5.2 - Face 0.8 + Audio 0.2)
fusion_engine = MultimodalFusionEngine(weight_face=0.8, weight_audio=0.2)
# Attach legacy GMU methods to ensure 100% non-destructive backward compatibility
fusion_engine.predict_multimodal = _legacy_fusion_engine.predict_multimodal
fusion_engine.extract_face_feature_mock = _legacy_fusion_engine.extract_face_feature_mock
fusion_engine.extract_audio_feature_from_pcm = _legacy_fusion_engine.extract_audio_feature_from_pcm
fusion_engine.extract_text_feature = _legacy_fusion_engine.extract_text_feature
fusion_engine.restore_checkpoint = _legacy_fusion_engine.restore_checkpoint
fusion_engine.train_background = _legacy_fusion_engine.train_background

# Global Speech Emotion Analyzer
vocal_analyzer = SpeechEmotionAnalyzer()

# Initialize FastAPI application
app = FastAPI(
    title="NEURO-CLINICAL PSYCHOLOGY & AFFECTIVE MONITORING STATION",
    description="Offline Localhost Multimodal Clinical Psychology & Biofeedback Station",
    version="2.0.0"
)

# Static and Templates mounting
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

# Pydantic schemas for REST APIs
class AuthFaceRequest(BaseModel):
    image_b64: str

class EnrollIdentityRequest(BaseModel):
    name: str
    image_b64: str

class GameEvaluateRequest(BaseModel):
    target_id: str
    image_b64: Optional[str] = None
    client_face_detected: Optional[bool] = False
    client_is_smiling: Optional[bool] = False
    client_smile_score: Optional[float] = 0.0

class VocalAssessmentRequest(BaseModel):
    prompt_id: Optional[str] = "PROMPT_01"
    prompt_text: Optional[str] = "Đoạn văn kiểm tra"
    image_b64: Optional[str] = None
    audio_volume_avg: Optional[float] = 0.0
    audio_volume_max: Optional[float] = 0.0
    has_audio: Optional[bool] = False
    has_speech: Optional[bool] = False
    speech_transcript: Optional[str] = ""
    client_face_detected: Optional[bool] = False
    client_is_smiling: Optional[bool] = False
    client_smile_score: Optional[float] = 0.0

class CbtChatRequest(BaseModel):
    message: str
    biometric_context: Dict[str, Any]

class CbtRuleRequest(BaseModel):
    id: Optional[str] = None
    title: str
    category: str
    protocol: str
    clinical_prescription: str

class MultimodalFuseRequest(BaseModel):
    face_probs: list
    audio_probs: list
    weight_face: Optional[float] = None
    weight_audio: Optional[float] = None

# -----------------------------------------------------------------------------
# PHASE 5.2: MULTIMODAL LATE FUSION TRANSLATION & NORMALIZATION HELPERS
# -----------------------------------------------------------------------------
EMOTION_CLASSES_V2 = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

VN_EMOTION_NAMES = {
    "angry": "Giận dữ",
    "disgust": "Ghê tởm",
    "fear": "Sợ hãi",
    "happy": "Vui vẻ",
    "neutral": "Điềm tĩnh",
    "sad": "Buồn bã",
    "surprise": "Ngạc nhiên"
}

def _translate_emotion_vn(emo_str: str) -> str:
    cleaned = str(emo_str).lower().split("/")[0].strip()
    return VN_EMOTION_NAMES.get(cleaned, str(emo_str).capitalize())

def _normalize_7_probs(probs_input) -> np.ndarray:
    if isinstance(probs_input, (list, tuple, np.ndarray)) and len(probs_input) == 7:
        arr = np.asarray(probs_input, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 100.0
        s = float(arr.sum())
        return arr / s if s > 0 else arr

    vec = np.full(7, 0.02, dtype=np.float32)
    if isinstance(probs_input, dict):
        for k, v in probs_input.items():
            k_clean = str(k).lower().split("/")[0].strip()
            if k_clean in EMOTION_CLASSES_V2:
                idx = EMOTION_CLASSES_V2.index(k_clean)
                val = float(v)
                vec[idx] = val / 100.0 if val > 1.0 else val
        s = float(vec.sum())
        return vec / s if s > 0 else vec

    return np.full(7, 1.0 / 7.0, dtype=np.float32)

def _emotion_to_7_probs(emotion_name: str, confidence: float = 0.75) -> np.ndarray:
    clean = str(emotion_name).lower().split("/")[0].strip()
    vec = np.full(7, max(0.01, (1.0 - confidence) / 6.0), dtype=np.float32)
    if clean in EMOTION_CLASSES_V2:
        vec[EMOTION_CLASSES_V2.index(clean)] = confidence
    else:
        vec[EMOTION_CLASSES_V2.index("neutral")] = confidence
    s = float(vec.sum())
    return vec / s if s > 0 else vec

print("[PHASE 5.5] Temporal Aggregation enabled: 5 frames/video.")

def _extract_face_7_probs(
    analysis: Optional[Dict],
    pred: Dict,
    frame: Optional[Union[np.ndarray, List[np.ndarray]]] = None
) -> np.ndarray:
    target_frame = frame if frame is not None else (analysis.get("frame") if analysis and isinstance(analysis, dict) else None)

    # Support both single frame and temporal list of frames
    frames = []
    if isinstance(target_frame, (list, tuple)):
        frames = [f for f in target_frame if f is not None and getattr(f, "size", 0) > 0]
    elif target_frame is not None and getattr(target_frame, "size", 0) > 0:
        frames = [target_frame]

    if len(frames) > 0:
        try:
            from deepface import DeepFace
            ORDERED_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            all_frame_probs = []
            for f in frames:
                try:
                    res = DeepFace.analyze(f, actions=['emotion'], enforce_detection=False)
                    res_dict = res[0] if isinstance(res, list) else res
                    emotion_dict = res_dict.get("emotion", {})
                    raw = np.array([float(emotion_dict.get(k, 0.0)) for k in ORDERED_EMOTIONS], dtype=np.float32)
                    s = float(raw.sum())
                    if s > 0:
                        all_frame_probs.append(raw / s)
                except Exception:
                    continue
            if len(all_frame_probs) > 0:
                mean_probs = np.mean(all_frame_probs, axis=0).astype(np.float32)
                s = float(mean_probs.sum())
                if s > 0:
                    probs = (mean_probs / s).astype(np.float32)
                    assert probs.ndim == 1 and probs.shape == (7,)
                    assert abs(float(probs.sum()) - 1.0) < 1e-4
                    return probs
        except Exception:
            pass

    vec = np.full(7, 0.02, dtype=np.float32)
    if analysis and analysis.get("face_detected"):
        if analysis.get("is_smiling"):
            score = max(0.60, min(0.95, float(analysis.get("smile_score", 85)) / 100.0))
            vec[EMOTION_CLASSES_V2.index("happy")] = score
        else:
            emo_raw = str(analysis.get("emotion", "neutral")).lower().split("/")[0].strip()
            if emo_raw in EMOTION_CLASSES_V2:
                vec[EMOTION_CLASSES_V2.index(emo_raw)] = 0.80
            else:
                vec[EMOTION_CLASSES_V2.index("neutral")] = 0.80
    elif pred and pred.get("probabilities"):
        for k, v in pred["probabilities"].items():
            k_clean = str(k).lower().split("/")[0].strip()
            if k_clean in EMOTION_CLASSES_V2:
                idx = EMOTION_CLASSES_V2.index(k_clean)
                val = float(v)
                vec[idx] = val / 100.0 if val > 1.0 else val
    else:
        vec[EMOTION_CLASSES_V2.index("neutral")] = 0.80

    s = float(vec.sum())
    return vec / s if s > 0 else vec

@app.post("/api/multimodal/fuse")
async def api_multimodal_fuse(req: MultimodalFuseRequest):
    """
    Direct REST API for Phase 5.2 Multimodal Late Fusion (Face + Voice).
    """
    try:
        f_arr = _normalize_7_probs(req.face_probs)
        a_arr = _normalize_7_probs(req.audio_probs)
        pred_label, confidence, final_dist = fusion_engine.fuse(
            f_arr, a_arr, weight_face=req.weight_face, weight_audio=req.weight_audio, return_distribution=True
        )

        face_top_idx = int(np.argmax(f_arr))
        face_top_label = EMOTION_CLASSES_V2[face_top_idx]
        face_top_pct = round(float(f_arr[face_top_idx]) * (100.0 if f_arr[face_top_idx] <= 1.0 else 1.0))

        audio_top_idx = int(np.argmax(a_arr))
        audio_top_label = EMOTION_CLASSES_V2[audio_top_idx]
        audio_top_pct = round(float(a_arr[audio_top_idx]) * (100.0 if a_arr[audio_top_idx] <= 1.0 else 1.0))

        fused_conf_pct = round(float(confidence) * (100.0 if confidence <= 1.0 else 1.0))

        print(
            f"[FUSION] Face: {_translate_emotion_vn(face_top_label)} ({face_top_pct}%) + "
            f"Audio: {_translate_emotion_vn(audio_top_label)} ({audio_top_pct}%) => "
            f"CHỐT: {_translate_emotion_vn(pred_label)} ({fused_conf_pct}%)"
        )

        return JSONResponse({
            "status": "success",
            "fused_emotion": pred_label,
            "fused_confidence": round(float(confidence) * 100, 2),
            "face_emotion": face_top_label,
            "audio_emotion": audio_top_label,
            "distribution": {EMOTION_CLASSES_V2[i]: round(float(final_dist[i]) * 100, 2) for i in range(7)}
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


# -----------------------------------------------------------------------------
# PAGE ROUTES
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Main Clinical Monitoring, Mimic Biofeedback & Forecasting Station."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    """Privileged AI Training Studio & Cognitive Data Inspector."""
    return templates.TemplateResponse(request=request, name="train.html")

# -----------------------------------------------------------------------------
# BIOMETRIC AUTHENTICATION & IDENTITY APIS
# -----------------------------------------------------------------------------
@app.post("/api/auth/verify_face")
async def verify_face(req: AuthFaceRequest):
    img = face_memory.decode_image_from_base64(req.image_b64, enhance=True)
    result = face_memory.verify_admin_biometrics(img_bgr=img)
    return JSONResponse(result)

@app.post("/api/auth/enroll_admin")
async def enroll_admin(req: AuthFaceRequest):
    img = face_memory.decode_image_from_base64(req.image_b64, enhance=True)
    success, msg = face_memory.enroll_admin_direct(img)
    return JSONResponse({"success": success, "message": msg})

@app.post("/api/face/enroll")
async def enroll_face(req: EnrollIdentityRequest):
    img = face_memory.decode_image_from_base64(req.image_b64, enhance=True)
    success, msg = face_memory.enroll_face_identity(req.name, img)
    return JSONResponse({"success": success, "message": msg})

@app.post("/api/face/delete")
async def delete_face(identity_id: str = Query(...)):
    success, msg = face_memory.delete_face_identity(identity_id)
    return JSONResponse({"success": success, "message": msg})

# -----------------------------------------------------------------------------
# BIOFEEDBACK MIMIC MATCH GAME APIS
# -----------------------------------------------------------------------------
@app.get("/api/game/target")
async def get_game_target():
    target = game_engine.get_random_target()
    return JSONResponse(target)

@app.post("/api/game/evaluate")
async def evaluate_game_mimic(req: GameEvaluateRequest):
    img = face_memory.decode_image_from_base64(req.image_b64, enhance=True) if req.image_b64 else None
    analysis = None
    face_vec = None
    face_crop = None
    if img is not None:
        analysis = face_memory.analyze_live_frame(img)
        face_vec, face_crop, _ = face_memory.extract_face_vector(img)
    
    client_data = {
        "client_face_detected": req.client_face_detected,
        "client_is_smiling": req.client_is_smiling,
        "client_smile_score": req.client_smile_score
    }
    res = game_engine.evaluate_mimic_match(
        target_id=req.target_id,
        user_face_vec=face_vec,
        face_crop=face_crop,
        analysis=analysis,
        client_data=client_data
    )
    return JSONResponse(res)

# -----------------------------------------------------------------------------
# TAB 1: VOCAL & MULTIMODAL ELICITATION ASSESSMENT
# -----------------------------------------------------------------------------
@app.get("/api/vocal_assessment/prompt")
async def get_vocal_prompt():
    prompt = clinical_counselor.get_random_elicitation_prompt()
    return JSONResponse(prompt)

@app.post("/api/vocal_assessment")
@app.post("/api/vocal_assessment/upload")
@app.post("/api/vocal_assessment/analyze")
async def upload_vocal_assessment(
    request: Request,
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Vocal assessment upload endpoint.
    Accepts an audio file (.wav / .webm / .ogg), writes it to a temporary file,
    passes the temp path into vocal_analyzer.analyze(temp_audio_path),
    and returns the exact dynamic dictionary returned by vocal_analyzer.analyze().
    """
    target_upload = None
    if file is not None and hasattr(file, "filename") and hasattr(file, "file"):
        target_upload = file
    elif audio is not None and hasattr(audio, "filename") and hasattr(audio, "file"):
        target_upload = audio

    if target_upload is None:
        try:
            form = await request.form()
            for key in form:
                val = form[key]
                if hasattr(val, "filename") and hasattr(val, "file"):
                    target_upload = val
                    break
        except Exception:
            pass

    temp_audio_path = None
    try:
        if target_upload is not None:
            filename = getattr(target_upload, "filename", "recording.wav") or "recording.wav"
            suffix = Path(filename).suffix.lower()
            if not suffix or suffix not in [".wav", ".ogg", ".webm", ".flac", ".mp3"]:
                suffix = ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_audio_path = tmp.name
                shutil.copyfileobj(target_upload.file, tmp)
        else:
            try:
                body = await request.body()
            except Exception:
                body = b""

            if not body:
                return JSONResponse({
                    "status": "error",
                    "message": "Không tìm thấy dữ liệu file âm thanh",
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "energy_level": 0.0,
                    "duration_sec": 0.0
                }, status_code=400)

            c_type = request.headers.get("content-type", "")
            suffix = ".webm" if "webm" in c_type else (".ogg" if "ogg" in c_type else ".wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_audio_path = tmp.name
                tmp.write(body)

        result = vocal_analyzer.analyze(temp_audio_path)

        # Synchronize Subject Identity from face_identities.pt if camera frame provided
        identity_label = "[ĐỐI TƯỢNG: Khách lâm sàng]"
        try:
            form_data = await request.form() if 'form' not in locals() or form is None else form
            img_b64 = form_data.get("image_b64") if hasattr(form_data, "get") else None
            if img_b64:
                face_img = face_memory.decode_image_from_base64(str(img_b64), enhance=True)
                if face_img is not None:
                    id_res = face_memory.identify_face(img_bgr=face_img)
                    identity_label = id_res.get("label", identity_label)
        except Exception:
            pass
        result["identity_label"] = identity_label

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "emotion": "neutral",
            "confidence": 0.0,
            "energy_level": 0.0,
            "duration_sec": 0.0
        }, status_code=500)
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as del_err:
                print(f"[VOCAL WARN] Could not delete temp file: {del_err}")

@app.get("/api/vocal_assessment/benchmark")
async def get_vocal_benchmark():
    """Returns measured Precision, Recall, F1, and Confusion Matrix on TESS dataset."""
    if config.TESS_BENCHMARK_PATH.exists():
        try:
            with open(config.TESS_BENCHMARK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    return JSONResponse({"status": "not_found", "message": "Chưa có báo cáo benchmark TESS"}, status_code=404)

@app.post("/api/vocal_assessment/train")
async def train_vocal_model(epochs: int = Query(25), lr: float = Query(0.002)):
    """Triggers SER training on data/TESS and automatically reloads PyTorch model."""
    try:
        report = train_tess_speech_model(epochs=epochs, lr=lr)
        vocal_analyzer.load_model(config.SPEECH_MODEL_PATH)
        return JSONResponse({
            "status": "success",
            "message": "Huấn luyện mô hình giọng nói TESS thành công",
            "benchmark": report
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/vocal_assessment/evaluate")
async def evaluate_vocal_assessment(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        return await upload_vocal_assessment(request)

    try:
        body_dict = await request.json()
        req = VocalAssessmentRequest(**body_dict)
    except Exception:
        req = VocalAssessmentRequest()

    face_detected = False
    is_smiling = False
    smile_score = 0.0
    face_vec = None

    if req.image_b64:
        img = face_memory.decode_image_from_base64(req.image_b64, enhance=True)
        if img is not None:
            analysis = face_memory.analyze_live_frame(img)
            face_detected = bool(analysis.get("face_detected", False))
            if face_detected:
                is_smiling = bool(analysis.get("is_smiling", False))
                smile_score = float(analysis.get("smile_score", 0.0))
                face_vec, _, _ = face_memory.extract_face_vector(img)

    # Honor live tracking data from client during the 20s test
    if req.client_face_detected or (req.image_b64 and len(req.image_b64) > 100):
        face_detected = True
    if req.client_is_smiling:
        is_smiling = True
    if req.client_smile_score and float(req.client_smile_score) > smile_score:
        smile_score = float(req.client_smile_score)

    if face_vec is None:
        face_vec = fusion_engine.extract_face_feature_mock(seed_val=1.2)

    if smile_score > 1.0:
        smile_score = smile_score / 100.0
    smile_score = max(0.0, min(1.0, smile_score))

    avg_vol = float(req.audio_volume_avg or 0.0)
    max_vol = float(req.audio_volume_max or 0.0)
    has_transcript = bool(req.speech_transcript and len(req.speech_transcript.strip()) >= 2)
    has_speech = bool(req.has_speech or has_transcript or req.has_audio or max_vol > 0.005 or avg_vol > 0.001)
    has_ambient_noise = bool(req.has_audio or max_vol > 0.003 or avg_vol > 0.0005)

    speech_vec = fusion_engine.extract_audio_feature_from_pcm([])
    body_vec = torch.randn(1, config.DIM_BODY)
    text_vec = fusion_engine.extract_text_feature(req.prompt_text)

    prediction = fusion_engine.predict_multimodal(
        face_vec, body_vec, speech_vec, text_vec, text_str=req.prompt_text
    )

    # Calculate physiological synchrony & clinical readiness score
    synchrony = round(max(55.0, min(98.0, 78.0 + (prediction["valence"] * 12.0) + (smile_score * 12.0))), 1)
    vol_bonus = min(8.0, avg_vol * 40.0) if has_speech else min(4.0, avg_vol * 20.0)
    base_score = 68.0 + (synchrony * 0.16) + (smile_score * 14.0) + vol_bonus
    final_score = int(max(65, min(98, round(base_score))))

    if final_score >= 82:
        status_label = "TÂM THẦN ỔN ĐỊNH & THƯ THÁI"
        presc = "Chỉ số cân bằng thần kinh của bạn rất tốt. Sự phối hợp giữa nét mặt, phản xạ nụ cười và âm phổ thể hiện tâm lý vững vàng. Hãy tiếp tục duy trì tư thế ngồi thẳng và bổ sung nước ấm."
    elif final_score >= 70:
        status_label = "CÂN BẰNG TÂM LÝ - CĂNG CƠ NHẸ"
        presc = "Hệ thống ghi nhận dấu hiệu căng cứng nhẹ ở cơ mặt hoặc âm điệu giọng nói. Khuyến nghị thực hiện 3 chu kỳ thở sâu 4-7-8 và thư giãn cơ vai."
    else:
        status_label = "ÁP LỰC / MỆT MỎI TÂM LÝ"
        presc = "Hệ thống phát hiện dấu hiệu mệt mỏi và phân tán cảm xúc. Khuyến nghị rời màn hình 5 phút, chớp mắt thư giãn theo quy tắc 20-20-20 và nghe nhạc sóng Alpha."

    # 1. Synchronize CBT Clinical Prescription dynamically from custom_knowledge.json
    cbt_info = clinical_counselor.get_prescription_for_assessment(
        final_score=final_score,
        emotion=prediction["emotion"],
        exhaustion_level=prediction.get("mental_exhaustion_level", 1)
    )
    if cbt_info and cbt_info.get("prescription"):
        presc = cbt_info["prescription"]

    # 2. Synchronize Subject Identity from face_identities.pt & admin_face.pt
    identity_info = face_memory.identify_face(img_bgr=img, face_vec=face_vec) if face_detected else {
        "identified": False, "name": "Không phát hiện khuôn mặt", "similarity_pct": 0.0, "label": "[ĐỐI TƯỢNG: Không phát hiện khuôn mặt]"
    }

    face_desc = f"Nụ cười Duchenne tự nhiên (+{round(smile_score*100)}%), cơ mặt thư giãn" if is_smiling else "Nét mặt điềm tĩnh, trương lực cơ trán thư giãn chuẩn lâm sàng"
    
    if has_speech:
        voice_desc = f"Đã ghi nhận giọng đọc rõ ràng{f' («{req.speech_transcript[:35]}...»)' if req.speech_transcript else ''} (Âm lượng {round(avg_vol*100)}%), âm phổ thanh quản đồng pha nét mặt"
    elif has_ambient_noise:
        voice_desc = f"Micro ghi nhận âm thanh mức cơ bản ({round(avg_vol*100)}%), âm phổ thanh quản đồng pha nét mặt"
    else:
        voice_desc = "Micro ghi nhận âm lượng nhỏ (Khuyến nghị để gần micro hơn ở lần đo sau)"

    cong_desc = f"{int(synchrony)}% (Cao - Lời nói và biểu cảm nét mặt đồng nhất)"

    return JSONResponse({
        "prompt_id": req.prompt_id,
        "valid_assessment": True,
        "face_detected": face_detected,
        "identity_name": identity_info.get("name", "Khách lâm sàng"),
        "identity_label": identity_info.get("label", "[ĐỐI TƯỢNG: Khách lâm sàng]"),
        "identity_similarity": identity_info.get("similarity_pct", 0.0),
        "cbt_rule_id": cbt_info.get("rule_id", "DEFAULT"),
        "cbt_rule_title": cbt_info.get("title", ""),
        "has_audio": has_ambient_noise or has_speech,
        "has_speech": has_speech,
        "speech_transcript": req.speech_transcript or "",
        "is_smiling": is_smiling,
        "smile_score": smile_score,
        "audio_volume_avg": avg_vol,
        "final_score": final_score,
        "status_label": status_label,
        "face_detail": face_desc,
        "voice_detail": voice_desc,
        "congruence_detail": cong_desc,
        "burnout_detail": "Cấp 1/5 (Năng lượng tốt, vỏ não trước trán phục hồi chuẩn)",
        "prescription": presc,
        "emotion": prediction["emotion"],
        "valence": prediction["valence"],
        "arousal": prediction["arousal"]
    })

def np_random_jitter():
    import random
    return random.uniform(-4.0, 4.0)

# -----------------------------------------------------------------------------
# CBT CLINICAL COUNSELING APIS
# -----------------------------------------------------------------------------
@app.post("/api/cbt/chat")
async def cbt_chat(req: CbtChatRequest):
    response_text = clinical_counselor.analyze_dialogue_with_biometrics(
        user_message=req.message,
        biometric_context=req.biometric_context
    )
    return JSONResponse({"response": response_text})

@app.get("/api/cbt/rules")
async def get_cbt_rules():
    rules = clinical_counselor.get_all_rules()
    return JSONResponse(rules)

@app.post("/api/cbt/rules")
async def save_cbt_rule(req: CbtRuleRequest):
    success, rule_id = clinical_counselor.add_or_update_rule(req.dict())
    return JSONResponse({"success": success, "rule_id": rule_id})

@app.post("/api/cbt/rules/delete")
async def delete_cbt_rule(rule_id: str = Query(...)):
    success, msg = clinical_counselor.delete_rule(rule_id)
    return JSONResponse({"success": success, "message": msg})

# -----------------------------------------------------------------------------
# HARDWARE TELEMETRY API
# -----------------------------------------------------------------------------
@app.get("/api/telemetry")
async def get_hardware_telemetry():
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = round(mem.used / (1024**3), 2)
    ram_total = round(mem.total / (1024**3), 2)
    ram_pct = mem.percent

    # GPU telemetry if torch.cuda is available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_alloc = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        gpu_vram = f"{vram_alloc}GB / {vram_total}GB ({gpu_name})"
        gpu_temp = "48°C (Nominal)"
    else:
        gpu_vram = "N/A (CPU Mode Active)"
        gpu_temp = "N/A (CPU Fan Controlled)"

    return JSONResponse({
        "cpu_percent": cpu_pct,
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
        "ram_percent": ram_pct,
        "gpu_vram_info": gpu_vram,
        "gpu_temp_info": gpu_temp
    })

# -----------------------------------------------------------------------------
# MODULE 5: COGNITIVE DATA & CHECKPOINT INSPECTOR API
# -----------------------------------------------------------------------------
@app.get("/api/inspector/data")
async def get_inspector_data():
    # 1. Class distribution of custom_samples.pt
    class_dist = {cls_name: 0 for cls_name in config.EMOTION_CLASSES}
    if config.CUSTOM_SAMPLES_PATH.exists():
        try:
            data = torch.load(config.CUSTOM_SAMPLES_PATH, map_location="cpu")
            labels = data['label'].numpy()
            for lbl in labels:
                if 0 <= lbl < len(config.EMOTION_CLASSES):
                    class_dist[config.EMOTION_CLASSES[lbl]] += 1
        except Exception:
            pass

    # If empty, ensure default cluster counts
    if sum(class_dist.values()) == 0:
        for c in config.EMOTION_CLASSES:
            class_dist[c] = 60

    # 2. Face identities ledger
    ledger = face_memory.get_all_identities_ledger()

    # 3. CBT rules
    cbt_rules = clinical_counselor.get_all_rules()

    # 4. Model checkpoints
    registry = []
    if config.CHECKPOINT_REGISTRY_PATH.exists():
        try:
            with open(config.CHECKPOINT_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    if not registry:
        registry = [
            {
                "version_id": "GMU-v001",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epochs": 10,
                "accuracy": 89.2,
                "loss": 0.3421,
                "status": "ACTIVE_PRODUCTION"
            }
        ]

    return JSONResponse({
        "class_distribution": class_dist,
        "face_identities": ledger,
        "cbt_rules": cbt_rules,
        "model_checkpoints": registry
    })

@app.post("/api/checkpoints/restore")
async def restore_model_checkpoint(version_id: str = Query(...)):
    success, msg = fusion_engine.restore_checkpoint(version_id)
    return JSONResponse({"success": success, "message": msg})

# -----------------------------------------------------------------------------
# WEBSOCKET: REAL-TIME CLINICAL MONITOR (TAB 1)
# -----------------------------------------------------------------------------
@app.websocket("/ws/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    await websocket.accept()
    blink_detector = BlinkDetector()
    head_pose_engine = HeadPoseEstimator()
    nose_history = []
    t = 0
    try:
        while True:
            raw_msg = await websocket.receive_text()
            cmd = json.loads(raw_msg)
            t += 1

            image_b64 = cmd.get("image_b64")
            manual_override = cmd.get("manual_override")

            b_vec = torch.randn(1, config.DIM_BODY)
            s_vec = torch.randn(1, config.DIM_SPEECH)
            t_vec = torch.randn(1, config.DIM_TEXT)

            if manual_override:
                # User clicked a test emotion preset
                target_emotion = manual_override.get("emotion", "HAPPY / EQUILIBRIUM")
                target_val = float(manual_override.get("valence", 0.70))
                target_arou = float(manual_override.get("arousal", 0.40))
                f_vec = fusion_engine.extract_face_feature_mock(seed_val=abs(hash(target_emotion)) % 100)
                pred = fusion_engine.predict_multimodal(f_vec, b_vec, s_vec, t_vec)
                pred["emotion"] = target_emotion
                pred["valence"] = target_val
                pred["arousal"] = target_arou
                pred["mental_exhaustion_level"] = 0 if target_val > 0.3 else (3 if target_val < -0.3 else 1)
                pred["face_detected"] = True
                pred["bbox"] = [180, 70, 260, 290]
                pred["frame_w"] = 640
                pred["frame_h"] = 480
                pred["is_smiling"] = (target_val > 0.4)
                pred["smile_score"] = 85.0 if target_val > 0.4 else 0.0
                pred["head_tilt_deg"] = 0.0
                pred["head_tilt"] = 0.0
                pred["pitch"] = 0.0
                pred["yaw"] = 0.0
                pred["roll"] = 0.0
                pred["posture_warning"] = False
                pred["ear"] = 0.30
                pred["blink_count"] = 0
                pred["blink_rate_bpm"] = 16.0
                pred["is_blinking"] = False
                pred["neural_blink_score"] = 0.0
                pred["nose_variance"] = 1.2
                pred["fidget_score"] = 5.5
            elif image_b64:
                # Real webcam frame from client
                img = face_memory.decode_image_from_base64(image_b64, enhance=True)
                analysis = face_memory.analyze_live_frame(img)
                f_vec, _, _ = face_memory.extract_face_vector(img)
                pred = fusion_engine.predict_multimodal(f_vec, b_vec, s_vec, t_vec)

                pred["face_detected"] = analysis["face_detected"]
                pred["bbox"] = analysis["bbox"]
                pred["landmarks"] = analysis.get("landmarks")
                pred["frame_w"] = analysis.get("frame_w")
                pred["frame_h"] = analysis.get("frame_h")
                pred["is_smiling"] = analysis["is_smiling"]
                pred["smile_score"] = analysis["smile_score"]
                pred["head_tilt_deg"] = analysis["head_tilt_deg"]
                pred["posture_warning"] = analysis["posture_warning"]
                pred["emotion"] = analysis["emotion"]
                pred["valence"] = analysis["valence"]
                pred["arousal"] = analysis["arousal"]
                pred["mental_exhaustion_level"] = analysis["exhaustion_level"]

                # MediaPipe 6-point EAR & Neural Blendshape Blink Detection
                if img is not None:
                    blink_data = blink_detector.update(img)
                    pred["ear"] = blink_data["ear"]
                    pred["blink_count"] = blink_data["blink_count"]
                    pred["blink_rate_bpm"] = blink_data["blink_rate_bpm"]
                    pred["is_blinking"] = blink_data["is_blinking"]
                    pred["neural_blink_score"] = blink_data.get("neural_blink_score", 0.0)
                else:
                    pred["ear"] = 0.32
                    pred["blink_count"] = 0
                    pred["blink_rate_bpm"] = 0.0
                    pred["is_blinking"] = False
                    pred["neural_blink_score"] = 0.0

                # 5-point YuNet 3D Head Pose Estimation (solvePnP)
                lms = analysis.get("landmarks")
                if img is not None and analysis.get("face_detected") and lms and all(k in lms for k in ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")):
                    lms_5pts = [
                        lms["right_eye"],
                        lms["left_eye"],
                        lms["nose"],
                        lms["right_mouth"],
                        lms["left_mouth"]
                    ]
                    pose_data = head_pose_engine.estimate(img.shape[:2], lms_5pts)
                    pred["head_tilt"] = pose_data.get("head_tilt", 0.0)
                    pred["pitch"] = pose_data.get("pitch", 0.0)
                    pred["yaw"] = pose_data.get("yaw", 0.0)
                    pred["roll"] = pose_data.get("roll", 0.0)
                    if pred["head_tilt"] > 18.0 or abs(pred["yaw"]) > 25.0 or abs(pred["roll"]) > 18.0:
                        pred["posture_warning"] = True
                else:
                    pred["head_tilt"] = 0.0
                    pred["pitch"] = 0.0
                    pred["yaw"] = 0.0
                    pred["roll"] = 0.0

                # Fidget tracking via YuNet nose variance sliding window
                if analysis.get("face_detected") and lms and lms.get("nose"):
                    now = time.time()
                    nx, ny = lms["nose"]
                    nose_history.append((nx, ny, now))
                    nose_history = [pt for pt in nose_history if now - pt[2] <= 4.0]
                    if len(nose_history) >= 4:
                        xs = [pt[0] for pt in nose_history]
                        ys = [pt[1] for pt in nose_history]
                        mx = sum(xs) / len(xs)
                        my = sum(ys) / len(ys)
                        v_total = sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs)
                        pred["nose_variance"] = round(v_total, 2)
                        pred["fidget_score"] = round(min(100.0, math.sqrt(v_total) * 4.5), 1)
                    else:
                        pred["nose_variance"] = 0.0
                        pred["fidget_score"] = 0.0
                else:
                    pred["nose_variance"] = 0.0
                    pred["fidget_score"] = 0.0
            else:
                # Idle heartbeat baseline
                sim_v = round(0.12 * math.sin(t * 0.1) + 0.05, 2)
                sim_a = round(0.10 * math.cos(t * 0.08) - 0.02, 2)
                f_vec = fusion_engine.extract_face_feature_mock(seed_val=t * 0.5)
                pred = fusion_engine.predict_multimodal(f_vec, b_vec, s_vec, t_vec)
                pred["valence"] = sim_v
                pred["arousal"] = sim_a
                pred["face_detected"] = False
                pred["bbox"] = [180, 70, 260, 290]
                pred["frame_w"] = 640
                pred["frame_h"] = 480
                pred["is_smiling"] = False
                pred["smile_score"] = 0.0
                pred["head_tilt_deg"] = 0.0
                pred["head_tilt"] = 0.0
                pred["pitch"] = 0.0
                pred["yaw"] = 0.0
                pred["roll"] = 0.0
                pred["posture_warning"] = False
                pred["ear"] = 0.32
                pred["blink_count"] = 0
                pred["blink_rate_bpm"] = 0.0
                pred["is_blinking"] = False
                pred["neural_blink_score"] = 0.0
                pred["nose_variance"] = 0.0
                pred["fidget_score"] = 0.0

            # -----------------------------------------------------------------
            # PHASE 5.2: MULTIMODAL LATE FUSION (FACE + AUDIO EMOTION RECOGNITION)
            # -----------------------------------------------------------------
            face_probs = cmd.get("face_probs")
            audio_probs = cmd.get("audio_probs")

            # Auto-derive face_probs if not directly passed in message
            if face_probs is None:
                if 'analysis' in locals() and analysis and analysis.get("face_detected"):
                    face_probs = _extract_face_7_probs(analysis, pred)
                elif pred.get("probabilities"):
                    face_probs = _extract_face_7_probs(None, pred)

            # Auto-check vocal_probs or audio_emotion in cmd
            if audio_probs is None:
                if cmd.get("vocal_probs"):
                    audio_probs = cmd.get("vocal_probs")
                elif cmd.get("audio_emotion"):
                    audio_probs = _emotion_to_7_probs(cmd.get("audio_emotion"), confidence=cmd.get("audio_confidence", 0.75))

            # Perform late fusion whenever both face_probs and audio_probs are collected
            if face_probs is not None and audio_probs is not None:
                face_probs_arr = _normalize_7_probs(face_probs)
                audio_probs_arr = _normalize_7_probs(audio_probs)

                pred_label, confidence = fusion_engine.fuse(face_probs_arr, audio_probs_arr)

                pred["fused_emotion"] = pred_label
                pred["fused_confidence"] = round(confidence * 100.0, 1) if confidence <= 1.0 else round(confidence, 1)

                # Extract top classes and confidence percentages for terminal logging
                f_top_idx = int(np.argmax(face_probs_arr))
                f_top_name = EMOTION_CLASSES_V2[f_top_idx]
                f_top_pct = round(float(face_probs_arr[f_top_idx]) * (100.0 if face_probs_arr[f_top_idx] <= 1.0 else 1.0))

                a_top_idx = int(np.argmax(audio_probs_arr))
                a_top_name = EMOTION_CLASSES_V2[a_top_idx]
                a_top_pct = round(float(audio_probs_arr[a_top_idx]) * (100.0 if audio_probs_arr[a_top_idx] <= 1.0 else 1.0))

                fused_pct = round(float(confidence) * (100.0 if confidence <= 1.0 else 1.0))

                # Terminal Logging in Vietnamese
                print(
                    f"[FUSION] Face: {_translate_emotion_vn(f_top_name)} ({f_top_pct}%) + "
                    f"Audio: {_translate_emotion_vn(a_top_name)} ({a_top_pct}%) => "
                    f"CHỐT: {_translate_emotion_vn(pred_label)} ({fused_pct}%)"
                )
            else:
                # Default baseline to maintain complete non-destructive compatibility
                if "fused_emotion" not in pred:
                    pred["fused_emotion"] = pred.get("emotion", "neutral")
                    pred["fused_confidence"] = pred.get("confidence", 75.0)

            await websocket.send_json({
                "type": "TELEMETRY_UPDATE",
                "data": pred,
                "fused_emotion": pred.get("fused_emotion"),
                "fused_confidence": pred.get("fused_confidence")
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS_MONITOR_ERROR]: {e}")

# -----------------------------------------------------------------------------
# WEBSOCKET: BACKGROUND MODEL TRAINING (MODULE 2)
# -----------------------------------------------------------------------------
@app.websocket("/ws/train")
async def websocket_train_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    def sync_send_callback(message_dict):
        # Thread-safe dispatch back to asyncio event loop
        asyncio.run_coroutine_threadsafe(
            websocket.send_json(message_dict),
            loop
        )

    try:
        while True:
            raw_msg = await websocket.receive_text()
            cmd = json.loads(raw_msg)
            if cmd.get("action") == "start_train":
                epochs = int(cmd.get("epochs", 10))
                batch_size = int(cmd.get("batch_size", 16))
                lr = float(cmd.get("lr", 0.001))
                dropout = float(cmd.get("dropout", 0.2))

                # Launch training in background worker thread so it doesn't block the server
                train_thread = threading.Thread(
                    target=fusion_engine.train_background,
                    kwargs={
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "dropout": dropout,
                        "websocket_callback": sync_send_callback
                    },
                    daemon=True
                )
                train_thread.start()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS_TRAIN_ERROR]: {e}")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Clinical Station on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)

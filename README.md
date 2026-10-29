# NEURO-CLINICAL PSYCHOLOGY & AFFECTIVE MONITORING STATION
### Trạm Giám Sát, Can Thiệp Tâm Lý & Huấn Luyện AI Đa Phương Thái Lâm Sàng

Hệ sinh thái mã nguồn hoàn chỉnh chạy **100% OFFLINE TRÊN LOCALHOST** (CPU hoặc GPU NVIDIA), TUYỆT ĐỐI KHÔNG GỌI API BÊN THỨ 3 (Không OpenAI, Claude hay Gemini API), bảo vệ an toàn tuyệt đối dữ liệu sinh trắc học y tế theo chuẩn HIPAA / GDPR.

---

## 1. TIÊU CHUẨN THIẾT KẾ APPLE GLASSMORPHISM
- **Màu sắc y tế Bắc Âu**: Nền sáng / Xám khói tinh tế (`#F8FAFC`, `#FFFFFF`, viền kính siêu mỏng bán trong suốt `rgba(0, 0, 0, 0.08)` và `rgba(255, 255, 255, 0.7)`).
- **Hình khối & Bo góc**: Góc bo 6px vững chãi, khoa học, tin cậy. Hiệu ứng `backdrop-filter: blur(20px)` và đổ bóng đa tầng `box-shadow` mịn màng.
- **Typography chính hãng Google Fonts**:
  - Giao diện, tiêu đề & văn bản: **Inter** (letter-spacing: -0.02em).
  - Số liệu y khoa, Thẻ kỹ thuật & Terminal: **JetBrains Mono** (letter-spacing: 0.05em, text-transform: uppercase).
- **Nguyên tắc bắt buộc**: **100% ZERO EMOJIS**. Thay thế toàn bộ bằng Thẻ trạng thái in hoa tối giản (Typographic Badges), chỉ số Monospace và biểu tượng SVG viền mảnh.

---

## 2. NỀN TẢNG KỸ THUẬT & MÔ HÌNH AI BACKBONE
- **Backend**: Python 3.10+ (FastAPI, Uvicorn, WebSockets truyền nhận video/telemetry/training log thời gian thực).
- **Frontend**: HTML5, Vanilla CSS3 Tokens, Vanilla JavaScript ES6+ (Web Audio API, Canvas, MediaDevices).
- **4 Nhánh Trích Xuất Đặc Trưng (Frozen Backbones theo Paper arXiv:2412.17907)**:
  1. **Face**: `ResNet50` trích xuất vector khuôn mặt $d_{face} = 2048$ chiều (kết hợp nhận diện vùng mặt OpenCV).
  2. **Body**: `MediaPipe Pose` trích xuất 33 keypoints 3D $d_{body} = 132$ chiều ($33 \times 4$), vẽ khung xương màu xám tro/bạc thanh lịch.
  3. **Speech**: Phân tích âm thanh 16kHz $d_{speech} = 768$ chiều (quang phổ tần số f0, jitter/shimmer, độ trầm bổng và năng lượng âm thanh).
  4. **Text**: Chiếu ngữ nghĩa văn bản $d_{text} = 768$ chiều.
- **Mạng Hợp Nhất (Gated Multimodal Unit - GMU & Dynamic Attention)**:
  - Tự động tính toán tỷ trọng đóng góp % của 4 kênh (Face %, Body %, Speech %, Text %).
  - Phân loại 6 trạng thái tâm lý:
    1. `[HAPPY / EQUILIBRIUM]` (Cân bằng tích cực)
    2. `[SAD / DEPRESSIVE]` (Trầm buồn / Suy giảm trương lực)
    3. `[ANGRY / HIGH AROUSAL]` (Căng thẳng kích động cao)
    4. `[NEUTRAL / CALM]` (Điềm tĩnh trung tính)
    5. `[SURPRISE]` (Bất ngờ cảnh giác)
    6. `[FEAR / ANXIETY]` (Lo âu / Báo động)
  - Ước lượng 2 chiều liên tục: Valence ($-1.0 \to +1.0$) và Arousal ($-1.0 \to +1.0$).
  - **Cơ chế Phát hiện Mâu thuẫn Cảm xúc (Incongruence / Sarcasm Detector)**: Tự động đo độ lệch pha giữa ngữ nghĩa Lời nói với (Face + Speech) để cảnh báo `[AFFECTIVE-SEMANTIC INCONGRUENCE DETECTED]` nếu người dùng nói tích cực nhưng nét mặt và giọng nói thể hiện kiệt sức hoặc lo âu.

---

## 3. CÁC PHÂN HỆ VÀ TÍNH NĂNG CHÍNH

### ROUTE '/' - TRẠM LÂM SÀNG CHÍNH (3 TABS):
- **TAB 1: GIÁM SÁT TÂM LÝ & ÂM THANH TRỊ LIỆU (CLINICAL MONITOR)**:
  - Khung giám sát trực quan (Camera & Audio Feed) với khung xương MediaPipe và khung ngắm Reticle chuẩn y tế.
  - Các chỉ số thời gian thực: `[STATUS: MENTAL EXHAUSTION LEVEL 1/5]`, `[VALENCE: +0.05 | AROUSAL: +0.02]`.
  - Thanh tiến độ % đóng góp 4 kênh (Face %, Body %, Speech %, Text %).
  - **Tính năng Đánh Giá Đồng Pha Giọng Nói - Nét Mặt (Vocal Elicitation)**:
    - AI tạo ngẫu nhiên một câu lâm sàng (nút `[GENERATE NEW PROMPT]`).
    - Bấm `[GHI ÂM & PHÂN TÍCH ĐỒNG PHA 4 KÊNH]`: Hệ thống thu âm 16kHz và chụp khung hình khuôn mặt + dáng ngồi cùng lúc để đo độ đồng nhất cảm xúc.
  - **Hệ thống Âm thanh Trị liệu (Web Audio API Synthesizer 100% Offline)**:
    - Tự động phát âm 5 giây (Sóng hài 432Hz, Chuông xoay Tây Tạng, Tiếng mưa nhẹ) khi phát hiện kiệt sức cấp 3/5 hoặc căng thẳng cao độ.
    - Công tắc bật/tắt nhạc nền thư giãn dài liên tục: Sóng não Alpha 10Hz (Binaural Drone 432Hz/442Hz).
  - **Bộ đếm nhịp thở 4-7-8**: Đồ họa vòng tròn kính mờ mở rộng (hít vào 4s) -> giữ hơi (7s) -> thu nhỏ (thở ra 8s).
  - **Trợ lý Tham vấn CBT (Cognitive Behavioral Therapy Chat)**: Đối thoại lâm sàng thấu cảm, khoa học, phân tích trực tiếp theo các chỉ số sinh trắc học đang đo từ camera.

- **TAB 2: TRÒ CHƠI BẮT CHƯỚC CẢM XÚC (AFFECTIVE MIMIC MATCH - BIOFEEDBACK GAME)**:
  - Rèn luyện cơ mặt và giải tỏa ức chế thần kinh.
  - Hiển thị biểu cảm mẫu SVG wireframe sắc nét (Nụ cười tự nhiên, Bất ngờ cảnh giác, Tĩnh tâm thiền định, Tập trung cao độ, Giải tỏa căng thẳng).
  - Đồng hồ đếm ngược 3 GIÂY to rõ: `03` -> `02` -> `01` -> Chụp ảnh.
  - AI so sánh Cosine Similarity với vector biểu cảm mẫu, báo điểm `[MATCHING SCORE: 88.5%]`, chi tiết từng nhóm cơ mặt, điểm chuỗi (Streak) và kỷ lục phiên.

- **TAB 3: DỰ BÁO TƯƠNG LAI CẢM XÚC & ĐƠN THUỐC SỨC KHỎE**:
  - Biểu đồ đường thời gian thực ghi nhận biến thiên Valence và Arousal trong phiên.
  - Chỉ số nguy cơ kiệt sức: `[BURNOUT RISK INDEX: 28% IN NEXT 2 HOURS]`.
  - Cảnh báo tư thế gù lưng / nghiêng cổ từ MediaPipe: `[POSTURE WARNING: FORWARD HEAD STRAIN]`.
  - Đơn thuốc hành vi: `[PRESCRIPTION: INTAKE 250ML WATER]`, `[EYE CARE: 20-20-20 RULE]`, `[ERGONOMICS: SHOULDER RETRACTION 10 REPS]`.

---

### ROUTE '/train' - LOCAL AI TRAINING STUDIO & PRIVILEGED BIOMETRIC PORTAL:
- **Cổng Xác Thực Sinh Trắc Học Khuôn Mặt (Biometric Face Gate)**:
  - Khóa toàn màn hình kính mờ khi truy cập `/train`.
  - Tự động bật camera quét khuôn mặt, đối sánh với `data/admin_face.pt` qua ResNet50 Cosine Similarity:
    - Nếu $\ge 70\%$: Mở khóa hiệu ứng trượt kính, cấp quyền Admin Session.
    - Nếu $< 70\%$: Báo đỏ `[ACCESS DENIED: BIOMETRIC MISMATCH]` và chặn truy cập.
    - Tích hợp nút `[ĐĂNG KÝ MẶT NÀY LÀM ADMIN]` để người dùng đăng ký khuôn mặt của mình làm quản trị viên.
- **5 Phân Hệ Quản Trị & Huấn Luyện**:
  1. **Face Identity Bank**: Đăng ký tên định danh và ảnh khuôn mặt vào `data/face_identities.pt`.
  2. **Multimodal Fusion Head Trainer**:
     - Bảng tinh chỉnh: Epochs, Batch Size, Learning Rate, Dropout.
     - Nút `[INITIATE TRAINING TRÊN LUỒNG PHỤ]`: Huấn luyện PyTorch không nghẽn web.
     - Biểu đồ Canvas vẽ trực tiếp Train Loss, Val Loss và Accuracy sau từng Epoch qua WebSocket.
     - Terminal console kính mờ hiển thị log chi tiết và tự lưu checkpoint `checkpoints/best_fusion_model.pt`.
     - Tích hợp sẵn bộ dữ liệu mẫu (Synthetic Feature Clusters) sẵn sàng huấn luyện ngay.
  3. **CBT Knowledge Base**: Soạn thảo và thêm quy tắc can thiệp tâm lý vào `data/custom_knowledge.json`.
  4. **Hardware Telemetry**: Đo tức thời CPU Load, System RAM, GPU VRAM và GPU Temp.
  5. **Cognitive Data & Checkpoint Inspector**:
     - Biểu đồ phân bố 6 nhãn cảm xúc (Class Distribution Bar Chart).
     - Danh mục khuôn mặt đã nhớ (Face Identities Ledger) kèm nút `[XÓA]`.
     - Kho tri thức CBT (Clinical Rule Repository) kèm nút `[XÓA]`.
     - Nhật ký và khôi phục phiên bản Checkpoint (Model Version Registry) kèm nút `[RESTORE]`.

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN
```
├── app.py                      # FastAPI App, WebSockets, API routes (/, /train, /api/auth, /api/data)
├── config.py                   # Cấu hình cổng 8000, checkpoints, device CUDA/CPU
├── core/
│   ├── fusion_engine.py        # Mạng PyTorch GMU, Attention, 4 nhánh và Incongruence detector
│   ├── face_memory.py          # ResNet50 embeddings, Admin Gate và Face Identity Bank
│   ├── clinical_counselor.py   # AI Bác sĩ tâm lý CBT & Elicitation Prompts
│   ├── sound_generator.py      # Bộ thông số âm thanh trị liệu & Web Audio API
│   └── game_engine.py          # Logic Mimic Match 3s & đánh giá cơ mặt
├── static/
│   ├── css/
│   │   └── style.css           # CSS chuẩn Apple Glassmorphism, 6px borders, Inter & JetBrains Mono
│   ├── js/
│   │   ├── main.js             # 3 Tabs lâm sàng, WebSocket stream, Canvas HUD, Vocal Elicitation
│   │   ├── audio_therapy.js    # Web Audio API 432Hz, chuông Tây Tạng, mưa hồng & sóng Alpha
│   │   ├── game.js             # Game Biofeedback 3s countdown & tính điểm Cosine
│   │   └── train.js            # Xác thực sinh trắc học, biểu đồ Loss/Acc & Data Inspector
│   └── assets/
│       └── memes/              # Ảnh mẫu biểu cảm SVG wireframe
├── templates/
│   ├── index.html              # Giao diện chính 3 Tabs kính mờ
│   └── train.html              # Giao diện Cổng sinh trắc học, Studio huấn luyện & Data Inspector
├── data/
│   ├── admin_face.pt           # Vector khuôn mặt Admin mở khóa
│   ├── face_identities.pt      # Ngân hàng nhận diện khuôn mặt người quen
│   ├── custom_samples.pt       # Dữ liệu cảm xúc tích lũy & synthetic clusters
│   └── custom_knowledge.json   # Bộ nhớ tri thức tâm lý học
├── checkpoints/
│   ├── best_fusion_model.pt    # Checkpoint mô hình PyTorch tối ưu
│   └── model_registry.json     # Nhật ký các phiên bản mô hình đã lưu
├── requirements.txt            # Danh sách thư viện Python
└── README.md                   # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG

### Bước 1: Khởi động máy chủ ứng dụng
Mở terminal tại thư mục dự án và chạy lệnh:
```bash
python app.py
```
*(Hoặc chạy qua uvicorn)*:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### Bước 2: Truy cập Trạm Lâm Sàng
Mở trình duyệt Web (Chrome, Edge, Firefox) và truy cập:
```
http://127.0.0.1:8000
```
- Trình duyệt sẽ yêu cầu cấp quyền Camera và Microphone để đo đạc sinh trắc học.
- Trải nghiệm Tab 1 (Giám sát tâm lý, Thử nghiệm phát âm đồng pha, Vòng thở 4-7-8, CBT Chat), Tab 2 (Thử thách Mimic 3s), Tab 3 (Dự báo nguy cơ kiệt sức).

### Bước 3: Truy cập Cổng Sinh Trắc & Studio Huấn Luyện
Nhấp vào nút `[TRAINING STUDIO & CỔNG SINH TRẮC]` ở góc trên bên phải hoặc truy cập trực tiếp:
```
http://127.0.0.1:8000/train
```
- Cổng sinh trắc sẽ quét khuôn mặt trực tiếp. Lần đầu tiên chạy, bạn có thể nhấp vào `[ĐĂNG KÝ MẶT NÀY LÀM ADMIN]` để lưu khuôn mặt của bạn làm Admin chính thức.
- Khi điểm trùng khớp $\ge 70\%$, cánh cửa kính mờ sẽ trượt mở, cho phép bạn truy cập toàn bộ 5 Module huấn luyện và Data Inspector.

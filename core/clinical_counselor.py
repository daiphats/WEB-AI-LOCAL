import json
import random
import re
from datetime import datetime
from pathlib import Path
import config

DEFAULT_KNOWLEDGE_RULES = [
    {
        "id": "RULE_DEPRESSION",
        "category": "MOOD_DISORDER",
        "trigger": "trầm cảm, depression, mdd, chán nản, tuyệt vọng, mất động lực, u sầu, anhedonia, buồn bã, chán đời, bế tắc",
        "title": "Rối Loạn Trầm Cảm Chủ Yếu (Major Depressive Disorder - MDD)",
        "protocol": "Phân tích Tam giác Nhận thức Tiêu cực của Aaron Beck (Tự ti bản thân - Khắt khe thế giới - Bi quan tương lai). Áp dụng Kích hoạt Hành vi (Behavioral Activation - BA): Thực hiện hành động trước khi có cảm xúc, chia nhỏ nhiệm vụ thành vi hành động 2 phút.",
        "socratic_question": "Bằng chứng khách quan nào chứng minh rằng bạn hoàn toàn bất lực trong hoàn cảnh này? Liệu trạng thái cạn kiệt dopamine hiện tại có đang đánh lừa bạn rằng tương lai không thể cải thiện không?",
        "clinical_prescription": "[PHÁC ĐỒ KÍCH HOẠT HÀNH VI CBT]\n1. Vi hành động 2 phút: Rời khỏi giường, mở cửa sổ đón ánh sáng mặt trời tự nhiên để điều hòa nhịp sinh học và kích hoạt tổng hợp Serotonin.\n2. Thiết lập bảng theo dõi Hoạt động - Làm chủ (Mastery & Pleasure Log) với thang điểm 0-10.\n3. Hạn chế tự cô lập: Duy trì ít nhất một tương tác ngắn 5 phút mỗi ngày với người thân hoặc đồng nghiệp."
    },
    {
        "id": "RULE_GAD_ANXIETY",
        "category": "ANXIETY_DISORDER",
        "trigger": "lo âu, lo lắng, bồn chồn, anxiety, gad, bất an, sợ hãi, lo sợ, căng thẳng thần kinh, run rẩy, đứng ngồi không yên",
        "title": "Rối Loạn Lo Âu Lan Tỏa (Generalized Anxiety Disorder - GAD)",
        "protocol": "Xác định hưng phấn quá mức tại Hạch hạnh nhân (Amygdala) và suy giảm ức chế tại Vỏ não trước trán. Phân biệt Lo âu Thực tế (Productive Worry - vấn đề có thể giải quyết ngay) và Lo âu Giả định (Unproductive Worry - các kịch bản 'Nếu như...').",
        "socratic_question": "Khả năng xảy ra kịch bản tồi tệ nhất theo xác suất thực tế là bao nhiêu phần trăm? Trong quá khứ, bạn đã từng vượt qua những tình huống bất định tương tự bằng nguồn lực nào?",
        "clinical_prescription": "[KỸ THUẬT QUẢN LÝ LO ÂU & PHẢN HỒI THỰC VẬT]\n1. Cửa sổ Lo âu (Worry Window): Dành cố định đúng 15 phút lúc 17h00 hàng ngày để viết ra mọi nỗi lo, nếu nỗi lo xuất hiện ngoài giờ thì ghi chú lại và tạm hoãn.\n2. Thực hiện 4 chu kỳ thở sâu 4-7-8 để kích thích Dây thần kinh phế vị (Vagus Nerve) làm chậm nhịp tim.\n3. Viết Brain Dump ra giấy: Giải phóng bộ nhớ đệm của vỏ não trước trán."
    },
    {
        "id": "RULE_PANIC_ATTACK",
        "category": "ACUTE_CRISIS",
        "trigger": "hoảng loạn, panic attack, cơn hoảng loạn, nghẹt thở, tim đập thình thịch, sắp xỉu, sắp chết, tê tay chân, hyperventilation, choáng váng",
        "title": "Cơn Hoảng Loạn Cấp Tính (Panic Attack & Panic Disorder)",
        "protocol": "Phản ứng tăng thông khí (Hyperventilation) khiến sụt giảm nồng độ CO2 trong máu, gây co thắt mạch não cục bộ và tạo cảm giác ngạt thở giả tạo. Trấn an bệnh nhân: Cơn hoảng loạn đạt đỉnh sau 10 phút và hoàn toàn vô hại về mặt tính mạng.",
        "socratic_question": "Cơ thể bạn đang trải qua phản xạ chiến-hay-biến bình thường để tự vệ. Bạn có nhận thấy rằng tim bạn đập nhanh chỉ vì adrenaline chứ không phải vì tổn thương cơ tim không?",
        "clinical_prescription": "[PHÁC ĐỒ CẤP CỨU CƠN HOẢNG LOẠN TẠI CHỖ]\n1. Kỹ thuật Định vị 5 Giác quan (Grounding 5-4-3-2-1): Gọi tên 5 vật nhìn thấy, 4 bề mặt chạm được, 3 âm thanh nghe thấy, 2 mùi ngửi được, 1 vị giác cảm nhận.\n2. Thở Hộp (Box Breathing): Hít 4s - Giữ 4s - Thở 4s - Giữ 4s để cân bằng tỷ lệ O2/CO2.\n3. Rửa mặt bằng nước lạnh kích hoạt Phản xạ Lặn Động Vật Có Vú (Mammalian Dive Reflex)."
    },
    {
        "id": "RULE_OVERTHINKING",
        "category": "COGNITIVE_DISTORTION",
        "trigger": "overthinking, overthink, suy nghĩ nhiều, nghĩ quá nhiều, lo xa, nghiền ngẫm, ruminate, quẩn quanh, không dừng suy nghĩ, não chạy loạn",
        "title": "Hội Chứng Suy Nghĩ Quá Mức & Nghiền Ngẫm (Overthinking & Rumination)",
        "protocol": "Hoạt động quá mức của Mạng lưới Chế độ Mặc định (Default Mode Network - DMN). Não bộ nhầm lẫn giữa việc 'suy nghĩ liên tục' và việc 'hành động giải quyết vấn đề'. Liệu pháp: Chuyển hóa từ câu hỏi trừu tượng 'Tại sao chuyện này lại xảy ra?' sang câu hỏi thực tiễn 'Bước tiếp theo có thể làm được là gì?'.",
        "socratic_question": "Việc tua đi tua lại tình huống này trong đầu trong 1 giờ qua đã mang lại thêm dữ kiện mới nào, hay chỉ đang làm tiêu hao năng lượng thần kinh của bạn?",
        "clinical_prescription": "[KỸ THUẬT NGẮT DÒNG SUY NGHĨ CBT]\n1. Kỹ thuật Gán nhãn Nhận thức (Cognitive Defusion): Tự nhủ 'Tôi đang có một suy nghĩ rằng...' thay vì đồng nhất 'Tôi là suy nghĩ đó'.\n2. Chuyển kênh cảm giác: Đứng dậy đi bộ nhanh 3 phút hoặc tập trung nghe nhạc sóng Alpha 432Hz.\n3. Giới hạn thời gian phân tích: Đặt chuông báo thức 10 phút để ra quyết định."
    },
    {
        "id": "RULE_INSOMNIA_SLEEP",
        "category": "SLEEP_DISORDER",
        "trigger": "mất ngủ, khó ngủ, trằn trọc, ngủ không sâu, thức trắng, insomnia, thức dậy sớm, mơ nhiều, giấc ngủ chập chờn, mệt mỏi buổi sáng",
        "title": "Rối Loạn Mất Ngủ Mạn Tính & Vệ Sinh Giấc Ngủ (Insomnia & CBT-I)",
        "protocol": "Áp dụng Phác đồ Liệu pháp Nhận thức Hành vi cho Mất ngủ (CBT-I - tiêu chuẩn vàng y khoa). Phá vỡ liên kết điều kiện hóa tiêu cực giữa 'Giường ngủ' và 'Sự thao thức/Lo âu'. Tái lập áp lực ngủ nội tại (Homeostatic Sleep Drive).",
        "socratic_question": "Liệu việc nằm cố ép mình phải ngủ có đang vô tình kích hoạt hệ thần kinh giao cảm và đẩy giấc ngủ ra xa hơn không?",
        "clinical_prescription": "[PHÁC ĐỒ VỆ SINH GIẤC NGỦ CBT-I]\n1. Quy tắc 20 phút: Nếu sau 20 phút không thể vào giấc, hãy rời khỏi giường sang phòng khác đọc sách dưới ánh sáng vàng mờ, chỉ quay lại giường khi có cơn buồn ngủ thực sự.\n2. Cố định giờ thức dậy buổi sáng kể cả cuối tuần để neo đồng hồ sinh học.\n3. Cắt toàn bộ ánh sáng xanh điện thoại và caffeine sau 15h00."
    },
    {
        "id": "RULE_BURNOUT_EXHAUSTION",
        "category": "OCCUPATIONAL_HEALTH",
        "trigger": "kiệt sức, burnout, cạn kiệt, đuối sức, quá tải, hết pin, mệt mỏi kéo dài, áp lực công việc, stress kéo dài, mất cảm xúc",
        "title": "Hội Chứng Kiệt Sức Nghề Nghiệp (Occupational Burnout Syndrome - ICD-11 QD85)",
        "protocol": "Suy kiệt trục Dưới đồi - Tuyến yên - Thượng thận (HPA Axis) do Cortisol duy trì ở mức cao kéo dài, dẫn đến suy giảm năng lượng nhận thức, tính hoài nghi và giảm hiệu quả lao động. Cần tái tạo ranh giới tâm lý (Boundary Architecture).",
        "socratic_question": "Nếu cơ thể bạn là một cỗ máy sinh học đang ở vạch pin 5%, bạn đang vay mượn năng lượng từ đâu để tiếp tục, và cái giá phải trả cho sức khỏe dài hạn là gì?",
        "clinical_prescription": "[PHỤC HỒI NĂNG LƯỢNG THẦN KINH]\n1. Nghỉ ngơi vi mô (Micro-breaks): Dừng màn hình 5 phút sau mỗi 50 phút làm việc.\n2. Phân loại ma trận Eisenhower: Cắt giảm ngay ít nhất 1 nhiệm vụ không cấp thiết trong tuần.\n3. Uống nước ấm và vận động kéo giãn cơ cổ - vai - gáy để giải phóng co thắt mạch."
    },
    {
        "id": "RULE_OCD_COMPULSION",
        "category": "ANXIETY_DISORDER",
        "trigger": "ocd, ám ảnh, cưỡng chế, sợ bẩn, kiểm tra đi kiểm tra lại, suy nghĩ xâm nhập, nghi thức, hoàn hảo quá mức",
        "title": "Rối Loạn Ám Ảnh Cưỡng Chế (Obsessive-Compulsive Disorder - OCD)",
        "protocol": "Vòng lặp bệnh lý: Ý nghĩ xâm nhập (Obsession) gây ra lo âu cực độ, dẫn đến Hành vi cưỡng chế (Compulsion) nhằm giải tỏa lo âu tức thời. Liệu pháp vàng: Phơi nhiễm và Ngăn ngừa Đáp ứng (Exposure and Response Prevention - ERP).",
        "socratic_question": "Hành động kiểm tra lặp lại này thực sự giúp bạn an toàn hơn, hay nó đang nuôi dưỡng con quái vật nghi ngờ ngày một lớn hơn?",
        "clinical_prescription": "[KỸ THUẬT PHƠI NHIỄM VÀ NGĂN ĐÁP ỨNG ERP]\n1. Trì hoãn phản ứng: Khi thôi thúc kiểm tra/rửa tay xuất hiện, bấm đồng hồ trì hoãn 3 phút trước khi làm, sau đó nâng dần lên 5 phút, 10 phút.\n2. Chấp nhận sự không chắc chắn: Lặp lại câu khẳng định 'Tôi sẵn sàng chấp nhận một mức độ rủi ro nhất định của cuộc sống'."
    },
    {
        "id": "RULE_ADHD_FOCUS",
        "category": "NEURODEVELOPMENTAL",
        "trigger": "adhd, mất tập trung, dễ xao nhãng, trì hoãn, không tập trung được, não cá vàng, đãng trí, lộn xộn, không thể ngồi yên",
        "title": "Rối Loạn Giảm Chú Ý & Rối Loạn Chức Năng Điều Hành (Adult ADHD)",
        "protocol": "Thiếu hụt dẫn truyền Dopamine và Norepinephrine tại vỏ não trước trán gây suy giảm Chức năng Điều hành (Executive Dysfunction): Trì hoãn xuất phát từ việc não bộ không nhận đủ kích thích để khởi động công việc chứ không phải do lười biếng.",
        "socratic_question": "Nhiệm vụ này đang quá lớn khiến thùy trán của bạn bị quá tải? Nếu chỉ yêu cầu bạn mở tài liệu ra và làm đúng 60 giây, bạn có thể thực hiện được không?",
        "clinical_prescription": "[PHÁC ĐỒ QUẢN TRỊ CHÚ Ý CHO NÃO BỘ]\n1. Phương pháp Pomodoro thích ứng: 20 phút làm việc tập trung cao độ + 5 phút vận động giải tỏa.\n2. Tạo môi trường thị giác tối giản: Chỉ để duy nhất một cửa sổ ứng dụng mở trên màn hình.\n3. Kỹ thuật Body Doubling: Làm việc cùng người khác hoặc bật âm thanh nền nhịp nhàng."
    },
    {
        "id": "RULE_SOCIAL_ANXIETY",
        "category": "ANXIETY_DISORDER",
        "trigger": "sợ đám đông, ngại giao tiếp, sợ bị phán xét, lo âu xã hội, social anxiety, run khi thuyết trình, sợ ánh nhìn, tự ti giao tiếp",
        "title": "Rối Loạn Lo Âu Xã Hội (Social Anxiety Disorder - SAD)",
        "protocol": "Mô hình nhận thức Clark & Wells: Tập trung chú ý hướng nội quá mức (Self-focused attention) và ảo giác rằng khuyết điểm của mình đang bị phóng đại dưới kính hiển vi của đám đông (Spotlight Effect).",
        "socratic_question": "Khi bạn nhìn người khác thuyết trình, bạn có săm soi từng lỗi nhỏ của họ như cách bạn đang tự soi mói chính mình không?",
        "clinical_prescription": "[TÁI ĐỊNH HƯỚNG CHÚ Ý RA BÊN NGOÀI]\n1. Hướng sự chú ý 100% ra xung quanh: Quan sát màu áo của người đối diện, lắng nghe âm thanh trong phòng thay vì quan sát nhịp tim của bản thân.\n2. Thực hiện các thử nghiệm hành vi nhỏ: Cố tình nói vấp một từ nhẹ nhàng để nhận ra rằng không có thảm họa nào xảy ra."
    },
    {
        "id": "RULE_BIPOLAR_MOOD",
        "category": "MOOD_DISORDER",
        "trigger": "lưỡng cực, bipolar, lúc vui lúc buồn, hưng cảm, mania, thay đổi tâm trạng thất thường, hưng phấn quá đà, bốc đồng",
        "title": "Rối Loạn Cảm Xúc Lưỡng Cực (Bipolar Affective Disorder)",
        "protocol": "Sự mất cân bằng chu kỳ sinh học thần kinh giữa các pha Hưng cảm/Hưng cảm nhẹ (Euphoria, giảm nhu cầu ngủ, nói nhanh, ý nghĩ dồn dập) và Trầm cảm cực độ. Liệu pháp bổ trợ: Liệu pháp Nhịp điệu Xã hội & Sinh học Cá nhân (IPSRT).",
        "socratic_question": "Trong những ngày bạn cảm thấy tràn trề năng lượng bất thường và không cần ngủ, các quyết định tài chính hoặc công việc đưa ra có để lại hậu quả dài hạn không?",
        "clinical_prescription": "[ỔN ĐỊNH NHỊP ĐIỆU SINH HỌC IPSRT]\n1. Duy trì nhật ký theo dõi khí sắc hàng ngày (Mood Chart) kết hợp chỉ số thời gian ngủ.\n2. Cố định tuyệt đối giờ thức dậy và giờ ăn để ổn định đồng hồ sinh học vùng dưới đồi.\n3. Tham vấn bác sĩ chuyên khoa tâm thần định kỳ để quản lý thuốc điều hòa khí sắc."
    },
    {
        "id": "RULE_PTSD_TRAUMA",
        "category": "TRAUMA_STRESS",
        "trigger": "ptsd, sang chấn, chấn thương tâm lý, flashback, ký ức tiêu cực, bóng ma tâm lý, ác mộng quá khứ, giật mình hoảng sợ",
        "title": "Rối Loạn Căng Thẳng Sau Sang Chấn (Post-Traumatic Stress Disorder - PTSD)",
        "protocol": "Ký ức đau buồn không được mã hóa đúng cách vào hồi hải mã (Hippocampus), khiến hạch hạnh nhân liên tục kích hoạt báo động như thể nguy hiểm đang tái diễn ở hiện tại. Trạng thái Siêu cảnh giác (Hypervigilance).",
        "socratic_question": "Cơ thể bạn đang phản ứng như thể mối đe dọa năm xưa đang ở ngay trong căn phòng này. Nhìn quanh xem: Bạn có đang an toàn ở thời điểm hiện tại không?",
        "clinical_prescription": "[KỸ THUẬT AN TOÀN & ĐỊNH VỊ HIỆN TẠI]\n1. Neo cảm giác an toàn: Đặt hai chân chạm vững chắc xuống sàn, đặt tay lên ngực và tự nhủ: 'Đó là chuyện quá khứ, hiện tại tôi đang an toàn'.\n2. Kỹ thuật Phơi nhiễm Kéo dài bằng văn bản (Narrative Exposure) dưới sự hỗ trợ chuyên môn."
    },
    {
        "id": "RULE_SOMATIC_HEADACHE",
        "category": "SOMATIC_REGULATION",
        "trigger": "đau đầu, chóng mặt, nghẹn họng, đau cổ vai gáy, đau dạ dày do stress, cơ thể hóa, somatic, căng cơ hàm, nặng đầu",
        "title": "Hội Chứng Rối Loạn Cơ Thể Hóa & Đau Đầu Căng Cơ (Somatization & Tension Headache)",
        "protocol": "Căng thẳng tâm lý chuyển đổi thành trương lực cơ vân tăng cao cục bộ tại cơ trán, cơ thái dương, cơ thang và cơ ức đòn chũm. Co thắt mạch máu vi mô làm giảm tưới máu vỏ não gây đau đầu âm ỉ và cảm giác vướng họng (Globus sensation).",
        "socratic_question": "Cơ hàm của bạn lúc này có đang cắn chặt không? Hãy thả lỏng hàm dưới và hạ thấp hai vai xuống 2cm, bạn cảm thấy thế nào?",
        "clinical_prescription": "[GIẢI PHÓNG TRƯƠNG LỰC CƠ JACOBSON PMR]\n1. Thả lỏng cơ hàm: Há miệng nhẹ, đặt đầu lưỡi chạm vòm miệng trên sau răng cửa để tách khớp cắn.\n2. Xoay tròn khớp vai 10 lần kết hợp thở ra chậm rãi.\n3. Chườm ấm vùng gáy và uống 250ml nước ấm."
    },
    {
        "id": "RULE_IMPOSTER_ESTEEM",
        "category": "COGNITIVE_DISTORTION",
        "trigger": "kẻ giả mạo, imposter, tự ti, kém cỏi, không xứng đáng, sợ bị phát hiện, tự ti bản thân, so sánh với người khác",
        "title": "Hội Chứng Kẻ Giả Mạo & Lòng Tự Trọng Thấp (Imposter Syndrome & Low Self-Worth)",
        "protocol": "Quy kết sai lệch nguyên nhân: Gán thành công cho 'may mắn/tình cờ' và gán thất bại cho 'sự bất tài cố hữu'. Không thể tích hợp các thành tựu thực tế vào bản sắc cá nhân.",
        "socratic_question": "Nếu một người bạn thân có những thành tích và nỗ lực giống hệt bạn, bạn có đánh giá họ là 'kẻ giả mạo may mắn' không?",
        "clinical_prescription": "[BẢNG CHỨNG CỨ KHÁCH QUAN VỀ NĂNG LỰC]\n1. Lập sổ tay 'Evidence File': Ghi lại ít nhất 5 phản hồi tích cực và dự án bạn đã tự tay hoàn thành.\n2. Tách biệt cảm xúc và sự thật: Tự nhủ 'Cảm giác nghi ngờ bản thân không đồng nghĩa với việc tôi thiếu năng lực'."
    },
    {
        "id": "RULE_COGNITIVE_DISTORTIONS",
        "category": "COGNITIVE_DISTORTION",
        "trigger": "tiêu cực, suy nghĩ tiêu cực, tự trách, bi quan, bóp méo nhận thức, đổ lỗi, phán xét bản thân, tư duy trắng đen",
        "title": "Tái Cấu Trúc Các Biến Dạng Nhận Thức (Cognitive Restructuring - David Burns)",
        "protocol": "Nhận diện 10 bẫy tư duy sai lệch: Tư duy trắng-đen (All-or-nothing), Khái quát hóa quá mức (Overgeneralization), Lọc tinh thần (Mental Filter), Đọc suy nghĩ (Mind Reading), Bói tương lai (Fortune Telling), Dán nhãn (Labeling).",
        "socratic_question": "Bạn có đang áp đặt tiêu chuẩn tuyệt đối 'phải' hoặc 'không được phép' lên chính mình không? Có góc nhìn nào đa chiều hơn cho tình huống này?",
        "clinical_prescription": "[PHIẾU GHI CHÉP SUY NGHĨ 3 CỘT (CBT THOUGHT RECORD)]\n1. Cột 1: Tình huống kích hoạt cảm xúc.\n2. Cột 2: Ý nghĩ tự động tiêu cực xuất hiện.\n3. Cột 3: Ý nghĩ thay thế thực tế và cân bằng hơn dựa trên bằng chứng khách quan."
    },
    {
        "id": "RULE_ERGONOMICS_POSTURE",
        "category": "PHYSICAL_ERGONOMICS",
        "trigger": "cổ rùa, gù lưng, nghiêng cổ, forward head, mỏi cổ, đau lưng, tư thế ngồi, slouch",
        "title": "Tư Thế Đầu Hướng Trước & Áp Lực Thần Kinh Cột Sống Cổ (Forward Head Posture)",
        "protocol": "Khi đầu gập về phía trước mỗi 2.5cm, trọng lực tác động lên đốt sống cổ C1-C7 tăng thêm 4.5kg. Căng thẳng cơ học kéo dài chèn ép dây thần kinh chẩm và gửi tín hiệu căng thẳng mạn tính về não bộ.",
        "socratic_question": "Bạn có đang rướn người sát vào màn hình để đọc chữ không?",
        "clinical_prescription": "[BÀI TẬP THU CẰM CHIN TUCK]\n1. Giữ ánh nhìn thẳng, nhẹ nhàng thu cằm về phía cổ họng tạo cằm đôi trong 5 giây, lặp lại 5 lần.\n2. Nâng độ cao màn hình ngang tầm mắt để duy trì trục thẳng sinh học."
    },
    {
        "id": "RULE_SELF_HARM_SAFETY",
        "category": "CRISIS_SAFETY",
        "trigger": "tự tử, tự hại, muốn chết, kết thúc cuộc sống, suicide, self harm, không muốn sống nữa, cùng đường",
        "title": "Giao Thức An Toàn Khủng Hoảng Lâm Sàng (Crisis Safety Protocol)",
        "protocol": "Kích hoạt ưu tiên an toàn tính mạng tuyệt đối. Trầm cảm và bế tắc cấp tính có thể điều trị và hồi phục được bằng can thiệp y khoa và tâm lý chuyên nghiệp.",
        "socratic_question": "Sự đau khổ hiện tại là thật, nhưng mong muốn kết thúc sinh mệnh chỉ là phản ánh của sự quá tải năng lượng chứ không phải là giải pháp duy nhất. Bạn có sẵn sàng để những người chuyên môn hỗ trợ bạn vượt qua giai đoạn này không?",
        "clinical_prescription": "[KÊNH LIÊN HỆ HỖ TRỢ KHẨN CẤP 24/7]\n1. Tổng đài Quốc gia Bảo vệ Sức khỏe Tinh thần & Phòng chống Khủng hoảng (Việt Nam): 1900 599 958 hoặc 115.\n2. Hãy liên hệ ngay với người thân tin cậy hoặc đến cơ sở y tế gần nhất.\n3. Bác sĩ luôn đồng hành cùng bạn và bạn không phải đơn độc đối diện với khó khăn này."
    }
]

CLINICAL_ELICITATION_PROMPTS = [
    {
        "id": "PROMPT_01",
        "target_emotion": "EQUILIBRIUM",
        "prompt_text": "Tôi cảm thấy hoàn toàn bình an và làm chủ được năng lượng tinh thần hôm nay.",
        "rationale": "Đánh giá sự đồng pha giữa khẳng định tích cực và trương lực cơ mặt."
    },
    {
        "id": "PROMPT_02",
        "target_emotion": "STRESS_COPING",
        "prompt_text": "Áp lực công việc gần đây khiến tôi cảm thấy quá tải, nhưng tôi chấp nhận giới hạn của bản thân.",
        "rationale": "Đo lường phản xạ thanh quản và độ võng lồng ngực khi thừa nhận căng thẳng."
    },
    {
        "id": "PROMPT_03",
        "target_emotion": "FOCUS_CALM",
        "prompt_text": "Hơi thở của tôi chậm lại, hai vai thả lỏng, tâm trí tập trung vào khoảnh khắc này.",
        "rationale": "Kiểm tra khả năng tự điều hòa nhịp thở và dải tần giọng nói 16kHz."
    },
    {
        "id": "PROMPT_04",
        "target_emotion": "EMOTIONAL_HONESTY",
        "prompt_text": "Tôi không cần phải luôn tỏ ra hoàn hảo trước mặt mọi người xung quanh.",
        "rationale": "Kích hoạt vùng cơ ức đòn chũm và khảo sát mức độ gượng ép biểu cảm."
    },
    {
        "id": "PROMPT_05",
        "target_emotion": "RESILIENCE",
        "prompt_text": "Mỗi trở ngại là một dữ liệu thực tế giúp tôi điều chỉnh chiến lược sống vững vàng hơn.",
        "rationale": "Đo lường năng lượng âm thanh và độ kiên định trong ngữ điệu."
    }
]

class ClinicalCounselor:
    """
    Cognitive Behavioral Therapy (CBT) & Clinical Psychiatric Knowledge Engine.
    Operates 100% Offline with zero cloud dependencies.
    Provides diagnostic insights, Socratic reframing, and interactive machine learning.
    """
    def __init__(self):
        self._ensure_knowledge_base()

    def _ensure_knowledge_base(self):
        """Synchronize custom knowledge base, ensuring all pre-trained rules are available."""
        existing_rules = []
        if config.CUSTOM_KNOWLEDGE_PATH.exists():
            try:
                with open(config.CUSTOM_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                    existing_rules = json.load(f)
            except Exception:
                existing_rules = []

        # Merge defaults: keep existing custom rules, add missing pre-trained ones
        existing_ids = {r.get("id") for r in existing_rules if isinstance(r, dict)}
        merged = list(existing_rules)
        for def_r in DEFAULT_KNOWLEDGE_RULES:
            if def_r["id"] not in existing_ids:
                merged.append(def_r)

        with open(config.CUSTOM_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"[COUNSELOR] Initialized CBT Knowledge Base ({len(merged)} clinical rules loaded) at {config.CUSTOM_KNOWLEDGE_PATH}")

    def get_all_rules(self):
        """Retrieve all registered clinical CBT intervention rules."""
        try:
            with open(config.CUSTOM_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_KNOWLEDGE_RULES

    def add_or_update_rule(self, rule_data):
        """Add or modify a CBT clinical rule in the offline knowledge database."""
        rules = self.get_all_rules()
        rule_id = rule_data.get("id") or f"RULE_{int(datetime.now().timestamp())}"
        rule_data["id"] = rule_id

        idx = next((i for i, r in enumerate(rules) if r["id"] == rule_id), None)
        if idx is not None:
            rules[idx] = rule_data
        else:
            rules.append(rule_data)

        with open(config.CUSTOM_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        return True, rule_id

    def delete_rule(self, rule_id):
        """Remove a rule from the CBT knowledge base."""
        rules = self.get_all_rules()
        updated = [r for r in rules if r["id"] != rule_id]
        if len(updated) < len(rules):
            with open(config.CUSTOM_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
                json.dump(updated, f, indent=2, ensure_ascii=False)
            return True, f"Rule {rule_id} removed."
        return False, f"Rule {rule_id} not found."

    def get_random_elicitation_prompt(self):
        """Return a random clinical prompt for vocal-multimodal evaluation in Tab 1."""
        return random.choice(CLINICAL_ELICITATION_PROMPTS)

    def get_prescription_for_assessment(self, final_score: int, emotion: str, exhaustion_level: int = 1):
        """
        Dynamically lookup the most appropriate CBT protocol and prescription from custom_knowledge.json
        based on diagnostic score, exhaustion level, and multimodal emotion classification.
        Synchronized with Module 3 (CBT Knowledge Base) of Train Studio.
        """
        rules = self.get_all_rules()
        emo_upper = (emotion or "").upper()
        matched_rule = None

        # 1. High exhaustion / burnout criteria
        if exhaustion_level >= 3 or "SAD" in emo_upper or "DEPRESSIVE" in emo_upper or final_score < 68:
            target_cats = ["BURNOUT_PREVENTION", "OCCUPATIONAL_HEALTH", "MOOD_DISORDER"]
            matched_rule = next((r for r in rules if r.get("category") in target_cats), None)

        # 2. Anxiety / Nervous tension criteria
        elif "ANXIETY" in emo_upper or "FEAR" in emo_upper or "ANGRY" in emo_upper:
            target_cats = ["ANXIETY_DISORDER", "SOMATIC_REGULATION", "ACUTE_CRISIS"]
            matched_rule = next((r for r in rules if r.get("category") in target_cats), None)

        # 3. Moderate tension / Slouching / Cognitive fatigue
        elif final_score < 80 or exhaustion_level >= 2:
            target_cats = ["SOMATIC_REGULATION", "PHYSICAL_ERGONOMICS", "COGNITIVE_DISTORTION"]
            matched_rule = next((r for r in rules if r.get("category") in target_cats), None)

        # 4. High equilibrium / wellness score (score >= 80)
        if matched_rule is None:
            target_cats = ["SOMATIC_REGULATION", "PHYSICAL_ERGONOMICS"]
            matched_rule = next((r for r in rules if r.get("category") in target_cats), None)

        if matched_rule is None and rules:
            matched_rule = rules[0]

        if matched_rule:
            presc = matched_rule.get("clinical_prescription") or matched_rule.get("protocol") or ""
            title = matched_rule.get("title", "Khuyến nghị phục hồi tâm lý")
            rule_id = matched_rule.get("id", "CBT_RULE")
            return {
                "rule_id": rule_id,
                "title": title,
                "prescription": f"[{title.upper()}]\n{presc}" if not presc.strip().startswith("[") else presc,
                "protocol": matched_rule.get("protocol", "")
            }

        return {
            "rule_id": "DEFAULT",
            "title": "Cân bằng tâm thần & Phục hồi",
            "prescription": "Chỉ số cân bằng thần kinh của bạn đạt mức chuẩn. Hãy tiếp tục duy trì tư thế ngồi thẳng, hít thở điều hòa và bổ sung nước ấm.",
            "protocol": ""
        }


    def check_learning_command(self, user_message):
        """
        Detects if user is commanding the assistant to learn/train new clinical knowledge.
        Patterns:
          - "dạy bác sĩ: [tên bệnh] : [phác đồ]"
          - "học kiến thức: [chủ đề] : [nội dung]"
          - "ghi nhớ: [bệnh] : [phác đồ]"
          - "học: [chủ đề] : [nội dung]"
        """
        msg = user_message.strip()
        patterns = [
            r'^(?:dạy bác sĩ|day bac si|học kiến thức|hoc kien thuc|ghi nhớ|ghi nho|học|hoc|dạy|day)\s*:\s*(.+?)\s*:\s*(.+)$',
            r'^(?:thêm bệnh|them benh|thêm kiến thức|them kien thuc)\s*:\s*(.+?)\s*:\s*(.+)$'
        ]
        for p in patterns:
            match = re.match(p, msg, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                content = match.group(2).strip()
                rule_id = f"CUSTOM_{int(datetime.now().timestamp())}"

                # Extract clean sub-triggers (separate base name from english parentheticals)
                topic_base = re.sub(r'\(.*?\)', '', topic).strip().lower()
                paren_terms = [p.strip().lower() for p in re.findall(r'\((.*?)\)', topic) if p.strip()]

                trigger_list = [t for t in [topic_base] + paren_terms + [topic.lower()] if len(t) > 1]
                trigger_str = ", ".join(list(dict.fromkeys(trigger_list)))

                new_rule = {
                    "id": rule_id,
                    "category": "USER_TRAINED_KNOWLEDGE",
                    "trigger": trigger_str,
                    "title": f"Kiến Thức Tự Học: {topic}",
                    "protocol": f"Phác đồ tùy chỉnh do người dùng huấn luyện về '{topic}': {content}",
                    "socratic_question": f"Khi đối diện với vấn đề '{topic}', bạn đã áp dụng những kinh nghiệm hoặc kỹ thuật cụ thể nào để tháo gỡ?",
                    "clinical_prescription": f"[PHÁC ĐỒ HUẤN LUYỆN NỘI BỘ VỀ {topic.upper()}]\n{content}"
                }
                self.add_or_update_rule(new_rule)
                return True, (
                    f"[XÁC NHẬN: ĐÃ NẠP THÀNH CÔNG VÀO BỘ NÃO LÂM SÀNG CỦA BÁC SĨ]\n\n"
                    f"Tôi đã học và lưu trữ thành công tri thức mới vào cơ sở dữ liệu ngoại tuyến:\n"
                    f"● CHỦ ĐỀ / BỆNH LÝ: {topic}\n"
                    f"● TỪ KHÓA KÍCH HOẠT: {trigger_str}\n"
                    f"● PHÁC ĐỒ ĐIỀU TRỊ: {content}\n"
                    f"● MÃ ĐỊNH DANH RULE: {rule_id}\n\n"
                    f"Kể từ bây giờ, bất cứ khi nào bạn đề cập đến '{topic}' trong cuộc trò chuyện, tôi sẽ lập tức kích hoạt và áp dụng phác đồ này để tư vấn cho bạn."
                )
        return False, None

    def search_clinical_knowledge(self, user_message, valence=0.0, arousal=0.0, exhaustion=0):
        """
        Perform intelligent semantic matching across all clinical rules in database.
        Returns the best matching rule or None.
        """
        rules = self.get_all_rules()
        msg_norm = user_message.lower().strip()
        words = set(re.findall(r'\w+', msg_norm))

        GENERIC_STOP = {"rối", "loạn", "bệnh", "hội", "chứng", "triệu", "tôi", "đang", "bị", "làm", "sao", "phải", "như", "thế", "nào", "có", "không", "disorder", "syndrome", "kiến", "thức", "tự", "học"}

        best_rule = None
        best_score = 0.0

        for r in rules:
            score = 0.0
            title = r.get("title", "").lower()
            trigger = r.get("trigger", "").lower()
            triggers = [t.strip() for t in trigger.split(",") if t.strip()]

            # 1. Direct phrase match on trigger or title
            for t in triggers:
                if t and t in msg_norm:
                    t_len = len(t.split())
                    score += 15.0 if t_len >= 2 else 8.0
                    break

            if score == 0.0:
                # 2. Distinct title words matching (require at least 2 distinct words or 100% of title words)
                dist_title_words = [w for w in re.findall(r'\w+', title) if len(w) > 2 and w not in GENERIC_STOP]
                if dist_title_words:
                    matched_dt = sum(1 for w in dist_title_words if w in words)
                    if matched_dt == len(dist_title_words):
                        score += 10.0
                    elif matched_dt >= 2 and matched_dt >= len(dist_title_words) * 0.6:
                        score += 6.0

            # 3. Biometric synergy score
            cat = r.get("category", "")
            if cat == "MOOD_DISORDER" and valence < -0.2:
                score += 1.0
            if cat == "ANXIETY_DISORDER" and arousal > 0.3:
                score += 1.0
            if cat in ("BURNOUT_PREVENTION", "OCCUPATIONAL_HEALTH") and exhaustion >= 3:
                score += 1.5

            if score > best_score:
                best_score = score
                best_rule = r

        return best_rule if best_score >= 4.0 else None

    def analyze_dialogue_with_biometrics(self, user_message, biometric_context):
        """
        Synthesize user message with real-time biometrics and offline knowledge base.
        Biometric context contains:
        {
          "emotion": "SAD / DEPRESSIVE",
          "valence": -0.45,
          "arousal": 0.20,
          "exhaustion_level": 3,
          "posture_warning": False,
          "incongruence_detected": False
        }
        Returns comprehensive, personalized clinical consultation with zero emojis.
        """
        valence = float(biometric_context.get("valence", 0.0))
        arousal = float(biometric_context.get("arousal", 0.0))
        emotion = str(biometric_context.get("emotion", "NEUTRAL / CALM"))
        exhaustion = int(biometric_context.get("exhaustion_level", 0))
        incongruence = bool(biometric_context.get("incongruence_detected", False))
        posture_warning = bool(biometric_context.get("posture_warning", False))

        msg_clean = user_message.strip()

        # 1. Check if user is issuing a teaching/learning command
        is_learn, learn_reply = self.check_learning_command(msg_clean)
        if is_learn:
            return learn_reply

        # 2. Check for simple casual greetings / check-ins
        greetings = [
            "hi", "hello", "chào", "chào bạn", "chào bác sĩ", "alo", "hey", "test", "xin chào",
            "bạn ơi", "bac si oi", "bác sĩ ơi", "ê", "e", "ơi", "co ai ko", "có ai không",
            "sao không trả lời", "sao ko tra loi", "chào anh", "chào em", "alo bac si", "alo bác sĩ",
            "bắt đầu", "start", "giúp tôi", "help", "tôi cần giúp", "bác sĩ có đó không"
        ]
        if msg_clean.lower() in greetings or len(msg_clean) <= 2:
            status_summary = f"Chỉ số: {emotion} | Valence: {valence:+.2f} | Mức kiệt sức: {exhaustion}/5"
            return (
                f"Chào bạn. Tôi là Bác sĩ Tham vấn Tâm lý Lâm sàng CBT của bạn.\n\n"
                f"[{status_summary}]\n\n"
                f"Hệ thống của tôi đã được trang bị sẵn Kho Bách Khoa Tri Thức Tâm Lý & Bệnh Học Tâm Thần Offline "
                f"(gồm Trầm cảm, Rối loạn lo âu, Cơn hoảng loạn, Overthinking, Mất ngủ, Kiệt sức, OCD, ADHD...). "
                f"Bạn có thể thoải mái hỏi tôi về bất kỳ bệnh lý tâm thần nào, chia sẻ triệu chứng của bạn, "
                f"hoặc trực tiếp dạy tôi kiến thức mới bằng cú pháp: 'Dạy bác sĩ: [Tên bệnh] : [Phác đồ]'.\n\n"
                f"Hôm nay bạn đang có thắc mắc gì hoặc cảm thấy thế nào, hãy chia sẻ cùng tôi nhé."
            )

        # 3. Search clinical knowledge base for matching disorder or topic
        matched_rule = self.search_clinical_knowledge(msg_clean, valence, arousal, exhaustion)

        response_paragraphs = []
        # Header metric line
        response_paragraphs.append(f"[TÌNH TRẠNG SINH TRẮC THỜI GIAN THỰC: {emotion} | VALENCE {valence:+.2f} | CĂNG THẲNG CẤP {exhaustion}/5]")

        # Affective Incongruence notification if detected
        if incongruence:
            response_paragraphs.append(
                "[QUAN SÁT LÂM SÀNG: MÂU THUẪN GIỮA PHÁT NGÔN & BIỂU HIỆN SINH TRẮC]\n"
                "Hệ thống phát hiện sự lệch pha: Ngữ nghĩa câu hỏi thể hiện sự kiềm chế, tuy nhiên trương lực cơ mặt "
                "và ngữ điệu ghi nhận mức tải thần kinh sâu sắc. Hãy an tâm rằng đây là không gian tham vấn an toàn tuyệt đối, "
                "bạn hoàn toàn có thể thừa nhận những mệt mỏi mà không cần gượng ép giữ vẻ ngoài bình thản."
            )

        # Posture warning if detected
        if posture_warning:
            response_paragraphs.append(
                "[CHỈNH SỬA TƯ THẾ CỘT SỐNG CỔ (ERGONOMICS)]\n"
                "Camera ghi nhận tư thế gập cổ về phía trước (Forward Head Strain). Hãy nâng cằm, kéo nhẹ hai bả vai về sau "
                "để giải phóng áp lực chèn ép lên mạch máu não."
            )

        if matched_rule:
            # Domain-specific diagnosis & reasoning
            title = matched_rule.get("title", "Tham Vấn Tâm Lý Lâm Sàng")
            protocol = matched_rule.get("protocol", "")
            socratic = matched_rule.get("socratic_question", "")
            prescription = matched_rule.get("clinical_prescription", "")

            # Tailor diagnosis paragraph based on user message and biometrics
            diagnosis = (
                f"[CHẨN ĐOÁN LÂM SÀNG & CƠ CHẾ SINH HỌC THẦN KINH: {title.upper()}]\n"
                f"Dựa trên nội dung bạn chia sẻ kết hợp cùng các chỉ số sinh trắc học hiện tại (Valence {valence:+.2f}, Arousal {arousal:+.2f}):\n"
                f"{protocol}"
            )
            response_paragraphs.append(diagnosis)

            if socratic:
                response_paragraphs.append(f"[CÂU HỎI TÁI CẤU TRÚC NHẬN THỨC SOCRATIC]\n{socratic}")

            if prescription:
                response_paragraphs.append(prescription)

        else:
            # Dynamic reflective synthesis for general emotional or open-ended inquiries
            reflection = (
                f"[PHÂN TÍCH NHẬN THỨC VÀ ĐỘNG LỰC HÀNH VI]\n"
                f"Tôi ghi nhận câu hỏi của bạn: \"{user_message}\".\n"
                f"Ở trạng thái sinh trắc {emotion} (Valence: {valence:+.2f}), não bộ của bạn đang phản hồi với các tác nhân tâm lý xã hội xung quanh. "
                "Trong tâm lý học nhận thức, mọi cảm xúc tiêu cực hay bế tắc đều là những tín hiệu sinh học lành mạnh báo hiệu rằng "
                "một ranh giới tâm lý đang bị xâm phạm hoặc một nhu cầu tinh thần chưa được lắng nghe thấu đáo."
            )
            socratic_q = (
                "[CÂU HỎI GỢI MỞ ĐA CHIỀU (SOCRATIC INQUIRY)]\n"
                "1. Điểm cốt lõi nào trong vấn đề này đang tạo ra sự bất an lớn nhất trong tâm trí bạn lúc này?\n"
                "2. Nếu đứng từ góc độ 6 tháng sau nhìn lại, điều bạn đang trăn trở hôm nay sẽ mang ý nghĩa bài học gì cho sự trưởng thành nội tâm của bạn?"
            )
            prescription = (
                "[PHÁC ĐỒ PHỤC HỒI & TỰ ĐIỀU HÒA THẦN KINH]\n"
                "1. Thực hiện 3 chu kỳ thở cơ hoành 4-7-8 với công cụ dẫn nhịp phía trên để giảm nhịp tim.\n"
                "2. Đặt tay lên lồng ngực và hít thở sâu, nhắc nhở bản thân rằng bạn đang làm chủ khoảnh khắc hiện tại.\n"
                "3. Bật âm thanh nền sóng Alpha 432Hz để đưa tần số sóng não về trạng thái thư giãn tích cực."
            )
            response_paragraphs.append(reflection)
            response_paragraphs.append(socratic_q)
            response_paragraphs.append(prescription)

        return "\n\n".join(response_paragraphs)

# Global singleton
clinical_counselor = ClinicalCounselor()

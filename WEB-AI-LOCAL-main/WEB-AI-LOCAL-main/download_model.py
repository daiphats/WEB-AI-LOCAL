import os
import urllib.request

urls = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
]

target = os.path.join("data", "face_detection_yunet_2023mar.onnx")
success = False

for url in urls:
    try:
        print(f"Dang tai model tu: {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) > 50000:  # File model thuc te la ~232 KB (232,589 bytes)
                with open(target, "wb") as f:
                    f.write(data)
                print(f">>> TAI THANH CONG! Dung luong model: {len(data):,} bytes.")
                success = True
                break
            else:
                print(f"Canh bao: File tai ve chi co {len(data)} bytes (Git LFS pointer), dang thu link khac...")
    except Exception as e:
        print(f"Loi: {e}")

if success:
    print(">>> HOAN TAT! Bay gio ban co the chay: python test_detect.py")
else:
    print("Khong the tai model tu dong, vui long kiem tra ket noi mang.")

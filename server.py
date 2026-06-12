import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import collections
import websockets
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

OUTPUT_DIR = r"C:\Users\isaca\OneDrive\Desktop\model_training\output"
TFLITE_PATH = os.path.join(OUTPUT_DIR, "sign_model.tflite")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")

print("Sunucu Başlatılıyor...")
print("Model ve etiketler yükleniyor...")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1, # Masaüstü PC gücünü kullanarak iki eli de rahatça bulması için 1 yapıldı
    min_detection_confidence=0.3, # Hassasiyet artırıldı (düşük çözünürlükte bile iki eli bulur)
    min_tracking_confidence=0.3,
    max_num_hands=2
)

def normalize_landmarks(landmarks_list):
    wrist_x = landmarks_list[0]
    wrist_y = landmarks_list[1]
    wrist_z = landmarks_list[2]

    normalized = []
    for i in range(0, len(landmarks_list), 3):
        normalized.append(landmarks_list[i] - wrist_x)
        normalized.append(landmarks_list[i + 1] - wrist_y)
        normalized.append(landmarks_list[i + 2] - wrist_z)

    max_val = max(abs(v) for v in normalized) or 1.0
    return [v / max_val for v in normalized]

async def process_frame(websocket):
    print("Yeni bir telefon bağlandı!")
    try:
        async for message in websocket:
            try:
                if not isinstance(message, bytes):
                    continue
                
                # Çözünürlüğü boyutlardan otomatik anla
                if len(message) == 76800:
                    h, w = 240, 320
                elif len(message) == 345600:
                    h, w = 480, 720
                elif len(message) == 307200:
                    h, w = 480, 640
                elif len(message) == 921600: # High (1280x720)
                    h, w = 720, 1280
                else:
                    continue
                
                # Siyah-beyaz Y kanalını doğrudan binary'den oluştur (JSON veya Base64 yok = 0 ms)
                img_gray = np.frombuffer(message, dtype=np.uint8).reshape((h, w))
                
                # Dikey mod için döndür (Android ön kameraları saat yönünün tersine 90 derece döndürülmelidir)
                if h < w:
                    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # --- ÇOK KRİTİK DÜZELTME: Aspect Ratio (En-Boy Oranı) ---
                # Model yatay masaüstü kamerasıyla (geniş ekran, 4:3 veya 16:9) eğitildiği için, 
                # dikey (dar) telefon ekranında koordinatlar yatayda sıkışır. 
                # Bu yüzden sağdan ve soldan siyah boşluklar (padding) ekleyerek görüntüyü 
                # eğitim verisindeki gibi yatay formata (4:3) sokuyoruz.
                current_h, current_w = img_gray.shape
                target_w = int(current_h * 4 / 3)
                if target_w > current_w:
                    pad_left = (target_w - current_w) // 2
                    pad_right = target_w - current_w - pad_left
                    img_gray = cv2.copyMakeBorder(img_gray, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                
                # MediaPipe RGB bekler
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                
                # Modeli eğitirken masaüstü kamerası ayna (mirror) efektiyle çalıştığı için
                # Android kamerasını da ayna gibi ters çeviriyoruz ki eller aynı gözüksün
                img_rgb = cv2.flip(img_rgb, 1)
                
                # Hata ayıklama için AI'nin gördüğü görüntüyü bilgisayar ekranında göster
                cv2.imshow("Yapay Zeka Nasil Goruyor?", img_rgb)
                cv2.waitKey(1)
                
                img_rgb.flags.writeable = False
                results = hands.process(img_rgb)

                left_hand = [0.0] * 63
                right_hand = [0.0] * 63
                hand_detected = False

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        raw = []
                        for lm in hand_landmarks.landmark:
                            raw.extend([lm.x, lm.y, lm.z])
                        
                        normalized = normalize_landmarks(raw)
                        
                        label = handedness.classification[0].label
                        if label == "Left":
                            left_hand = normalized
                        else:
                            right_hand = normalized
                    
                    hand_detected = True

                if hand_detected:
                    features = np.array(left_hand + right_hand, dtype=np.float32).reshape(1, -1)
                    interpreter.set_tensor(input_details[0]['index'], features)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    
                    class_index = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_index])
                    predicted_label = labels[class_index]

                    await websocket.send(json.dumps({
                        "type": "prediction",
                        "label": predicted_label,
                        "confidence": confidence,
                        "hands": len(results.multi_hand_landmarks)
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "prediction",
                        "label": "",
                        "confidence": 0.0,
                        "hands": 0
                    }))
                    
            except Exception as e:
                print(f"Frame işleme hatası: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("Telefon bağlantısı koptu.")

async def main():
    print("WebSocket Sunucusu 8765 portunda başlatılıyor...")
    print("Telefonunuzdan şu IP adresine bağlanın: 10.104.1.57:8765")
    async with websockets.serve(process_frame, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

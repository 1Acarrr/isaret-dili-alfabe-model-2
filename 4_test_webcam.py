import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = r"C:\Users\isaca\OneDrive\Desktop\model_training\output"
TFLITE_PATH = os.path.join(OUTPUT_DIR, "sign_model.tflite")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")

print("TFLite Modeli ve etiketler yukleniyor...")

# Etiketleri yukle
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# TFLite Modelini yukle (Yeni saf FP32 model)
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model yuklendi! Kamera baslatiliyor...")

def normalize_landmarks(landmarks_list):
    """El koordinatlarini bilege gore normalize eder."""
    wrist_x = landmarks_list[0]
    wrist_y = landmarks_list[1]
    wrist_z = landmarks_list[2]

    normalized = []
    for i in range(0, len(landmarks_list), 3):
        normalized.append(landmarks_list[i]     - wrist_x)
        normalized.append(landmarks_list[i + 1] - wrist_y)
        normalized.append(landmarks_list[i + 2] - wrist_z)

    max_val = max(abs(v) for v in normalized) or 1.0
    normalized = [v / max_val for v in normalized]
    return normalized

# MediaPipe ayarları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# --- KELIME BIRLESTIRICI (WORD BUILDER) DEGISKENLERI ---
current_word = ""
predictions_buffer = []
FRAMES_TO_CONFIRM = 5  # Bir harfi kaydetmek icin ayni hareketin 5 kare (yaklasik 0.2 saniye) surmesi lazim
last_confirmed = None
frames_since_no_hand = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan goruntu alinamadi.")
            break

        # Aynadaki gibi görünmesi için görüntüyü çevir (flip)
        image = cv2.flip(image, 1)
        
        # BGR'den RGB'ye çevir (MediaPipe RGB bekler)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Performans artışı için writeable=False yapıyoruz
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        
        image_rgb.flags.writeable = True

        left_hand = [0.0] * 63
        right_hand = [0.0] * 63
        hand_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            frames_since_no_hand = 0
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # El iskeletini ekrana ciz
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Koordinatlari cikar
                raw = []
                for lm in hand_landmarks.landmark:
                    raw.extend([lm.x, lm.y, lm.z])

                normalized = normalize_landmarks(raw)

                # MediaPipe etiketini gerçek ele göre ata
                label = handedness.classification[0].label
                if label == "Left":
                    left_hand = normalized
                else:
                    right_hand = normalized

                hand_detected = True

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 30)
            font_large = ImageFont.truetype("arial.ttf", 50)
        except IOError:
            font = ImageFont.load_default()
            font_large = ImageFont.load_default()

        if hand_detected:
            # 126 ozelligi birlestir (63 sol + 63 sag)
            features = np.array(left_hand + right_hand, dtype=np.float32).reshape(1, -1)
            
            # Tahmin yap (TFLite uzerinden)
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            class_index = np.argmax(prediction[0])
            confidence = prediction[0][class_index]
            predicted_label = labels[class_index]
            
            # Ekranin sol ustunde anlik durumu HER ZAMAN goster (guven dusuk olsa bile)
            status_text = f"Anlik: {predicted_label} ({confidence:.2f})"
            try:
                bbox = draw.textbbox((10, 10), status_text, font=font)
                draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0,0,0))
            except AttributeError:
                draw.rectangle([5, 5, 300, 50], fill=(0,0,0))
            
            draw.text((10, 10), status_text, font=font, fill=(0, 255, 0))

            # Guven %40'tan yuksekse isleme al (Kelime birlestirici)
            if confidence > 0.4:
                # --- KELIME BIRLESTIRICI MANTIGI ---
                predictions_buffer.append(predicted_label)
                if len(predictions_buffer) > FRAMES_TO_CONFIRM:
                    predictions_buffer.pop(0)
                
                # Eger son 'FRAMES_TO_CONFIRM' karenin tamami ayni harfse:
                if len(predictions_buffer) == FRAMES_TO_CONFIRM and all(x == predicted_label for x in predictions_buffer):
                    if predicted_label != last_confirmed:
                        # Harfi isleme al
                        if predicted_label == "space":
                            current_word += " "
                        elif predicted_label == "del":
                            current_word = current_word[:-1]
                        elif predicted_label == "nothing":
                            pass # Bir sey yapma
                        else:
                            current_word += predicted_label
                        
                        last_confirmed = predicted_label
                        predictions_buffer.clear() # Islem sonrasi bufferi temizle, hizli yazmayi onle
        else:
            # El ekranda yoksa bekleme suresini artir
            frames_since_no_hand += 1
            if frames_since_no_hand > 10:
                last_confirmed = None # El indirildiginde ayni harfi tekrar yazabilmek icin resetle
                predictions_buffer.clear()

        # Ekranin alt kisminda olusturulan KELIMEYI goster
        word_text = f"Kelime: {current_word}"
        try:
            # Ekran cozunurlugune gore alt kisim koordinatlari: Y=400 civari (standart 640x480 kamerada)
            bbox_word = draw.textbbox((10, 400), word_text, font=font_large)
            draw.rectangle([bbox_word[0]-10, bbox_word[1]-10, bbox_word[2]+10, bbox_word[3]+10], fill=(0,0,0))
        except AttributeError:
            draw.rectangle([0, 390, 640, 460], fill=(0,0,0))
            
        draw.text((10, 400), word_text, font=font_large, fill=(255, 255, 0))
        
        # Çizimleri bitir ve BGR'ye dönüştür
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Görüntüyü ekrana bas
        cv2.imshow('Akici Isaret Dili Yazici', image)
        
        # Cikis icin 'q' tusuna basilmasini bekle (Gecikmeyi 1ms yaptik = hizli FPS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

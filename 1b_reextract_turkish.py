# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================
 TURKCE KARAKTER YENIDEN CIKARTICI
=============================================================
 C, G, I, O, U, S harfleri icin MediaPipe'i daha dusuk
 guven esigi ile yeniden calistirir ve mevcut CSV'ye ekler.

 Denenecek yontemler:
   1. Dusuk guven esigi (0.3)
   2. Daha agir model (complexity=1)
   3. Goruntu on isleme (kontrast artirma)
=============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# Ayarlar
DATASET_PATH = r"C:\Users\isaca\OneDrive\Desktop\dataset"
EXISTING_CSV = r"C:\Users\isaca\OneDrive\Desktop\model_training\data\landmarks.csv"
OUTPUT_CSV   = r"C:\Users\isaca\OneDrive\Desktop\model_training\data\landmarks.csv"

# Türkçe karakter klasörleri
TURKISH_CLASSES = ["Ç", "Ö", "Ü", "Ğ", "İ", "Ş"]

# Düşük güven eşiği
MIN_DETECTION_CONF = 0.3
MIN_TRACKING_CONF  = 0.3


def normalize_landmarks(landmarks_list):
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


def preprocess_image(img):
    """Kontrastı artır — bulanık/karanlık görseller için."""
    # CLAHE ile kontrast artırma
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


def extract_landmarks(image_path, hands_detector, use_preprocess=False):
    # Türkçe karakter içeren yolları okumak için np.fromfile kullanıyoruz (Windows'ta cv2.imread hata verir)
    img_array = np.fromfile(str(image_path), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    if use_preprocess:
        img = preprocess_image(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = hands_detector.process(img_rgb)
    img_rgb.flags.writeable = True

    left_hand  = [0.0] * 63
    right_hand = [0.0] * 63
    hand_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
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

    return (left_hand + right_hand) if hand_detected else None


def main():
    print("=" * 60)
    print(" Turkce Karakter Yeniden Cikartici")
    print("=" * 60)

    # Mevcut CSV'yi yükle
    if os.path.exists(EXISTING_CSV):
        df_existing = pd.read_csv(EXISTING_CSV)
        print(f"Mevcut CSV: {len(df_existing)} ornek")
        # Türkçe sınıfları temizle (varsa)
        df_existing = df_existing[~df_existing['label'].isin(TURKISH_CLASSES)]
        print(f"Turkce siniflar temizlendi. Kalan: {len(df_existing)} ornek")
    else:
        print("Mevcut CSV bulunamadi!")
        return

    all_new_data = []

    # Her iki model karmaşıklığını dene
    for complexity in [0, 1]:
        print(f"\n--- Model Complexity={complexity} ---")

        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=complexity,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        ) as hands:

            for class_name in TURKISH_CLASSES:
                class_dir = Path(DATASET_PATH) / class_name
                if not class_dir.exists():
                    print(f"[UYARI] Klasor bulunamadi: {class_dir}")
                    continue

                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))

                if not images:
                    print(f"[UYARI] {class_name}: Goruntu yok")
                    continue

                success = 0
                for img_path in tqdm(images, desc=f"  {class_name}(c={complexity})", ncols=70):
                    # Önce normal dene
                    lm = extract_landmarks(img_path, hands, use_preprocess=False)
                    if lm is None:
                        # Başarısız olursa kontrast artırarak dene
                        lm = extract_landmarks(img_path, hands, use_preprocess=True)

                    if lm is not None:
                        all_new_data.append([class_name] + lm)
                        success += 1

                print(f"  [OK] {class_name}: {success}/{len(images)} basarili")

    # Sonuçları göster
    print("\n" + "=" * 60)
    print(" Yeniden Cikartma Sonuclari")
    print("=" * 60)

    if not all_new_data:
        print("[HATA] Hic ornek cikarilamadi!")
        print("Goruntuleri manuel kontrol edin.")
        return

    columns = ["label"] + [f"f{i}" for i in range(126)]
    df_new = pd.DataFrame(all_new_data, columns=columns)

    # Tekrar eden görüntüleri kaldır (complexity=0 ve 1 ikisi de bulduysa)
    df_new = df_new.drop_duplicates()

    print("\nYeni sinif dagilimi:")
    print(df_new['label'].value_counts().to_string())

    # Birleştir ve kaydet
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[KAYIT] Guncellendi: {OUTPUT_CSV}")
    print(f"   Toplam: {len(df_combined)} ornek, {df_combined['label'].nunique()} sinif")
    print("\nTum sinif dagilimi:")
    print(df_combined['label'].value_counts().to_string())
    print("\n[TAMAMLANDI] Sonraki adim: python 2_train_model.py")


if __name__ == "__main__":
    main()

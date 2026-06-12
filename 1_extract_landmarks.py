# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================
 AŞAMA 1: MediaPipe ile Landmark Çıkarma
=============================================================
 Veri setindeki her görüntüden el iskelet koordinatlarını
 çıkarır ve bir CSV dosyasına kaydeder.

 Çıktı: data/landmarks.csv
   - Her satır: [label, x0, y0, z0, ..., x125, y125, z125]
   - Toplam: 1 etiket + 126 koordinat = 127 sütun
   - İki el: Sol el (0-62), Sağ el (63-125)
   - Görünmeyen el: 0.0 ile doldurulur
=============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm

# ─── Ayarlar ────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\isaca\OneDrive\Desktop\dataset"
OUTPUT_DIR   = r"C:\Users\isaca\OneDrive\Desktop\model_training\data"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "landmarks.csv")

# MediaPipe ayarları — en hafif mod (eski telefon uyumlu)
MIN_DETECTION_CONF = 0.5
MODEL_COMPLEXITY   = 0   # 0=Lite, 1=Full (daha yavaş)
MAX_HANDS          = 2
# ────────────────────────────────────────────────────────────


def normalize_landmarks(landmarks_list):
    """
    El landmark'larını bilekten (0. nokta) görece koordinatlara
    normalize eder. Bu sayede elin ekran konumundan bağımsız,
    sadece el şekli tanınır.
    """
    # Bilek (wrist) noktasını referans al
    wrist_x = landmarks_list[0]
    wrist_y = landmarks_list[1]
    wrist_z = landmarks_list[2]

    normalized = []
    for i in range(0, len(landmarks_list), 3):
        normalized.append(landmarks_list[i]     - wrist_x)
        normalized.append(landmarks_list[i + 1] - wrist_y)
        normalized.append(landmarks_list[i + 2] - wrist_z)

    # -1 ile 1 arasına ölçekle
    max_val = max(abs(v) for v in normalized) or 1.0
    normalized = [v / max_val for v in normalized]
    return normalized


def extract_landmarks_from_image(image_path, hands_detector):
    """
    Tek bir görüntüden her iki elin landmark'larını çıkarır.

    Dönüş: [sol_el_63_koordinat + sağ_el_63_koordinat]  (uzunluk=126)
           El görünmüyorsa o bölge 0.0 ile doldurulur.
           Hiç el yoksa None döner.
    """
    img_array = np.fromfile(str(image_path), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # BGR → RGB (MediaPipe RGB bekler)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Görüntüyü yaz korumasına al (performans)
    img_rgb.flags.writeable = False
    results = hands_detector.process(img_rgb)
    img_rgb.flags.writeable = True

    # Başlangıçta her iki eli sıfırla
    left_hand  = [0.0] * 63
    right_hand = [0.0] * 63
    hand_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Ham koordinatları düz listeye al
            raw = []
            for lm in hand_landmarks.landmark:
                raw.extend([lm.x, lm.y, lm.z])

            # Normalize et
            normalized = normalize_landmarks(raw)

            # MediaPipe 'Left'/'Right' etiketini gerçek ele göre ata
            label = handedness.classification[0].label
            if label == "Left":
                left_hand = normalized
            else:
                right_hand = normalized

            hand_detected = True

    if not hand_detected:
        return None  # Hiç el bulunamadı, bu görüntüyü atla

    return left_hand + right_hand  # 126 değer


def main():
    print("=" * 60)
    print(" MediaPipe Landmark Çıkarıcı")
    print("=" * 60)

    # Çıktı dizinini oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sınıf dizinlerini bul
    dataset_path = Path(DATASET_PATH)
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    print(f"\nToplam sınıf sayısı: {len(classes)}")
    print(f"Sınıflar: {classes}\n")

    all_data = []
    class_stats = {}

    # MediaPipe Hands başlat
    with mp.solutions.hands.Hands(
        static_image_mode=True,    # Fotoğraf modu (video değil)
        max_num_hands=MAX_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONF,
    ) as hands:

        for class_name in classes:
            class_dir = dataset_path / class_name
            images = list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png"))

            if not images:
                print(f"[UYARI] {class_name}: Goruntu bulunamadi, atlandi.")
                continue

            success = 0
            failed  = 0

            for img_path in tqdm(images, desc=f"{class_name:>4}", ncols=70):
                landmarks = extract_landmarks_from_image(img_path, hands)
                if landmarks is not None:
                    all_data.append([class_name] + landmarks)
                    success += 1
                else:
                    failed += 1

            class_stats[class_name] = {"success": success, "failed": failed}
            print(f"   [OK] {success} basarili  [ATLA] {failed} atlandi")

    # ─── İstatistik Özeti ───────────────────────────────────
    print("\n" + "=" * 60)
    print(" Ozet")
    print("=" * 60)
    total_success = sum(s["success"] for s in class_stats.values())
    total_failed  = sum(s["failed"]  for s in class_stats.values())
    print(f"Toplam basarili ornek : {total_success}")
    print(f"Toplam atlanan goruntu: {total_failed}")

    # ─── CSV'ye Kaydet ──────────────────────────────────────
    if not all_data:
        print("\n[HATA] Hic landmark cikarilamadi!")
        print("   MediaPipe el bulamadi. Veri setini kontrol edin.")
        sys.exit(1)

    columns = ["label"] + [f"f{i}" for i in range(126)]
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[KAYIT] {OUTPUT_CSV}")
    print(f"   Boyut: {df.shape[0]} satir x {df.shape[1]} sutun")
    print("\nSinif dagilimi:")
    print(df["label"].value_counts().to_string())
    print("\n[TAMAMLANDI] Landmark cikarma bitti!")
    print("   Sonraki adim: python 2_train_model.py")


if __name__ == "__main__":
    import mediapipe as mp
    main()

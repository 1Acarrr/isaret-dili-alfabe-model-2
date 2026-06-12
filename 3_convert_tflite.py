# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================
 AŞAMA 3: TFLite Dönüşümü ve Quantization (DNN)
=============================================================
 Eğitilen DNN modelini Flutter'a uygun TFLite formatına
 çevirir ve Int8 quantization uygular.

 Int8 Quantization faydaları:
   - Boyut: ~4× küçülür
   - Hız  : ~2-3× artar (eski telefonlarda kritik!)

 Çıktı:
   output/sign_model.tflite        ← Flutter'a kopyalanacak
   output/sign_model_info.txt      ← Model bilgileri
=============================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# ─── Ayarlar ────────────────────────────────────────────────
MODEL_PATH       = r"C:\Users\isaca\OneDrive\Desktop\model_training\output\best_model.keras"
LANDMARKS_CSV    = r"C:\Users\isaca\OneDrive\Desktop\model_training\data\landmarks.csv"
OUTPUT_DIR       = r"C:\Users\isaca\OneDrive\Desktop\model_training\output"
TFLITE_PATH      = os.path.join(OUTPUT_DIR, "sign_model.tflite")
INFO_PATH        = os.path.join(OUTPUT_DIR, "sign_model_info.txt")

NUM_FEATURES     = 126
NUM_CALIB_SAMPLES = 200   # Quantization kalibrasyonu için örnek sayısı
# ────────────────────────────────────────────────────────────


def load_calibration_data():
    """
    Int8 quantization için kalibrasyon verisi hazırlar.
    CSV'den rastgele örnekler alınır. (Giriş: 126)
    """
    df = pd.read_csv(LANDMARKS_CSV)
    feature_cols = [c for c in df.columns if c != "label"]

    # Rastgele örnekler seç
    sample = df.sample(
        n=min(NUM_CALIB_SAMPLES, len(df)),
        random_state=42
    )
    X = sample[feature_cols].values.astype(np.float32)
    return X


def representative_dataset_gen():
    """TFLite quantization için kalibrasyon veri üreteci."""
    calib_data = load_calibration_data()
    for i in range(len(calib_data)):
        yield [calib_data[i:i+1]]   # Batch boyutu 1


def convert_and_quantize(model_path, output_path):
    """Modeli yükler, optimize eder ve TFLite'a çevirir."""

    print("📦 Keras modeli yükleniyor...")
    model = tf.keras.models.load_model(model_path)
    print(f"   Giriş şekli : {model.input_shape}")
    print(f"   Çıkış şekli : {model.output_shape}")

    # Dönüştürücü (Converter)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Hiçbir kuantizasyon veya optimizasyon yapmıyoruz (Saf FP32 TFLite)
    # Bu, orijinal Keras modelinin %100 aynısını üretir ve güven skorları korunur.

    print("\n⚙️  TFLite'a dönüştürülüyor (Saf FP32)...")
    tflite_model = converter.convert()

    # Kaydet
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def analyze_model(model_path, tflite_path, tflite_model):
    """Boyut ve hız karşılaştırması yapar."""

    keras_size  = os.path.getsize(model_path) / 1024 / 1024   # MB
    tflite_size = len(tflite_model) / 1024                     # KB

    print("\n" + "=" * 50)
    print(" Model Boyut Karşılaştırması")
    print("=" * 50)
    print(f"  Keras modeli (.keras) : {keras_size:.2f} MB")
    print(f"  TFLite modeli         : {tflite_size:.1f} KB")
    print(f"  Küçülme oranı         : {keras_size * 1024 / tflite_size:.1f}×")

    # TFLite model detayları
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    info_text = f"""
TFLite Model Bilgileri (DNN)
============================
Dosya: {tflite_path}
Boyut: {tflite_size:.1f} KB

Giriş:
  - Şekil: {input_details[0]['shape']}
  - Tip  : {input_details[0]['dtype']}

Çıkış:
  - Şekil: {output_details[0]['shape']}
  - Tip  : {output_details[0]['dtype']}

Keras Boyutu  : {keras_size:.2f} MB
TFLite Boyutu : {tflite_size:.1f} KB
Küçülme Oranı : {keras_size * 1024 / tflite_size:.1f}×

Flutter Entegrasyon Notu:
  - Giriş boyutu: [1, {NUM_FEATURES}]  (batch=1, features=126)
  - Buffer'a (geçmişe) gerek YOK. Anlık kareyi gönderin.
  - Sol el(63) + Sağ el(63) = {NUM_FEATURES} değer
"""

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write(info_text)

    print(info_text)
    return info_text


def run_sanity_check(tflite_path):
    """Dönüştürülen modeli basit bir test girdisiyle çalıştırır."""
    print("🧪 Sağlık kontrolü (sanity check)...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Rastgele test girdisi (1, 126)
    test_input = np.random.uniform(-1, 1, (1, NUM_FEATURES)).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    predicted_class = np.argmax(output[0])
    confidence      = output[0][predicted_class] * 100

    print(f"   Test girdisi şekli : {test_input.shape}")
    print(f"   Çıkış şekli        : {output.shape}")
    print(f"   Tahmin sınıfı      : {predicted_class}")
    print(f"   Güven skoru        : {confidence:.1f}%")
    print("   ✅ Model sorunsuz çalışıyor!")


def print_flutter_integration_guide():
    """Flutter'a entegrasyon için kısa rehber yazdırır."""
    guide = """
╔══════════════════════════════════════════════════════════╗
║          Flutter Entegrasyon Rehberi (DNN)              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Kopyalanacak dosyalar:                               ║
║     output/sign_model.tflite  → assets/                 ║
║     output/labels.txt         → assets/                 ║
║                                                          ║
║  2. pubspec.yaml:                                        ║
║     flutter:                                             ║
║       assets:                                            ║
║         - assets/sign_model.tflite                      ║
║         - assets/labels.txt                             ║
║                                                          ║
║  3. pubspec.yaml bağımlılıkları:                         ║
║     tflite_flutter: ^0.10.4                             ║
║     camera: ^0.10.5+9                                   ║
║                                                          ║
║  4. Model giriş formatı:                                ║
║     [1, 126] → Float32 (Array/List)                     ║
║     * DİKKAT: Buffer / Sequence TUTMAYIN! Sadece        ║
║       o anki kameradan gelen tek kareyi gönderin.       ║
║     * Sol el: index 0-62, Sağ el: index 63-125          ║
║     * Görünmeyen el varsa: 0.0 ile doldurun             ║
║                                                          ║
║  5. Performans ipuçları:                                 ║
║     - Model aşırı hızlı çalışacaktır. İsterseniz her    ║
║       kareyi işleyebilirsiniz veya pil tasarrufu için   ║
║       saniyede 15 kare işleyebilirsiniz.                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(guide)


# ─── ANA PROGRAM ────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" TFLite Dönüşümü — Saf FP32 Model (DNN)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ HATA: Model bulunamadı: {MODEL_PATH}")
        print("   Önce: python 2_train_model.py")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dönüştür ve quantize et
    tflite_model = convert_and_quantize(MODEL_PATH, TFLITE_PATH)
    print(f"\n✅ TFLite modeli kaydedildi: {TFLITE_PATH}")

    # 2. Analiz et
    analyze_model(MODEL_PATH, TFLITE_PATH, tflite_model)

    # 3. Sağlık kontrolü
    run_sanity_check(TFLITE_PATH)

    # 4. Flutter rehberi
    print_flutter_integration_guide()

    print("=" * 60)
    print(" Tüm adımlar tamamlandı! 🎉")
    print("=" * 60)
    print(f"\n Flutter'a aktarılacak dosyalar:")
    print(f"   📱 {TFLITE_PATH}")
    print(f"   🏷️  {OUTPUT_DIR}\\labels.txt")


if __name__ == "__main__":
    main()

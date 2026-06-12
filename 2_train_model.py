# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
=============================================================
 AŞAMA 2: DNN Model Eğitimi (Statik İşaretler İçin)
=============================================================
 landmarks.csv dosyasından veri okur, (126,) boyutlu girdi
 alan hızlı ve kararlı bir DNN modeli eğitir.
 
 Strateji:
   - Statik işaretler: Anlık kareye gürültü eklenerek artırılır
   - LSTM/Sekans mantığı kaldırıldı, sadece anlık konuma odaklanılır
   - Giriş boyutu: (126,)
   - Çıkış: Sınıf sayısı (softmax)

 Çıktı:
   output/best_model.keras
   output/labels.txt
   output/training_history.png
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ─── Ayarlar ────────────────────────────────────────────────
LANDMARKS_CSV  = r"C:\Users\isaca\OneDrive\Desktop\model_training\data\landmarks.csv"
OUTPUT_DIR     = r"C:\Users\isaca\OneDrive\Desktop\model_training\output"
MODEL_PATH     = os.path.join(OUTPUT_DIR, "best_model.keras")
LABELS_PATH    = os.path.join(OUTPUT_DIR, "labels.txt")
HISTORY_PLOT   = os.path.join(OUTPUT_DIR, "training_history.png")
CONF_MATRIX    = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

NUM_FEATURES   = 126   # Sol el (63) + Sağ el (63)
NOISE_STD      = 0.015 # Veri artırımı için gürültü şiddeti artırıldı
AUGMENT_TIMES  = 20    # Her örneği 20 kez geometrik varyasyonla artır
BATCH_SIZE     = 64
EPOCHS         = 100
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.10
RANDOM_SEED      = 42
# ────────────────────────────────────────────────────────────

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── 1. Veri Yükleme ────────────────────────────────────────
def load_data():
    print("📂 Veri yükleniyor...")
    df = pd.read_csv(LANDMARKS_CSV)
    
    # "nothing" sınıfını eğitimden çıkarıyoruz (el yok mantığı)
    df = df[df["label"] != "nothing"]
    
    print(f"   {len(df)} örnek, {df['label'].nunique()} sınıf yüklendi.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)   # (N, 126)
    y = df["label"].values

    return X, y


import math

# ─── 2. Gelişmiş Veri Artırımı (Geometric Augmentation) ─────
def augment_data(X, y, augment=True):
    """
    Vektörize edilmiş aşırı hızlı Data Augmentation algoritması.
    """
    print("🔄 Gelişmiş Veri Artırımı (Data Augmentation) uygulanıyor...")
    if not augment:
        return X, y
        
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(AUGMENT_TIMES):
        X_new = X.copy()
        
        # 1. Ölçeklendirme (Her satır için tek bir rastgele değer)
        scales = np.random.uniform(0.8, 1.2, size=(len(X), 1))
        X_new *= scales
        
        # 2. Döndürme
        angles = np.random.uniform(-15, 15, size=(len(X), 1))
        thetas = np.radians(angles)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        for i in range(0, 126, 3):
            x_val = X_new[:, i]
            y_val = X_new[:, i+1]
            
            new_x = x_val * cos_t[:, 0] - y_val * sin_t[:, 0]
            new_y = x_val * sin_t[:, 0] + y_val * cos_t[:, 0]
            
            mask = (X[:, i] != 0.0) | (X[:, i+1] != 0.0) | (X[:, i+2] != 0.0)
            X_new[mask, i] = new_x[mask]
            X_new[mask, i+1] = new_y[mask]
            
        # 3. Gürültü
        noise = np.random.normal(0, NOISE_STD, X_new.shape)
        X_new += noise
        
        X_new = np.clip(X_new, -1.0, 1.0)
        X_aug.append(X_new)
        y_aug.append(y)
        
    X_final = np.vstack(X_aug).astype(np.float32)
    y_final = np.concatenate(y_aug)
    print(f"   Toplam örnek (artırımdan sonra): {len(X_final)}")
    return X_final, y_final


# ─── 3. Etiket Kodlama ──────────────────────────────────────
def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = tf.keras.utils.to_categorical(y_enc)
    return y_cat, le


# ─── 4. Model Mimarisi (DNN) ────────────────────────────────
def build_model(num_classes):
    """
    DNN (Dense Neural Network) Modeli
    - Zaman/sekans bağımsızdır, anlık kareye bakar
    - Çok hızlı eğitilir ve çalışır
    - L2 regularization + Dropout → overfitting önlenir
    """
    inp = layers.Input(shape=(NUM_FEATURES,), name="input")

    # ── Dense Blok 1 ─────────────────────────────────────────
    x = layers.Dense(
        512, # Kapasite 256'dan 512'ye çıkarıldı
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1"
    )(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="drop_1")(x)

    # ── Dense Blok 2 ─────────────────────────────────────────
    x = layers.Dense(
        256, # Kapasite 128'den 256'ya çıkarıldı
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_2"
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    # ── Dense Blok 3 ─────────────────────────────────────────
    x = layers.Dense(
        128, # Kapasite 64'ten 128'e çıkarıldı
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_3"
    )(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(0.3, name="drop_3")(x)

    # ── Çıkış ────────────────────────────────────────────────
    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="output"
    )(x)

    model = models.Model(inputs=inp, outputs=out, name="sign_language_dnn")
    return model


# ─── 5. Callback'ler ────────────────────────────────────────
def get_callbacks():
    return [
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, "training_log.csv")
        ),
    ]


# ─── 6. Grafik Çizimi ───────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Eğitim Geçmişi (DNN)", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Eğitim")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama")
    axes[0].set_title("Doğruluk (Accuracy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Eğitim")
    axes[1].plot(history.history["val_loss"], label="Doğrulama")
    axes[1].set_title("Kayıp (Loss)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT, dpi=150, bbox_inches="tight")
    print(f"📊 Eğitim grafiği kaydedildi: {HISTORY_PLOT}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, linewidths=0.5
    )
    plt.title("Karışıklık Matrisi (Confusion Matrix)", fontsize=14)
    plt.ylabel("Gerçek Etiket")
    plt.xlabel("Tahmin")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX, dpi=150, bbox_inches="tight")
    print(f"📊 Karışıklık matrisi kaydedildi: {CONF_MATRIX}")


# ─── ANA PROGRAM ────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" DNN (Dense) Model Eğitimi")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Veriyi yükle
    X, y = load_data()

    # 2. Etiketleri kodla
    y_cat, le = encode_labels(y)
    num_classes = y_cat.shape[1]
    class_names = le.classes_
    print(f"   Sınıf sayısı: {num_classes}")
    print(f"   Sınıflar: {list(class_names)}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"   ✅ Etiketler kaydedildi: {LABELS_PATH}")

    # 3. Veri artırımı
    X_aug, y_aug_raw = augment_data(X, y, augment=True)

    y_aug_enc = le.transform(y_aug_raw)
    y_aug_cat = tf.keras.utils.to_categorical(y_aug_enc, num_classes=num_classes)

    # 4. Train / Validation / Test böl
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug_cat,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_aug_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_SEED,
        stratify=np.argmax(y_train, axis=1)
    )

    print(f"\n📊 Veri bölümü:")
    print(f"   Eğitim   : {X_train.shape[0]} örnek")
    print(f"   Doğrulama: {X_val.shape[0]} örnek")
    print(f"   Test     : {X_test.shape[0]} örnek")

    # 5. Modeli oluştur
    print("\n🏗️  Model oluşturuluyor...")
    model = build_model(num_classes)
    model.summary()

    # 6. Derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 7. Eğit
    print(f"\n🚀 Eğitim başlıyor... (max {EPOCHS} epoch)")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 8. Test değerlendirmesi
    print("\n📋 Test sonuçları:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy : {test_acc * 100:.2f}%")
    print(f"   Test Loss     : {test_loss:.4f}")

    # 9. Sınıf bazlı rapor
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test, axis=1)
    y_pred_names = le.inverse_transform(y_pred)
    y_true_names = le.inverse_transform(y_true)

    print("\n📋 Sınıf Bazlı Rapor:")
    print(classification_report(y_true_names, y_pred_names, zero_division=0))

    # 10. Grafikleri kaydet
    plot_history(history)
    plot_confusion_matrix(y_true_names, y_pred_names, class_names)

    print("\n✅ Eğitim tamamlandı!")
    print(f"   En iyi model: {MODEL_PATH}")
    print(f"   Sonraki adım: python 3_convert_tflite.py")

if __name__ == "__main__":
    main()

# SIGNA - İşaret Dili Yapay Zeka Model Eğitimi

Bu repository, **SIGNA** mobil uygulamasında kullanılan el işaretlerini tanıyan TensorFlow Lite (.tflite) yapay zeka modelinin eğitim aşamalarını, veri işleme kodlarını ve test scriptlerini içerir.

## Eğitim Aşamaları ve Dosyalar

Projeyi veya modeli geliştirmek isterseniz aşağıdaki adımları sırasıyla uygulayabilirsiniz:

### 1. Veri Hazırlama ve Landmark Çıkarma
* **`1_extract_landmarks.py`**: Ham görüntü veya videolardan Mediapipe kullanarak ellerdeki 21 referans noktasını (landmark) çıkarır ve bu koordinatları eğitim için uygun bir formata (CSV vs.) dönüştürür.
* **`1b_reextract_turkish.py`**: İşaret dili verilerini Türkçe etiketleme/kategorize etme veya tekrar işleme işlemlerini yürütür.

## Model Mantığı ve Başarı Oranı (Accuracy)

Bu model, ham görüntü işlemek (Image Processing - CNN) yerine **Landmark Tabanlı Vektörel Sınıflandırma (DNN - Deep Neural Network)** kullanır. 
* **Çalışma Mantığı:** İlk aşamada Google MediaPipe kullanılarak kullanıcının sağ ve sol ellerinden toplam 42 referans noktası (x, y, z koordinatları) çıkarılır. Bu noktalar merkeze göre normalize edilerek 126 elemanlı bir vektöre `[1, 126]` dönüştürülür.
* **Eğitim Mimarisi:** Çıkarılan bu vektör dizileri TensorFlow/Keras ile oluşturulmuş Çok Katmanlı Algılayıcı (MLP/DNN) modeline verilir. Bu sayede model çok hafif (yaklaşık 900 KB) ve inanılmaz hızlı çalışır.
* **Güven Oranı:** Modelin son test ve doğrulama aşamasında elde ettiği Doğruluk Oranı (Validation Accuracy) **%86.7** olarak ölçülmüştür.

### 2. Modelin Eğitimi
* **`2_train_model.py`**: İşlenen landmark koordinatlarını alır ve TensorFlow / Keras kullanarak derin öğrenme (Deep Learning) modelini eğitir. Bu aşamada modelin başarı oranları ölçülür, grafikleri `output/` klasörüne çizilir ve `.keras` olarak ağırlıkları kaydedilir.

### 3. Mobil Uyumluluk (TFLite Çevirisi)
* **`3_convert_tflite.py`**: Eğitilen ve ağırlıkları kaydedilen büyük TensorFlow modelini, mobil cihazlarda düşük gecikme ve yüksek hızla çalışabilmesi için **TensorFlow Lite (.tflite)** formatına dönüştürür. Çıkan bu dosya mobil uygulamaya yüklenir.
* **`fix_model.py`**: Çeviri sonrası veya eğitim esnasında model shape'lerinde oluşan olası hataları (metadata vs.) düzeltmek için kullanılır.

### 4. Test
* **`4_test_webcam.py`**: Mobil uygulamaya geçmeden önce, eğitilen TFLite modelinin doğruluğunu bilgisayar kamerasından (Webcam) canlı olarak test etmek için kullanılır.
* **`server.py`**: Modelin olası bir uzak sunucu üzerinden test edilmesi veya farklı bir mimaride sunucu üzerinden serve edilmesi (inferencing) amacıyla oluşturulmuştur. (Mobil uygulamamız modeli lokal çalıştırmaktadır, bu script sadece test/geliştirme amaçlıdır).

## Kurulum
Python gereksinimlerini yüklemek için:
```bash
pip install -r requirements.txt
```

Bu kodların çalışabilmesi için Python ortamınızda `tensorflow`, `mediapipe`, `opencv-python` vb. paketlerin kurulu olması gerekmektedir. Eğitim verilerinizi (görsellerinizi) `data` klasörü içerisine kategorize edilmiş bir şekilde yerleştirmeniz beklenmektedir.

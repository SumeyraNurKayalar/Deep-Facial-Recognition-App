import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import uuid
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Model

# =================================================================
# 1. ÖZEL L1Dist KATMANI TANIMI (GÜNCELLENMİŞ ÇÖZÜM)
# =================================================================
class L1Dist(Layer):
    """L1 Mutlak Uzaklık Katmanı"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        # inputs: [input_embedding, validation_embedding]
        input_embedding, validation_embedding = inputs
        # Tensör farkı alma işlemi
        return tf.math.abs(input_embedding - validation_embedding)
    
    # KESİN ÇÖZÜM: compute_output_shape metodunu, Batch_Size'ı None olarak koruyacak şekilde güncelliyoruz.
    def compute_output_shape(self, input_shape):
        # input_shape bir tuple: (shape1, shape2). shape1 = (None, 256) olmalı.
        # Bu, çıkışın tam shape tuple'ını döndürür: (None, 256)
        return input_shape[0]
    
    def get_config(self):
        return super().get_config()

# GPU memory ayarları
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) 

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Application data paths
APPLICATION_DATA_DIR = 'application_data'
INPUT_IMAGE_DIR = os.path.join(APPLICATION_DATA_DIR, 'input_image')
VERIFICATION_IMAGES_DIR = os.path.join(APPLICATION_DATA_DIR, 'verification_images')

# Make the directories
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)
os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(VERIFICATION_IMAGES_DIR, exist_ok=True)

print("Sistem hazır!")

# Preprocess fonksiyonu
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# Model oluşturma fonksiyonları
def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')
    
    # Basit model
    c1 = Conv2D(32, (3, 3), activation='relu')(inp)
    m1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu')(m1)
    m2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu')(m2)
    m3 = MaxPooling2D((2, 2))(c3)
    
    f1 = Flatten()(m3)
    d1 = Dense(256, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# =================================================================
# 3. make_siamese_model FONKSİYONU 
# =================================================================
def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    
    embedding_model = make_embedding()
    
    inp_embedding = embedding_model(input_image)
    val_embedding = embedding_model(validation_image)
    
    # L1Dist KULLANILMALI
    distances = L1Dist(name='l1_distance')([inp_embedding, val_embedding]) 
    
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# Negative verileri hazırla
def prepare_negative_data():
    print("Negative veriler kontrol ediliyor...")
    if len(os.listdir(NEG_PATH)) == 0:
        print("Negative klasörü boş, örnek resimler oluşturuluyor...")
        for i in range(50):
            random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            imgname = os.path.join(NEG_PATH, f'negative_{i:03d}.jpg')
            cv2.imwrite(imgname, random_image)
        print("50 adet negative örnek oluşturuldu.")

# Verification images hazırla
def prepare_verification_data():
    if len(os.listdir(VERIFICATION_IMAGES_DIR)) == 0:
        print("Verification images klasörü dolduruluyor...")
        positive_files = os.listdir(POS_PATH)
        if len(positive_files) > 0:
            for i, file in enumerate(positive_files[:5]):
                src = os.path.join(POS_PATH, file)
                dst = os.path.join(VERIFICATION_IMAGES_DIR, file)
                import shutil
                shutil.copy2(src, dst)
            print(f"{min(5, len(positive_files))} adet verification image eklendi.")

# Model eğitimi
def train_model():
    print("\n=== MODEL EĞİTİMİ BAŞLIYOR ===")
    
    # Veri kontrolü
    anchor_files = [f for f in os.listdir(ANC_PATH) if f.endswith('.jpg')]
    positive_files = [f for f in os.listdir(POS_PATH) if f.endswith('.jpg')]
    negative_files = [f for f in os.listdir(NEG_PATH) if f.endswith('.jpg')]

    print(f"Anchor dosyaları: {len(anchor_files)}")
    print(f"Positive dosyaları: {len(positive_files)}")
    print(f"Negative dosyaları: {len(negative_files)}")

    min_files = min(len(anchor_files), len(positive_files), len(negative_files))
    if min_files < 3:
        print("HATA: Eğitim için en az 3'er adet resim gerekiyor!")
        return None

    take_count = min(20, min_files)
    print(f"Her kategoriden {take_count} resim kullanılacak")

    # BASİT DATASET OLUŞTURMA
    def create_simple_dataset():
        anchors = []
        comparisons = []
        labels = []
        
        # Random seçim
        selected_anchors = random.sample(anchor_files, take_count)
        selected_positives = random.sample(positive_files, take_count)
        selected_negatives = random.sample(negative_files, take_count)
        
        # Positive pairs
        for i in range(take_count):
            anchor_path = os.path.join(ANC_PATH, selected_anchors[i])
            positive_path = os.path.join(POS_PATH, selected_positives[i])
            
            anchor_img = preprocess(anchor_path).numpy()
            positive_img = preprocess(positive_path).numpy()
            
            anchors.append(anchor_img)
            comparisons.append(positive_img)
            labels.append(1.0)  # Positive pair
        
        # Negative pairs
        for i in range(take_count):
            anchor_path = os.path.join(ANC_PATH, selected_anchors[i])
            negative_path = os.path.join(NEG_PATH, selected_negatives[i])
            
            anchor_img = preprocess(anchor_path).numpy()
            negative_img = preprocess(negative_path).numpy()
            
            anchors.append(anchor_img)
            comparisons.append(negative_img)
            labels.append(0.0)  # Negative pair
        
        return (np.array(anchors), np.array(comparisons), np.array(labels))
    
    # Dataset oluştur
    anchors_array, comparisons_array, labels_array = create_simple_dataset()
    
    print(f"Dataset boyutu: {len(anchors_array)} örnek")
    
    # Modeli oluştur
    try:
        siamese_model = make_siamese_model()
        print("✓ Model başarıyla oluşturuldu!")
        
        # Modeli compile et
        siamese_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    except Exception as e:
        print(f"Model oluşturma hatası: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # BASİT EĞİTİM
    EPOCHS = 3
    
    print(f"\nEğitim başlıyor... ({EPOCHS} epoch)")
    
    try:
        history = siamese_model.fit(
            [anchors_array, comparisons_array],
            labels_array,
            epochs=EPOCHS,
            batch_size=8,
            validation_split=0.2,
            verbose=1,
            shuffle=True
        )
        
        print("✓ Eğitim başarıyla tamamlandı!")
        
        # Sonuçları göster
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        print(f"Final Loss: {final_loss:.4f}, Final Accuracy: {final_accuracy:.4f}")
        
    except Exception as e:
        print(f"Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Modeli kaydet
    try:
        siamese_model.save('siamese_model.h5')
        print("✓✓✓ Model 'siamese_model.h5' olarak KAYDEDİLDİ! ✓✓✓")
        
        if os.path.exists('siamese_model.h5'):
            file_size = os.path.getsize('siamese_model.h5')
            print(f"✓ Model dosya boyutu: {file_size} bytes")
            return siamese_model
            
    except Exception as e:
        print(f"Model kaydetme hatası: {e}")
        return None

# Verify fonksiyonu
def verify(model, detection_threshold=0.5, verification_threshold=0.5):
    results = []
    input_image_path = os.path.join(INPUT_IMAGE_DIR, 'input_image.jpg')
    
    if not os.path.exists(input_image_path):
        print("HATA: Input image bulunamadı!")
        return results, False
    
    verification_images = os.listdir(VERIFICATION_IMAGES_DIR)
    if len(verification_images) == 0:
        print("HATA: Verification images bulunamadı!")
        return results, False
    
    print(f"{len(verification_images)} adet verification image işleniyor...")
    
    for image in verification_images:
        try:
            input_img = preprocess(input_image_path)
            validation_img = preprocess(os.path.join(VERIFICATION_IMAGES_DIR, image))
            
            # Batch dimension ekle
            input_batch = tf.expand_dims(input_img, axis=0)
            validation_batch = tf.expand_dims(validation_img, axis=0)
            
            result = model.predict([input_batch, validation_batch], verbose=0)
            results.append(result[0][0])
        except Exception as e:
            continue

    if len(results) == 0:
        print("HATA: Hiç sonuç üretilemedi!")
        return results, False

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(verification_images)
    verified = verification > verification_threshold
    
    print(f"Detection: {detection}/{len(results)}, Verification score: {verification:.2f}")
    
    return results, verified

# Ana fonksiyon
def main():
    print("=== DEEP FACIAL RECOGNITION ===")
    print("Kamera açılıyor...")
    
    # Gerekli verileri hazırla
    prepare_negative_data()
    prepare_verification_data()
    
    # Model kontrolü
    model = None
    if os.path.exists('siamese_model.h5'):
        try:
            # Model yüklenirken L1Dist özel katmanı tanıtılmalı
            model = tf.keras.models.load_model(
                'siamese_model.h5', 
                custom_objects={'L1Dist': L1Dist}, 
                compile=False
            )
            print("✓ Önceden eğitilmiş model yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            model = None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
        
    print("\nKONTROLLER:")
    print("s: Modeli EĞİT ve KAYDET")
    print("v: Yüz DOĞRULAMA yap") 
    print("a: ANCHOR resmi çek")
    print("p: POSITIVE resmi çek")
    print("q: ÇIKIŞ")
    
    anchor_count = len([f for f in os.listdir(ANC_PATH) if f.endswith('.jpg')])
    positive_count = len([f for f in os.listdir(POS_PATH) if f.endswith('.jpg')])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Frame crop
        if frame.shape[0] > 250 and frame.shape[1] > 450:
            frame_cropped = frame[120:120+250, 200:200+250, :]
        else:
            frame_cropped = frame
            
        status = "Model: " + ("✓ HAZIR" if model is not None else "✗ EĞİTİLMEDİ")
        cv2.imshow(f'Yüz Tanıma - {status} | s:Eğit, v:Doğrula, a/p:Veri Topla, q:Çık', frame_cropped)

        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('s'):
            print("\n=== MODEL EĞİTİMİ BAŞLATILIYOR ===")
            trained_model = train_model()
            if trained_model is not None:
                model = trained_model
                print("✓✓✓ MODEL HAZIR! Artık 'v' tuşuyla doğrulama yapabilirsiniz. ✓✓✓")
            else:
                print("✗ Model eğitilemedi!")
        
        elif key == ord('v'):
            if model is None:
                print("HATA: Önce modeli eğitin! 's' tuşuna basın.")
                continue
                
            cv2.imwrite(os.path.join(INPUT_IMAGE_DIR, 'input_image.jpg'), frame_cropped)
            print("Input image kaydedildi, doğrulama yapılıyor...")
            try:
                results, verified = verify(model, 0.5, 0.5)
                status = "DOĞRULANDI ✅" if verified else "DOĞRULANMADI ❌"
                print(f"SONUÇ: {status}")
            except Exception as e:
                print(f"Doğrulama hatası: {e}")
        
        elif key == ord('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame_cropped)
            anchor_count += 1
            print(f"✓ Anchor kaydedildi: {anchor_count}")
        
        elif key == ord('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame_cropped)
            positive_count += 1
            print(f"✓ Positive kaydedildi: {positive_count}")
        
        elif key == ord('q'):
            print("Program sonlandırılıyor...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
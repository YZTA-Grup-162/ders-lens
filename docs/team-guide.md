# 👥 Takım Görev Dağılımı ve Rehber

Bu döküman takım üyelerinin görevlerini ve çalışma yönergelerini içerir.

## 🎯 Genel Proje Hedefi

Web tabanlı gerçek zamanlı AI sınıf etkileşim analizi aracı geliştirmek. Öğrenci dikkat seviyelerini webcam görüntülerinden analiz ederek öğretmenlere anlık geri bildirim sağlamak.

## 👨‍💼 Roller ve Sorumluluklar

### (Başak Dilara Çevik) - Repository & Docker Setup

#### ✅ Tamamlanan Görevler
- [x] Repository yapısı oluşturma
- [x] README.md ve temel dokümantasyon
- [x] .gitignore konfigürasyonu
- [x] Docker ve docker-compose yapısı
- [x] Proje iskelet yapısı

#### 📋 Devam Eden Görevler
- [ ] **Yüksek Öncelik**: Environment variables (.env dosyaları)
- [ ] **Yüksek Öncelik**: Database initialization scripts
- [ ] CI/CD pipeline setup (GitHub Actions)
- [ ] Production deployment stratejisi

#### 🛠️ Teknik Detaylar
```bash
# Yapılacaklar
1. Environment dosyaları oluştur:
   - backend/.env.example
   - backend/.env
   - frontend/.env.example

2. Database migration scripts:
   - backend/alembic/ klasörü
   - Migration dosyaları

3. GitHub Actions workflow:
   - .github/workflows/ci.yml
   - Test automation
   - Docker image building
```

#### 📚 Öğrenme Kaynakları
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [JWT Authentication Guide](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)



### 📹 Muhammed Enes Güler - OpenCV POC 

#### 📋 Görev Listesi
- [ ] **Yüksek Öncelik**: OpenCV face detection POC test
- [ ] **Yüksek Öncelik**: Webcam stream processing

### Enes Yıldırım - Backend API
#### 📋 Görev Listesi
- [ ] **Yüksek Öncelik**: FastAPI authentication system
- [ ] **Yüksek Öncelik**: User model ve database operations





#### Teknik Metrikler
- **Face Detection**: %95+ accuracy hedefle
- **Processing Speed**: Max 1 saniye per frame
- **Memory Usage**: Max 500MB RAM ? 

#### Test Senaryoları
```python
# Test edilecek durumlar:
1. Normal dikkat (yüz kamerada, düz bakış)
2. Dikkatsizlik (başı çevirme, telefon kullanma)
3. Kötü ışık koşulları
4. Birden fazla yüz
5. Gözlük, maske gibi aksesuarlar
```

#### 📚 Öğrenme Kaynakları
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=tr)
- [Head Pose Estimation](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)




## Success Metrics

### Technical Metrics
- **Code Coverage**: Min %80
- **API Response Time**: Max 200ms
- **Face Detection Accuracy**: Min %90
- **System Uptime**: %99+


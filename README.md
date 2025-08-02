

# DersLens

Web tabanlı gerçek zamanlı öğrenci dikkat analizi sistemi. Webcam görüntülerinden öğrenci dikkat seviyelerini tespit eder ve öğretmenlere anlık geri bildirim sağlar.
## Takım İsmi
Grup 162
## Takım Rolleri

**Product Owner:** Başak Dilara Çevik  
**Scrum Master:** Süleyman Kayyum Buberka  
**Developers:** Enes Yıldırım, Hümeyra Betül Şahin, Muhammed Enes Güler
## Ürün İsmi
DersLens



## Ürün Logosu
<img src="https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Logo.png?raw=true" alt="" width="200" />


## Ürün Açıklaması

DersLens, eğitimde verimliliği ve öğrenci başarısını artırmak amacıyla geliştirilmiş, mahremiyeti temel alan web tabanlı bir yapay zekâ platformudur. Sistemin en temel özelliği, video görüntülerinin asla sunucularımıza kaydedilmemesi; tüm analizlerin doğrudan kullanıcının kendi tarayıcısı (browser) üzerinde gerçekleşmesidir. Platform, bu güvenli yaklaşımla öğrencilerin dikkat, duygu ve davranışlarını anlık olarak analiz eder ve yalnızca bu analizin anonim sayısal çıktılarını öğretmenler için anlamlı geri bildirimlere dönüştürür.

## Ürün Özellikleri

- Webcam görüntülerinden gerçek zamanlı dikkat seviyesi tespiti
- Öğrenci ve öğretmen için ayrı web arayüzleri  
- OpenCV ve MediaPipe tabanlı yüz analizi
- RESTful API ve WebSocket desteği
- Docker ile kolay deployment


## Hedef Kitle

Lise ve üniversite çağındaki öğrencilerinin dikkatini ve derse odaklanmasını teknolojiyle artırarak onların akademik yeterliliklerini en üst düzeye çıkarmayı hedefleyen yenilikçi eğitim kurumlarıdır.

## Product Backlog
---
https://github.com/orgs/YZTA-Grup-162/projects

<details>
   
<summary><h2> Sprint 1 </h2></summary>
   
**Sprint içinde tamamlanması tahmin edilen puan:** 100 Puan  
**Puan tamamlama mantığı:** Proje boyunca tamamlanması gereken toplam 300 puanlık backlog bulunmaktadır. 3 sprinte bölündüğünde ilk sprintin 100 ile başlaması gerektiği kararlaştırıldı.  
**Backlog düzeni ve Story seçimleri:** Ürün backlog'umuz, kullanıcı deneyimini destekleyecek mekanik ve içeriklere öncelik verilerek yapılandırılmıştır. Sprint Board üzerinde görünen etiketler frontend, backend, görüntü işleme, model, devops (sunucu vb. işler için) ve R&D şeklindedir.
https://github.com/orgs/YZTA-Grup-162/projects/1
![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Sprint_1.png?raw=true)
Daily Scrum toplantıları, Google Meet üzerinden yapıldı. Örneklerin bazıları toplanti-notlari dosyasında Tarih.md belgelerinden ulaşabilirsiniz. 
https://github.com/YZTA-Grup-162/ders-lens/blob/main/toplanti-notlari/Daily_Scrum_Chat.png?raw=true  
   
https://github.com/YZTA-Grup-162/ders-lens/tree/main/toplanti-notlari

**Sprint Retrospective:**
- Görev dağılımları netleşti ve benimsendi, sorumluluklar kavrandı.
- Toplantıların iki günde bir yapılmasına karar verildi.
- Yapılan çalışmaların birkaç cümle ile özetlenip toplantı notlarına eklenmesine karar verildi.

**Sprint Review:**

- Ekip, tamamlanan proje kısımlarını ve bu konudaki geri bildirimlerini paylaştı. Projenin geleceğe yönelik nasıl geliştirilebileceği üzerine görüşmeler yapıldı.
- FastAPI kullanılarak temel API iskeleti kuruldu, veritabanı modeli tasarlandı ve uygulandı.
- Docker kullanılarak geliştirme ortamları senkronize edildi.
- Ekip içi görev dağılımları ve araştırma konuları belirlendi.
- Sprint Review katılımcıları: Başak Dilara Çevik,Süleyman Kayyum Buberka, Enes Yıldırım, Hümeyra Betül Şahin, Muhammed Enes Güler
</details>
<details>
  <summary><h2> Sprint 2 </h2></summary>  
   
  **Sprint içinde tamamlanması tahmin edilen puan:** 100 Puan  
  **Puan tamamlama mantığı:** Proje boyunca tamamlanması gereken toplam 300 puanlık backlog bulunmaktadır. 3 sprinte bölündüğünde ikinci sprintin 100 ile başlaması gerektiği kararlaştırıldı.  
  **Backlog düzeni ve Story seçimleri:** Ürün backlog'umuz, kullanıcı deneyimini destekleyecek mekanik ve içeriklere öncelik verilerek yapılandırılmıştır. Sprint Board üzerinde görünen etiketler frontend, backend, görüntü işleme, model, devops (sunucu vb. işler için) ve R&D şeklindedir.  
  https://github.com/orgs/YZTA-Grup-162/projects/6  

Daily Scrum toplantıları, Google Meet ve Whatsapp üzerinden devam etmiştir. 
  <details>
  <summary><h3> Daly Scrum: Ekran Görüntüleri </h3></summary>  
     
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Toplanti_2_1.png?raw=true)
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Toplanti_2_2.png?raw=true)
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Toplanti_2_3.png?raw=true)
</details>

**Sprint Board**

![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Sprint_2.png?raw=true) 

  <details>
  <summary><h3> Uygulama Durumu: Ekran Görüntüleri </h3></summary>  
     
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Uygalama_Ekran_Goruntusu_1.png?raw=true)
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Uygulama_Ekran_Goruntusu_2.png?raw=true)  
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Uygulama_Ekran_Goruntusu_3.png?raw=true)  
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Uygulama_Ekran_Goruntusu_4.png?raw=true)    
</details>

**Sprint Review:**
- Ekip, görevlerdeki gelişmeleri açıklayınca mevcut performansın yükseltilmesi gerektiği anlaşıldı. Bu doğrultuda, modelin geliştirilmesi ve hataların giderilmesi üzerine görüşmeler yapıldı.
- Sprint Review katılımcıları: Başak Dilara Çevik,Süleyman Kayyum Buberka, Enes Yıldırım, Hümeyra Betül Şahin, Muhammed Enes Güler

**Sprint Retrospective:**
- Ekip belirlenen veri setleri ile modelin geliştirilmesine devam edecek.
-  Model geliştirme ve hata giderme süreçlerimize "hızlı düzeltme" (hotfix) veya "öncelikli hata çözümü" mekanizmaları dahil edilecek.
</details>
<details>
  <summary><h2> Sprint 3 </h2></summary>  
   
  **Sprint içinde tamamlanması tahmin edilen puan:** 100 Puan  
  **Puan tamamlama mantığı:** Proje boyunca tamamlanması gereken toplam 300 puanlık backlog bulunmaktadır. 3 sprinte bölündüğünde üçüncü sprint 100 ile başlatıldı.  
  **Backlog düzeni ve Story seçimleri:** Ürün backlog'umuz, kullanıcı deneyimini destekleyecek mekanik ve içeriklere öncelik verilerek yapılandırılmıştır. Sprint Board üzerinde görünen etiketler frontend, backend, görüntü işleme, model, devops (sunucu vb. işler için) ve R&D şeklindedir.  
  https://github.com/orgs/YZTA-Grup-162/projects/8  
  
Daily Scrum toplantıları, Google Meet ve Whatsapp üzerinden devam etmiştir. 
  <details>
  <summary><h3> Daly Scrum: Ekran Görüntüleri </h3></summary>  
     
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Toplanti_3_1.png)
   ![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Toplanti_3_2.png)
   ![]()
</details>

**Sprint Board**

![](https://github.com/YZTA-Grup-162/ders-lens/blob/main/assets/Sprint_3.png) 

  <details>
  <summary><h3> Uygulama Durumu: Ekran Görüntüleri </h3></summary>  
     
   ![]()
   ![]()  
   ![]()  
   ![]()    
</details>

**Sprint Review:**
- Ekip üyeleri birbirini tebrik etti.
- Birden fazla veri seti incelendikten sonra, model fine-tuning yapılarak eğitildi.
- Sprint Review katılımcıları: Başak Dilara Çevik,Süleyman Kayyum Buberka, Enes Yıldırım, Hümeyra Betül Şahin, Muhammed Enes Güler

**Sprint Retrospective:**
- Spirntlerde bahsedilen özellikler eklendi ve proje başarıyla tamamlandı.
- Ekip sprint sonunu kutladı.
</details>  

## Model Dosyaları

**⚠️ Önemli:** Büyük AI model dosyaları (*.pth, *.onnx, *.pkl) boyut kısıtlamaları nedeniyle Git'ten hariç tutulmuştur.

### Gerekli Model Dosyaları
Uygulamayı çalıştırmak için aşağıdaki model dosyalarını indirmeniz/eğitmeniz gerekir:

**AI Servis Modelleri** (`ai-service/` klasörüne yerleştirin):
- `mendeley_model.pth` - Ana duygu tespit modeli
- `mendeley_nn_best.pth` - En iyi sinir ağı modeli
- `mendeley_scaler.pkl` - Veri ön işleme ölçekleyici
- `mendeley_random_forest.pkl` - Rastgele orman sınıflandırıcı
- `mendeley_gradient_boosting.pkl` - Gradyan artırma modeli
- `mendeley_logistic_regression.pkl` - Lojistik regresyon modeli

**Backend Modelleri** (`backend/models/` klasörüne yerleştirin):
- `daisee_emotional_model_best.pth` - DAISEE veri seti eğitimli model
- `fine_tuned_randomforest_deep.pkl` - İnce ayarlı rastgele orman
- `best_model.onnx` - ONNX optimize edilmiş model
- Çeşitli ensemble ve ölçekleyici dosyaları

**FER2013 Modelleri** (`models_fer2013/` klasörüne yerleştirin):
- `fer2013_model.pth` - FER2013 veri seti modeli
- `fer2013_model.onnx` - ONNX versiyonu
- Model metrikleri ve ölçekleyici dosyaları

### Model Eğitimi
Modelleri sıfırdan eğitmek için:
```bash
python train.py
python run_all_dataset_training.py
```

## Tech Stack

**Backend:** FastAPI, Python 3.9+, OpenCV, MediaPipe  
**Frontend:** React 18, TypeScript, Tailwind CSS  
**Database:** SQLite (development), PostgreSQL (production)  
**Deployment:** Docker, Docker Compose

## Proje Yapısı

```
attention-pulse/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── models/       # AI models
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Core logic
│   │   └── utils/        # Utility functions
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Pages
│   │   └── services/     # API services
│   ├── package.json
│   └── Dockerfile
├── docs/                 # Documentation
├── scripts/              # Deployment scripts
└── docker-compose.yml    # Multi-container setup
```


## API Endpoints

### System
- `GET /` - Sistem durumu
- `GET /health` - Sistem sağlık kontrolü
- `GET /docs` - API dokümantasyonu

### Authentication
- `POST /api/auth/login` - Kullanıcı girişi
- `POST /api/auth/register` - Kullanıcı kaydı
- `GET /api/auth/me` - Mevcut kullanıcı bilgisi

### Student
- `POST /api/student/session/start` - Öğrenme oturumu başlatma
- `POST /api/student/session/{session_id}/end` - Oturum sonlandırma
- `POST /api/student/video/frame` - Video karesi gönderme
- `GET /api/student/attention/score` - Dikkat skoru alma
- `GET /api/student/attention/history` - Dikkat geçmişi
- `GET /api/student/feedback` - Geri bildirim alma
- `GET /api/student/sessions` - Kullanıcı oturumları

### Teacher
- `POST /api/teacher/session/start` - Öğretmen oturumu başlatma
- `POST /api/teacher/session/{session_id}/end` - Öğretmen oturumu sonlandırma
- `GET /api/teacher/students` - Öğrenci listesi
- `GET /api/teacher/overview` - Sınıf genel durumu
- `GET /api/teacher/session/{id}/analytics` - Oturum analizi
- `GET /api/teacher/alerts` - Gerçek zamanlı uyarılar
- `POST /api/teacher/alerts/{alert_id}/acknowledge` - Uyarı onaylama
- `GET /api/teacher/analytics/summary` - Analiz özeti
- `GET /api/teacher/students/{student_id}/profile` - Öğrenci profili

### WebSocket
- `/ws/system` - Sistem güncellemeleri
- `/ws/student/{session_id}` - Öğrenci gerçek zamanlı veri
- `/ws/teacher/{class_id}` - Öğretmen izleme



## Hızlı Başlangıç

### Kurulum
```powershell
git clone https://github.com/YZTA-Grup-162/attention-pulse.git
cd attention-pulse
docker-compose up --build
```

**Kurulum Tamamlandıktan Sonra:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000  
- API Docs: http://localhost:8000/docs

### ✅ Kurulum Doğrulama
```powershell
powershell -ExecutionPolicy Bypass -File quick_test.ps1
```


### Önkoşullar (Windows)

#### Docker ile çalıştırmak için:
1. **Docker Desktop** indirin ve kurun: [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Kurulum sonrası bilgisayarı yeniden başlatın
3. Docker Desktop'ı açın ve çalıştığından emin olun

#### Manuel kurulum için:
1. **Python 3.9+**: [Python Downloads](https://www.python.org/downloads/)
2. **Node.js 18+**: [Node.js Downloads](https://nodejs.org/)
3. **Git**: [Git for Windows](https://git-scm.com/download/win)

### Docker ile Tek Adımda Kurulum (Önerilen)
```powershell
# Windows PowerShell için:
git clone https://github.com/YZTA-Grup-162/attention-pulse.git
cd attention-pulse
docker-compose up --build

# Kurulum testi (opsiyonel):
powershell -ExecutionPolicy Bypass -File quick_test.ps1
```

### Manuel kurulum (Docker yoksa)
```powershell
# Repository'yi clone edin
git clone https://github.com/YZTA-Grup-162/attention-pulse.git
cd attention-pulse

# Backend kurulumu
py -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
pip install email-validator pydantic[email]

# Backend başlatma (yeni terminal açın)
cd d:\attention-pulse
venv\Scripts\activate
python backend/run_server.py

# Frontend kurulumu (başka bir terminal açın)
cd frontend
# NPM çalışıyorsa:
npm install
npm start

# NPM çalışmıyorsa Yarn kullanın:
yarn install
yarn start
```

**Erişim:** 
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Sorun Giderme (Windows)

#### Docker Desktop bağlantı hatası:
```powershell
# Hata: "unable to get image" veya "dockerDesktopLinuxEngine" bulunamıyor
# 1. Docker Desktop'ın çalıştığından emin olun
docker --version

# 2. Docker Desktop açık değilse, Start menüsünden başlatın
# 3. Docker Desktop'ta "Engine running" yazısını bekleyin

# 4. Docker servisinin çalıştığını kontrol edin:
docker ps

# 5. Çalışmıyorsa Docker Desktop'ı yeniden başlatın
# Start Menu > Docker Desktop > Restart
```

#### Docker Compose version uyarısı:
```yaml
# docker-compose.yml dosyasından version satırını kaldırın:
# version: '3.8'  # Bu satırı silin veya yorum satırı yapın
```

#### Python bulunamıyor hatası:
```powershell
# Python'un PATH'e eklendiğinden emin olun
python --version
# Çalışmıyorsa, Python'u Microsoft Store'dan kurun
```

#### NPM bulunamıyor hatası:
```powershell
# Node.js kurulu olduğundan emin olun
node --version

# NPM hata veriyorsa alternatif paket yöneticisi kullanın:
# 1. Yarn kurun (önerilen):
npm install -g yarn


# 2. Yarn ile frontend kurun:
cd frontend
yarn install
yarn start

# 3. Veya NPM'i yeniden kurun:
# Node.js'i tamamen kaldırıp tekrar kurun: https://nodejs.org/
```

#### Virtual environment aktivasyon sorunu:
```powershell
# PowerShell execution policy sorunu varsa:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sonra tekrar deneyin:
venv\Scripts\activate
```

#### Docker build hatası (frontend):
```powershell
# Frontend'de package-lock.json eksikse:
cd frontend

# Önce src klasörünü kontrol edin, yoksa oluşturun
# Eğer frontend kaynak kodu eksikse:
git pull origin main  # Son sürümü çekin

# Veya manuel olarak:
npm install  # package-lock.json oluşturur
npm run build  # Build test edin

# Docker build için:
# Dockerfile'da "npm ci" yerine "npm install" kullanın
```
```powershell
# Port 8000 kullanımda ise farklı port kullanın:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Port 3000 kullanımda ise:
# package.json'da port değiştirin veya:
set PORT=3001 && npm start
```

#### Docker Desktop bağlantı hatası:
```
unable to get image: error during connect: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**Çözüm adımları:**
1. Docker Desktop'ın tamamen çalıştığından emin olun (sistem tepsisinde whale ikonu)
2. Docker Desktop'ı yeniden başlatın: Sağ tık → "Restart Docker Desktop"
3. Windows'u yeniden başlatın
4. Docker servisi kontrolü:
```powershell
# Docker Desktop çalışıyor mu kontrol et:
docker version

# Çalışmıyorsa Manuel başlatma:
"C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

#### Docker Compose version uyarısı:
```
the attribute `version` is obsolete, it will be ignored
```

**docker-compose.yml** dosyasının başındaki `version:` satırını kaldırın:
```yaml
# Bu satırı silin:
# version: '3.8'

# Doğrudan services ile başlayın:
services:
  backend:
    ...
```

#### Docker tamamen çalışmıyorsa alternatif:
Manuel kurulum yöntemini kullanın (yukarıdaki "Manuel kurulum" bölümüne bakın)

## Development

### OpenCV test (Backend çalıştıktan sonra)
```powershell
cd backend
venv\Scripts\activate
python app/utils/opencv_utils.py
```

### Code quality
```powershell
# Backend
cd backend
venv\Scripts\activate
black app/
flake8 app/

# Frontend
cd frontend
npm run lint
npm run format
```

### Hızlı başlatma script'i (Windows)
Proje klasöründe `start.bat` oluşturun:
```batch
@echo off
echo AttentionPulse başlatılıyor...

echo Virtual environment aktive ediliyor...
call venv\Scripts\activate

echo Backend başlatılıyor...
start cmd /k "cd /d %~dp0 && venv\Scripts\activate && python backend\run_server.py"

timeout /t 5

echo Frontend başlatılıyor...
start cmd /k "cd /d frontend && yarn install && yarn start"

echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000  
echo API Docs: http://localhost:8000/docs
pause
```

---

*Çevrimiçi eğitim için computer vision tabanlı dikkat takip sistemi.*

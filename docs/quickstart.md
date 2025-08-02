# Hızlı Başlangıç Kılavuzu

Bu döküman DersLens projesini için geliştirme ortamının hızlıca çalıştırılması için hazırlanmıştır.

## Gereksinimler

### Docker ile Çalıştırma (Önerilen)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) yüklü olmalı
- Docker Compose (Docker Desktop ile birlikte gelir)

### Local Development
- **Python 3.9+** ([Python.org](https://www.python.org/downloads/))
- **Node.js 18+** ([Node.js](https://nodejs.org/))
- **Git** ([Git](https://git-scm.com/))

## Docker ile Çalıştırma

### 1. Repository'yi Clone Edin
```bash
git clone https://github.com/YZTA-Grup-162/attention-pulse.git
cd attention-pulse
```

### 2. Environment Dosyası Oluşturun
```bash
# Backend için .env dosyası oluşturun
cp backend/.env.example backend/.env
```

### 3. Servisleri Başlatın
```bash
# Tüm servisleri başlat (production mode)
docker-compose up --build

# Veya development mode için 
docker-compose -f docker-compose.dev.yml up --build
```

### 4. Erişim
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 💻 Local Development

### Backend Setup
```bash
# Backend klasörüne gidin
cd backend

# Virtual environment oluşturun
python -m venv venv

# Virtual environment'ı aktive edin
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Dependencies yükleyin
pip install -r requirements.txt

# Veritabanı tablolarını oluşturun
python -c "from app.core.database import engine, Base; Base.metadata.create_all(bind=engine)"

# API'yi başlatın
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
# Frontend klasörüne gidin
cd frontend

# Dependencies yükleyin
npm install

# Development server'ı başlatın
npm start
```



## API Kullanımı

### Authentication
```bash
# Kullanıcı kaydı
curl -X POST "http://localhost:8000/api/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","username":"testuser","full_name":"Test User","password":"password123","role":"student"}'

# Giriş yapma
curl -X POST "http://localhost:8000/api/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=testuser&password=password123"
```

### Student Endpoints
```bash
# Session başlatma
curl -X POST "http://localhost:8000/api/student/session/start" \
     -H "Authorization: Bearer SECRET_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"session_name":"Matematik Dersi"}'

# Video frame gönderme
curl -X POST "http://localhost:8000/api/student/video/frame" \
     -H "Authorization: Bearer SECRET_TOKEN" \
     -F "file=@image.jpg" \
     -F "session_id=1"
```

## Geliştirme Araçları

### Backend Code Quality
```bash
cd backend

# Code formatting
black app/

# Import sorting
isort app/

# Linting
flake8 app/

# Tests
pytest tests/
```

### Frontend Code Quality
```bash
cd frontend

# Linting
npm run lint

# Format code
npm run format

# Tests
npm test
```

## Sorun Giderme

### Webcam Erişim Sorunu
- Docker kullanırken, `--device=/dev/video0` parametresi gerekebilir

### Port Çakışması
```bash
# Kullanılan portları kontrol edin
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# Çakışan servisleri durdurun veya farklı port kullanın
```

### Database Sorunu
```bash
# SQLite database'i sıfırlama
rm backend/attention_pulse.db

# Tabloları yeniden oluştur
python -c "from app.core.database import engine, Base; Base.metadata.create_all(bind=engine)"
```

## Yardım

Sorunlarla karşılaştığınızda:

1. **Logları kontrol edin**:
   ```bash
   # Docker logs
   docker-compose logs backend
   docker-compose logs frontend
   ```


---
**Not**: Bu proje hızlı geliştirme amaçlı hazırlanmıştır. Production ortamında ek güvenlik ve performans optimizasyonları gerekebilir.

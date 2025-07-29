# 🎓 Ders Lens - AI Destekli Öğrenci Analizi Platformu

Modern, futuristik bir React uygulaması ile yapay zeka destekli öğrenci dikkat, katılım, duygu ve bakış analizi.

![Ders Lens](./derslens-logo.png)

## ✨ Özellikler

### 🧠 AI Destekli Analiz
- **Dikkat Takibi**: Yüz yönü ve varlık tespiti ile gerçek zamanlı dikkat analizi
- **Katılım Analizi**: Hareket, ekran etkileşimi ve duruş analizi
- **Duygu Tanıma**: Mikro ifadeler ile 7 farklı duygu sınıflandırması
- **Bakış Haritalama**: Göz izleme ve ekran odak tespiti

### 🎨 Modern UI/UX
- **Futuristik Tasarım**: Glassmorphism ve gradient efektleri
- **Karanlık/Açık Mod**: Kullanıcı tercihi ile tema değiştirme
- **Responsive Design**: Tüm ekran boyutlarında mükemmel görünüm
- **Smooth Animasyonlar**: Framer Motion ile profesyonel geçişler

### 🔬 Kullanılan AI Modelleri
- **FER2013**: Duygu tanıma modeli
- **DAiSEE**: Dikkat ve katılım analizi
- **Mendeley**: Öğrenci davranış analizi
- **ONNX Runtime**: Hızlı model çıkarımı

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Node.js 18+
- npm veya yarn
- Modern web tarayıcısı (Chrome, Firefox, Safari, Edge)
- Kamera erişimi (demo için)

### Kurulum

1. **Repository'yi klonlayın**
```bash
git clone <repository-url>
cd ders-lens
```

2. **Frontend bağımlılıklarını yükleyin**
```bash
cd frontend
npm install
```

3. **Geliştirme sunucusunu başlatın**
```bash
npm run dev
```

4. **Tarayıcıda açın**
```
http://localhost:3000
```

### Production Build

```bash
npm run build
npm run preview
```

## 📱 Kullanım

### Ana Sayfa
- Modern, futuristik tasarım ile Ders Lens'in yeteneklerini keşfedin
- Özellik kartları ile AI destekli analiz özelliklerini görün
- Gerçek zamanlı demo görselini inceleyin

### Demo Sayfası
1. **"Demo'ya Başla"** butonuna tıklayın
2. Kamera erişimine izin verin
3. **"Başlat"** butonuna tıklayarak analizi başlatın
4. Gerçek zamanlı sonuçları görün:
   - Dikkat seviyesi (%)
   - Katılım oranı (%)
   - Duygu durumu
   - Bakış yönü

## 🏗️ Teknik Detaylar

### Frontend Stack
- **React 18**: Modern React özellikleri
- **TypeScript**: Tip güvenliği
- **Vite**: Hızlı geliştirme ve build
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animasyon kütüphanesi
- **Zustand**: State yönetimi
- **React Router**: Sayfa yönlendirme

### API Entegrasyonu
- **WebRTC**: Kamera erişimi
- **Canvas API**: Frame yakalama
- **Fetch API**: Backend iletişimi
- **WebSocket**: Gerçek zamanlı veri (gelecek)

### Responsive Breakpoints
- **Mobile**: 320px - 768px
- **Tablet**: 768px - 1024px
- **Desktop**: 1024px+

## 🎯 Backend Entegrasyonu

Frontend, aşağıdaki endpoint'leri kullanarak backend ile iletişim kurar:

```typescript
POST /api/analyze-frame
- Frame analizi için kamera görüntüsü gönderir
- Dikkat, katılım, duygu ve bakış verilerini alır

POST /api/calibrate-gaze
- Bakış kalibrasyonu için kalibrasyon noktalarını gönderir

GET /api/health
- Backend sağlık durumunu kontrol eder

GET /api/models/info
- Kullanılan AI modellerinin bilgilerini alır
```

## 🔧 Geliştirme

### Proje Yapısı
```
frontend/
├── src/
│   ├── components/          # Yeniden kullanılabilir bileşenler
│   │   ├── LiveCamera.tsx   # Canlı kamera bileşeni
│   │   ├── AnalysisDashboard.tsx # Analiz sonuçları
│   │   └── FuturisticBackground.tsx # Animasyonlu arkaplan
│   ├── pages/               # Sayfa bileşenleri
│   │   ├── HomePage.tsx     # Ana sayfa
│   │   └── DemoPage.tsx     # Demo sayfası
│   ├── services/            # API servisleri
│   │   └── apiService.ts    # Backend iletişimi
│   ├── contexts/            # React context'leri
│   │   └── ThemeContext.tsx # Tema yönetimi
│   ├── store/               # State yönetimi
│   │   └── dersLensStore.ts # Zustand store
│   └── styles/              # CSS dosyaları
├── public/                  # Statik dosyalar
└── package.json
```

### Stil Kılavuzu
- **Renk Paleti**: Mavi, mor ve pembe tonlarında gradientler
- **Tipografi**: Modern, sans-serif fontlar
- **Spacing**: Tailwind CSS spacing sistemi
- **Animasyonlar**: Smooth, performanslı geçişler

### Component Geliştirme
```typescript
// Örnek component yapısı
interface ComponentProps {
  data: AnalysisResult;
  onUpdate?: (data: AnalysisResult) => void;
}

const Component = ({ data, onUpdate }: ComponentProps) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass rounded-xl p-6"
    >
      {/* Component içeriği */}
    </motion.div>
  );
};
```

## 🚀 Deployment

### Docker ile Deploy
```bash
# Frontend için Dockerfile mevcuttur
docker build -t ders-lens-frontend .
docker run -p 3000:3000 ders-lens-frontend
```

### Vercel/Netlify Deploy
1. Repository'yi bağlayın
2. Build command: `npm run build`
3. Output directory: `dist`
4. Deploy edin

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- **FER2013 Dataset**: Duygu tanıma modeli için
- **DAiSEE Dataset**: Dikkat ve katılım analizi için
- **Mendeley Dataset**: Öğrenci davranış analizi için
- **MediaPipe**: Yüz ve göz tespiti için
- **React Ecosystem**: Modern web geliştirme için

---

**Ders Lens** - AI ile Eğitimin Geleceği 🚀

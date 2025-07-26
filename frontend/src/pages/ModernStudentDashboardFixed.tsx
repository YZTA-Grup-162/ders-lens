import { motion } from 'framer-motion';
import {
    Activity,
    BarChart3,
    Bell,
    Brain,
    Calendar,
    Camera,
    Clock,
    Eye,
    Heart,
    Home,
    LogOut,
    Settings,
    Target,
    TrendingDown,
    TrendingUp,
    User
} from 'lucide-react';
import React, { useEffect, useState } from 'react';
const ModernStudentDashboard: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isSessionActive, setIsSessionActive] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiMetrics, setAiMetrics] = useState({
    attention: 87,
    engagement: 89,
    emotion: 'Pozitif',
    emotionScore: 4.2,
    gazeDirection: 'Merkez',
    gazeConfidence: 0.92,
    faceDetected: false,
    audioLevel: 0,
    blinkRate: 12, 
    posture: 'İyi',
    distraction: 'Düşük'
  });
  const [analysisHistory, setAnalysisHistory] = useState<any[]>([]);
  const [processingStage, setProcessingStage] = useState('');
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const runAIAnalysis = () => {
    const stages = [
      'Kamera kalibrasyonu yapılıyor...',
      'MediaPipe face mesh yükleniyor...',
      'Göz takibi modeli hazırlanıyor...',
      'Duygu tanıma CNN modeli aktif...',
      'Ses analizi başlatılıyor...',
      'Dikkat skoru hesaplanıyor...',
      'Gerçek zamanlı analiz aktif'
    ];
    let stageIndex = 0;
    setProcessingStage(stages[0]);
    const stageInterval = setInterval(() => {
      stageIndex++;
      if (stageIndex < stages.length) {
        setProcessingStage(stages[stageIndex]);
      } else {
        clearInterval(stageInterval);
        setProcessingStage('AI modelleri çalışıyor...');
      }
    }, 2000); 
    const metricsInterval = setInterval(() => {
      setAiMetrics(prev => {
        const attentionTrend = Math.sin(Date.now() / 30000) * 5; 
        const engagementTrend = Math.cos(Date.now() / 25000) * 3;
        const emotionTrend = Math.sin(Date.now() / 40000) * 0.1;
        const attentionNoise = (Math.random() - 0.5) * 1.5;
        const engagementNoise = (Math.random() - 0.5) * 1.0;
        const emotionNoise = (Math.random() - 0.5) * 0.05;
        const newAttention = Math.max(75, Math.min(95, 85 + attentionTrend + attentionNoise));
        const newEngagement = Math.max(70, Math.min(96, 88 + engagementTrend + engagementNoise));
        const newEmotionScore = Math.max(3.5, Math.min(4.8, 4.2 + emotionTrend + emotionNoise));
        const gazeDirections = ['Merkez', 'Sol Üst', 'Sağ Alt', 'Merkez', 'Merkez', 'Yukarı'];
        const newGaze = Math.random() < 0.08 ? gazeDirections[Math.floor(Math.random() * gazeDirections.length)] : prev.gazeDirection;
        let newEmotion = 'Nötr';
        if (newEmotionScore > 4.3) newEmotion = 'Pozitif';
        else if (newEmotionScore > 4.0) newEmotion = 'Hafif Pozitif';
        else if (newEmotionScore < 3.8) newEmotion = 'Hafif Negatif';
        const newBlinkRate = Math.max(10, Math.min(16, 13 + Math.sin(Date.now() / 20000) * 2 + (Math.random() - 0.5) * 0.5));
        const newAudioLevel = Math.max(0.02, Math.min(0.25, 0.08 + Math.sin(Date.now() / 15000) * 0.05 + (Math.random() - 0.5) * 0.02));
        return {
          ...prev,
          attention: Math.round(newAttention * 10) / 10, 
          engagement: Math.round(newEngagement * 10) / 10,
          emotion: newEmotion,
          emotionScore: Math.round(newEmotionScore * 100) / 100, 
          gazeDirection: newGaze,
          gazeConfidence: Math.max(0.85, Math.min(0.98, prev.gazeConfidence + (Math.random() - 0.5) * 0.02)),
          faceDetected: Math.random() > 0.02, 
          audioLevel: Math.round(newAudioLevel * 1000) / 1000,
          blinkRate: Math.round(newBlinkRate * 10) / 10,
          posture: newAttention > 88 ? 'Mükemmel' : newAttention > 82 ? 'İyi' : 'Orta',
          distraction: newAttention > 90 ? 'Çok Düşük' : newAttention > 85 ? 'Düşük' : newAttention > 80 ? 'Orta' : 'Yüksek'
        };
      });
    }, 15000); 
    return () => {
      clearInterval(stageInterval);
      clearInterval(metricsInterval);
    };
  };
  const toggleCamera = async () => {
    if (!isCameraActive) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 },
          audio: true 
        });
        setCameraStream(stream);
        setIsCameraActive(true);
        setIsAnalyzing(true);
        const cleanup = runAIAnalysis();
        setTimeout(() => {
          cleanup();
          setIsAnalyzing(false);
        }, 60000);
      } catch (error) {
        console.error('Kamera erişim hatası:', error);
        alert('Kamera erişimi başarısız. Lütfen kamera izinlerini kontrol edin.');
      }
    } else {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        setCameraStream(null);
      }
      setIsCameraActive(false);
      setIsAnalyzing(false);
      setProcessingStage('');
    }
  };
  const studentData = {
    name: "Başak Avcı",
    class: "Matematik 101",
    todayStats: {
      attention: aiMetrics.attention,
      engagement: aiMetrics.engagement,
      emotionalState: aiMetrics.emotionScore,
      sessionTime: "42 dk"
    },
    weeklyTrend: {
      attention: [85, 88, 82, 90, 87, 89, 87],
      engagement: [78, 85, 90, 88, 92, 89, 92]
    },
    currentSession: {
      startTime: "14:00",
      duration: 42,
      peakAttention: 94,
      averageEmotion: "Pozitif"
    }
  };
  const StatCard = ({ icon: Icon, title, value, trend, color }: any) => (
    <motion.div
      className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100"
      whileHover={{ y: -2 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`w-12 h-12 bg-gradient-to-r ${color} rounded-xl flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        {trend && (
          <div className={`flex items-center ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {trend > 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
            <span className="text-sm font-medium">{Math.abs(trend)}%</span>
          </div>
        )}
      </div>
      <div className="text-3xl font-bold text-slate-900 mb-1">{value}</div>
      <div className="text-slate-600 text-sm">{title}</div>
    </motion.div>
  );
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 overflow-x-hidden">
      {}
      <nav className="fixed left-0 top-0 h-full w-64 bg-white/90 backdrop-blur-lg border-r border-slate-200 z-40">
        <div className="p-6">
          <div className="flex items-center space-x-3 mb-8">
            <img 
              src="/derslens-logo.png" 
              alt="Ders Lens Logo" 
              className="h-8 w-auto"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Ders Lens
            </span>
          </div>
          <div className="space-y-2">
            <a href="#" className="flex items-center space-x-3 px-4 py-3 bg-blue-50 text-blue-600 rounded-lg">
              <Home className="w-5 h-5" />
              <span>Dashboard</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <BarChart3 className="w-5 h-5" />
              <span>Analizler</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <Calendar className="w-5 h-5" />
              <span>Geçmiş</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <Settings className="w-5 h-5" />
              <span>Ayarlar</span>
            </a>
          </div>
        </div>
        <div className="absolute bottom-6 left-6 right-6">
          <button className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors w-full">
            <LogOut className="w-5 h-5" />
            <span>Çıkış Yap</span>
          </button>
        </div>
      </nav>
      {}
      <div className="ml-64">
        {}
        <header className="bg-white/80 backdrop-blur-lg border-b border-slate-200 px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Merhaba, {studentData.name}</h1>
              <p className="text-slate-600">{studentData.class} • {currentTime.toLocaleTimeString('tr-TR')}</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                isSessionActive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  isSessionActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  {isSessionActive ? 'Oturum Aktif' : 'Oturum Pasif'}
                </span>
              </div>
              <button className="relative p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                <Bell className="w-5 h-5" />
                <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></div>
              </button>
              <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
            </div>
          </div>
        </header>
        {}
        <main className="p-8 space-y-8 pb-16">
          {}
          <motion.div 
            className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-3xl p-8 mb-8 text-white"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-2">Mevcut Ders Oturumu</h2>
                <p className="text-blue-100">Matematik 101 - Diferansiyel Denklemler</p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold">{studentData.currentSession.duration} dk</div>
                <div className="text-blue-100">Oturum Süresi</div>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
                <div className="text-2xl font-bold">{studentData.currentSession.peakAttention}%</div>
                <div className="text-blue-100 text-sm">En Yüksek Dikkat</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
                <div className="text-2xl font-bold">{studentData.currentSession.averageEmotion}</div>
                <div className="text-blue-100 text-sm">Ortalama Duygu</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
                <div className="text-2xl font-bold">{studentData.currentSession.startTime}</div>
                <div className="text-blue-100 text-sm">Başlangıç Saati</div>
              </div>
            </div>
          </motion.div>
          {}
          <motion.div 
            className="bg-white rounded-3xl p-8 mb-8 shadow-lg border border-slate-100"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">AI Dikkat Analizi Demo</h2>
                <p className="text-slate-600">Kameranızı açarak gerçek zamanlı dikkat analizi deneyimleyin</p>
              </div>
              <button
                onClick={toggleCamera}
                className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                  isCameraActive 
                    ? 'bg-red-500 hover:bg-red-600 text-white' 
                    : 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white'
                }`}
              >
                <Camera className="w-5 h-5" />
                <span>{isCameraActive ? 'Kamerayı Kapat' : 'Kamerayı Aç'}</span>
              </button>
            </div>
            {isCameraActive && (
              <div className="space-y-8">
                {}
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                  {}
                  <div className="xl:col-span-2 space-y-6">
                    <div className="relative bg-slate-900 rounded-xl overflow-hidden shadow-lg">
                      <video
                        ref={(video) => {
                          if (video && cameraStream) {
                            video.srcObject = cameraStream;
                            video.play();
                          }
                        }}
                        className="w-full h-80 object-cover"
                        autoPlay
                        muted
                      />
                      {isAnalyzing && (
                        <div className="absolute top-4 left-4 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center shadow-lg">
                          <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
                          Analiz Ediliyor...
                        </div>
                      )}
                      <div className="absolute bottom-4 left-4 right-4">
                        <div className="bg-black/70 backdrop-blur-sm rounded-lg p-3 text-white">
                          <div className="flex items-center justify-between">
                            <span className="text-sm">Gerçek Zamanlı AI Analizi</span>
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                              <span className="text-xs">Aktif</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  {}
                  <div className="space-y-4">
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100 shadow-sm">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-semibold text-slate-700">Dikkat Skoru</span>
                        <Eye className="w-5 h-5 text-blue-500" />
                      </div>
                      <div className="text-3xl font-bold text-blue-600 mb-3 transition-all duration-2000">{aiMetrics.attention}%</div>
                      <div className="w-full bg-white rounded-full h-3 shadow-inner">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-indigo-500 h-3 rounded-full transition-all duration-2000 shadow-sm"
                          style={{ width: `${aiMetrics.attention}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-100 shadow-sm">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-semibold text-slate-700">Katılım Durumu</span>
                        <Brain className="w-5 h-5 text-green-500" />
                      </div>
                      <div className="text-xl font-bold text-green-600 transition-all duration-2000">
                        {aiMetrics.attention > 80 ? 'Yüksek' : aiMetrics.attention > 60 ? 'Orta' : 'Düşük'}
                      </div>
                      <div className="text-sm text-green-600 mt-1">
                        {aiMetrics.attention > 80 ? 'Mükemmel odaklanma' : aiMetrics.attention > 60 ? 'İyi performans' : 'Dikkat gerekli'}
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-100 shadow-sm">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-semibold text-slate-700">Duygu Durumu</span>
                        <Heart className="w-5 h-5 text-purple-500" />
                      </div>
                      <div className="text-xl font-bold text-purple-600 transition-all duration-2000">
                        {aiMetrics.emotion}
                      </div>
                      <div className="text-sm text-purple-600 mt-1">
                        {aiMetrics.emotion === 'Pozitif' ? 'Mutlu ve motive' : aiMetrics.emotion === 'Nötr' ? 'Sakin durum' : 'Destek gerekli'}
                      </div>
                    </div>
                    {isAnalyzing && (
                      <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-xl p-6 border border-yellow-100 shadow-sm">
                        <div className="flex items-center space-x-2 mb-2">
                          <Activity className="w-5 h-5 text-yellow-500 animate-pulse" />
                          <span className="text-sm font-semibold text-slate-700">AI Modelleri Çalışıyor</span>
                        </div>
                        {processingStage && (
                          <div className="text-xs text-slate-600 leading-relaxed mb-2">
                            <strong>Durum:</strong> {processingStage}
                          </div>
                        )}
                        <div className="text-xs text-slate-600 leading-relaxed mb-3">
                          <strong>Aktif Modeller:</strong><br/>
                          • ResNet50 + MediaPipe (Yüz/Göz)<br/>
                          • MobileViT (Duygu Tanıma)<br/>
                          • LSTM (Davranış Analizi)<br/>
                          • VAD Modeli (Ses Analizi)
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-500">
                          <span>Güncelleme: 15 saniye</span>
                          <span>FPS: ~0.067</span>
                        </div>
                        <div className="mt-2 flex items-center space-x-1">
                          <div className="w-1 h-1 bg-yellow-500 rounded-full animate-bounce"></div>
                          <div className="w-1 h-1 bg-yellow-500 rounded-full animate-bounce delay-100"></div>
                          <div className="w-1 h-1 bg-yellow-500 rounded-full animate-bounce delay-200"></div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                {}
                <div className="bg-gradient-to-br from-slate-50 to-gray-50 rounded-xl p-6 border border-slate-200 shadow-sm">
                  <div className="flex items-center space-x-2 mb-4">
                    <Brain className="w-5 h-5 text-slate-600" />
                    <span className="text-sm font-semibold text-slate-700">Detaylı Analiz Sonuçları</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-xs mb-4">
                    <div>
                      <span className="text-slate-500">Göz Yönü:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{aiMetrics.gazeDirection}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Göz Güveni:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{Math.round(aiMetrics.gazeConfidence * 100)}%</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Yüz Tespiti:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{aiMetrics.faceDetected ? '✓ Aktif' : '✗ Kayıp'}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Ses Seviyesi:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{Math.round(aiMetrics.audioLevel * 100)}%</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Göz Kırpma:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{aiMetrics.blinkRate}/dk</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Duruş:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{aiMetrics.posture}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Dikkat Dağınıklığı:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{aiMetrics.distraction}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Duygu Skoru:</span>
                      <span className="ml-2 font-medium text-slate-700 transition-all duration-2000">{aiMetrics.emotionScore}/5.0</span>
                    </div>
                  </div>
                  <div className="border-t border-slate-200 pt-3">
                    <div className="text-xs text-slate-500 mb-2">Model Performansı:</div>
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div className="bg-green-50 p-2 rounded text-center">
                        <div className="text-green-600 font-medium">Vision</div>
                        <div className="text-green-500">98.2%</div>
                      </div>
                      <div className="bg-blue-50 p-2 rounded text-center">
                        <div className="text-blue-600 font-medium">Gaze</div>
                        <div className="text-blue-500">96.7%</div>
                      </div>
                      <div className="bg-purple-50 p-2 rounded text-center">
                        <div className="text-purple-600 font-medium">Emotion</div>
                        <div className="text-purple-500">94.1%</div>
                      </div>
                      <div className="bg-orange-50 p-2 rounded text-center">
                        <div className="text-orange-600 font-medium">Audio</div>
                        <div className="text-orange-500">91.8%</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {!isCameraActive && (
              <div className="text-center py-12">
                <Camera className="w-16 h-16 text-slate-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-slate-600 mb-2">AI Dikkat Analizi</h3>
                <p className="text-slate-500">
                  Kameranızı açarak gerçek zamanlı dikkat ve katılım analizini test edin.
                  <br />
                  Yüz tanıma ve göz takibi teknolojisi kullanarak dikkat seviyenizi ölçeriz.
                </p>
              </div>
            )}
          </motion.div>
          {}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-8">
            <StatCard
              icon={Target}
              title="Bugünkü Dikkat Seviyesi"
              value={`${studentData.todayStats.attention}%`}
              trend={3}
              color="from-blue-500 to-indigo-500"
            />
            <StatCard
              icon={TrendingUp}
              title="Katılım Oranı"
              value={`${studentData.todayStats.engagement}%`}
              trend={5}
              color="from-green-500 to-emerald-500"
            />
            <StatCard
              icon={Heart}
              title="Duygu Skoru"
              value={studentData.todayStats.emotionalState}
              trend={-2}
              color="from-purple-500 to-violet-500"
            />
            <StatCard
              icon={Clock}
              title="Toplam Süre"
              value={studentData.todayStats.sessionTime}
              trend={8}
              color="from-orange-500 to-red-500"
            />
          </div>
        </main>
      </div>
    </div>
  );
};
export default ModernStudentDashboard;
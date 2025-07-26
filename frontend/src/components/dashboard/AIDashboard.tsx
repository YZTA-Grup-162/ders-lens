import { AnimatePresence, motion } from 'framer-motion';
import {
    Activity,
    BarChart3,
    Brain,
    Camera,
    CheckCircle,
    Eye,
    Monitor,
    Pause,
    Play,
    Settings,
    Users,
    Wifi,
    Zap
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useAI } from '../../stores/aiStore';
import { OptimizedCameraAnalysis } from '../camera/OptimizedCameraAnalysis';
import { CameraTestCallToAction } from '../common/CameraTestCallToAction';
import { AttentionHeatmap } from './components/AttentionHeatmap';
import { EmotionVisualization } from './components/EmotionVisualization';
import { NeuralNetworkBackground } from './components/NeuralNetworkBackground';
import { RealTimeMetrics } from './components/RealTimeMetrics';
import { StudentGrid } from './components/StudentGrid';
interface DashboardMetrics {
  totalStudents: number;
  activeStudents: number;
  averageAttention: number;
  averageEngagement: number;
  dominantEmotion: string;
  modelAccuracy: number;
  processingSpeed: number;
  frameRate: number;
}
const EMOTION_COLORS = {
  neutral: '#6B7280',
  happiness: '#10B981',
  surprise: '#F59E0B',
  sadness: '#3B82F6',
  anger: '#EF4444',
  disgust: '#F97316',
  fear: '#8B5CF6',
  contempt: '#EC4899'
};
export function AIDashboard() {
  const { state, actions } = useAI();
  const [isLive, setIsLive] = useState(false);
  const [selectedView, setSelectedView] = useState<'overview' | 'emotions' | 'attention' | 'students' | 'camera'>('overview');
  const [showCameraTooltip, setShowCameraTooltip] = useState(true);
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalStudents: 24,
    activeStudents: 22,
    averageAttention: 84,
    averageEngagement: 81,
    dominantEmotion: 'neutral',
    modelAccuracy: 94.2,
    processingSpeed: 23,
    frameRate: 30
  });
  const dashboardRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!isLive) return;
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        averageAttention: Math.max(70, Math.min(98, prev.averageAttention + (Math.random() - 0.5) * 6)),
        averageEngagement: Math.max(65, Math.min(95, prev.averageEngagement + (Math.random() - 0.5) * 4)),
        processingSpeed: Math.max(15, Math.min(35, prev.processingSpeed + (Math.random() - 0.5) * 2)),
        frameRate: Math.max(25, Math.min(32, prev.frameRate + (Math.random() - 0.5) * 1))
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, [isLive]);
  const handleStartAnalysis = async () => {
    setIsLive(true);
    setSelectedView('camera'); 
    await actions.startAnalysis();
  };
  const handleStopAnalysis = () => {
    setIsLive(false);
    actions.stopAnalysis();
  };
  const navItems = [
    { id: 'camera', label: '🔴 Canlı Kamera Testi', icon: Camera, priority: true },
    { id: 'overview', label: 'Genel Bakış', icon: BarChart3 },
    { id: 'emotions', label: 'Duygu Analizi', icon: Brain },
    { id: 'attention', label: 'Dikkat Analizi', icon: Eye },
    { id: 'students', label: 'Katılımcı Detayları', icon: Users }
  ] as const;
  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {}
      <NeuralNetworkBackground />
      {}
      <div className="relative z-10 p-6">
        {}
        <motion.header 
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <motion.div
                  className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Brain className="w-6 h-6 text-white" />
                </motion.div>
                <motion.div
                  className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  Ders Lens - Online Eğitim AI Analizi
                </h1>
                <p className="text-gray-400 text-sm mt-1">
                  Uzaktan eğitimde katılımcı dikkat, katılım ve duygu analizi
                </p>
              </div>
            </div>
            {}
            <div className="flex items-center space-x-4">
              {}
              <div className="flex items-center space-x-2 px-4 py-2 bg-white/5 backdrop-blur-md rounded-lg border border-white/10">
                <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-gray-500'}`} />
                <span className="text-sm font-medium">
                  {isLive ? 'Canlı Analiz' : 'Beklemede'}
                </span>
              </div>
              {}
              <motion.button
                className={`px-6 py-3 rounded-lg font-medium flex items-center space-x-2 transition-all duration-300 ${
                  isLive 
                    ? 'bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30' 
                    : 'bg-blue-500/20 border border-blue-500/30 text-blue-400 hover:bg-blue-500/30'
                }`}
                onClick={isLive ? handleStopAnalysis : handleStartAnalysis}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isLive ? (
                  <>
                    <Pause className="w-4 h-4" />
                    <span>Durdur</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Başlat</span>
                  </>
                )}
              </motion.button>
              {}
              <motion.button
                className="p-3 bg-white/5 backdrop-blur-md rounded-lg border border-white/10 hover:bg-white/10 transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Settings className="w-5 h-5" />
              </motion.button>
            </div>
          </div>
          {}
          <nav className="flex space-x-2 bg-white/5 backdrop-blur-md rounded-lg p-2 border border-white/10">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isCameraTab = item.id === 'camera';
              const isPriority = 'priority' in item && item.priority;
              return (
                <motion.button
                  key={item.id}
                  className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-md font-medium transition-all duration-300 relative ${
                    selectedView === item.id
                      ? isCameraTab
                        ? 'bg-gradient-to-r from-red-500/40 to-orange-500/40 text-white shadow-lg border border-red-400/50'
                        : 'bg-blue-500/30 text-blue-400 shadow-lg'
                      : isCameraTab
                      ? 'text-orange-300 hover:text-orange-200 hover:bg-red-500/20 bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/30'
                      : 'text-gray-400 hover:text-white hover:bg-white/10'
                  }`}
                  onClick={() => setSelectedView(item.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {isCameraTab && (
                    <>
                      <motion.div
                        className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                      <motion.div
                        className="absolute -top-2 -right-2 w-5 h-5 bg-red-500/30 rounded-full"
                        animate={{ scale: [1, 1.5, 1], opacity: [0.3, 0, 0.3] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                    </>
                  )}
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                  {isPriority && (
                    <span className="text-xs bg-red-500 text-white px-1.5 py-0.5 rounded-full ml-1">
                      DEMO
                    </span>
                  )}
                </motion.button>
              );
            })}
          </nav>
          {}
          <AnimatePresence>
            {showCameraTooltip && selectedView !== 'camera' && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="relative mt-4"
              >
                <div className="bg-gradient-to-r from-red-500/90 to-orange-500/90 backdrop-blur-sm text-white px-6 py-4 rounded-xl shadow-xl border border-red-400/50 max-w-lg mx-auto">
                  <div className="flex items-center gap-3 mb-2">
                    <div>
                      <Camera className="w-5 h-5" />
                    </div>
                    <span className="font-bold text-lg">Öne Çıkan Özellik</span>
                  </div>
                  <p className="text-sm mb-3 leading-relaxed">
                    <strong>"Canlı Kamera Analizi"</strong> ile gerçek zamanlı duygu, dikkat ve katılım analizlerini görüntüleyin.
                  </p>
                  <div className="flex items-center gap-3">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setSelectedView('camera');
                        setShowCameraTooltip(false);
                      }}
                      className="bg-white/20 hover:bg-white/30 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300"
                    >
                      Analiz Başlat
                    </motion.button>
                    <button
                      onClick={() => setShowCameraTooltip(false)}
                      className="text-white/70 hover:text-white text-sm transition-colors"
                    >
                      Daha sonra
                    </button>
                  </div>
                  <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-gradient-to-r from-red-500/90 to-orange-500/90 rotate-45 border-r border-b border-red-400/50"></div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.header>
        {}
        <motion.main
          className="space-y-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Users,
                label: 'Toplam Katılımcı',
                value: metrics.totalStudents,
                change: '+2',
                color: 'blue'
              },
              {
                icon: Eye,
                label: 'Ortalama Dikkat',
                value: `${metrics.averageAttention}%`,
                change: '+3.2%',
                color: 'green'
              },
              {
                icon: Activity,
                label: 'Katılım Oranı',
                value: `${metrics.averageEngagement}%`,
                change: '+1.8%',
                color: 'purple'
              },
              {
                icon: Zap,
                label: 'İşleme Hızı',
                value: `${metrics.processingSpeed}ms`,
                change: '-2ms',
                color: 'cyan'
              }
            ].map((stat, index) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={index}
                  className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, y: -2 }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-3 bg-${stat.color}-500/20 rounded-lg`}>
                      <Icon className={`w-6 h-6 text-${stat.color}-400`} />
                    </div>
                    <span className="text-green-400 text-sm font-medium">
                      {stat.change}
                    </span>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">{stat.label}</p>
                    <p className="text-2xl font-bold text-white">{stat.value}</p>
                  </div>
                </motion.div>
              );
            })}
          </div>
          {}
          <AnimatePresence mode="wait">
            {selectedView === 'overview' && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                {}
                <CameraTestCallToAction 
                  onStartCamera={() => setSelectedView('camera')}
                  className="mb-8"
                />
                <RealTimeMetrics metrics={metrics} isLive={isLive} />
              </motion.div>
            )}
            {selectedView === 'emotions' && (
              <motion.div
                key="emotions"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <EmotionVisualization emotions={state.students} isLive={isLive} />
              </motion.div>
            )}
            {selectedView === 'attention' && (
              <motion.div
                key="attention"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <AttentionHeatmap students={state.students} isLive={isLive} />
              </motion.div>
            )}
            {selectedView === 'students' && (
              <motion.div
                key="students"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <StudentGrid students={state.students} isLive={isLive} />
              </motion.div>
            )}
            {selectedView === 'camera' && (
              <motion.div
                key="camera"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <OptimizedCameraAnalysis 
                  onAnalysisResult={(result) => {
                    console.log('Real-time student analysis result:', result);
                  }}
                  analysisInterval={2000}
                  autoStart={isLive}
                />
              </motion.div>
            )}
          </AnimatePresence>
          {}
          <motion.div
            className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <Camera className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-gray-400">Kamera: Aktif</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Wifi className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-gray-400">WebSocket: Bağlı</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Monitor className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-gray-400">FPS: {metrics.frameRate}</span>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-sm text-gray-400">Model: FER2013+ v2.1</span>
                <span className="text-sm text-green-400 font-medium">
                  {metrics.modelAccuracy}% doğruluk
                </span>
              </div>
            </div>
          </motion.div>
        </motion.main>
      </div>
    </div>
  );
}
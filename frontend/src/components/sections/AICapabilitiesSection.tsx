import { motion, useInView } from 'framer-motion';
import { BarChart3, Brain, Camera, Cpu, Eye, Shield, Target, Zap } from 'lucide-react';
import { useRef, useState } from 'react';
import { useAI } from '../../stores/aiStore';
const aiCapabilities = [
  {
    id: 'fer2013',
    title: 'FER2013+ Duygu Tanıma',
    description: 'Yüz ifadelerinden 8 farklı duyguyu %94.2 doğrulukla tespit eder',
    features: [
      'Nötr, mutluluk, şaşırma, üzüntü duyguları',
      'Öfke, iğrenme, korku, küçümseme analizı',
      'Gerçek zamanlı yüz ifadesi takibi',
      'Çoklu öğrenci eş zamanlı analizi'
    ],
    metrics: {
      accuracy: '94.2%',
      speed: '< 23ms',
      emotions: '8 Farklı',
      dataset: 'FER2013+'
    },
    icon: <Brain className="w-8 h-8" />,
    color: 'from-purple-500 to-pink-500',
    demoType: 'emotion'
  },
  {
    id: 'daisee',
    title: 'DAISEE Dikkat Analizi',
    description: 'Öğrenci dikkat seviyesini gözbebeği ve göz hareketleri ile izler',
    features: [
      'Göz kapağı pozisyon analizi',
      'Bakış yönü tespiti',
      'Dikkat dağınıklığı uyarıları',
      'Odaklanma süresi ölçümü'
    ],
    metrics: {
      accuracy: '92.5%',
      speed: '30 FPS',
      tracking: '16 Nokta',
      latency: '< 16ms'
    },
    icon: <Eye className="w-8 h-8" />,
    color: 'from-blue-500 to-cyan-500',
    demoType: 'attention'
  },
  {
    id: 'mpiigaze',
    title: 'MPIIGaze Bakış Takibi',
    description: 'Öğrencilerin hangi noktalara baktığını hassas şekilde belirler',
    features: [
      'Bakış yönü vektör analizi',
      'İlgi alanı haritalama',
      'Odak noktası geçmişi',
      'Dikkat dağılım analizi'
    ],
    metrics: {
      accuracy: '±2.5°',
      range: '360°',
      calibration: 'Otomatik',
      points: 'Sınırsız'
    },
    icon: <Target className="w-8 h-8" />,
    color: 'from-green-500 to-teal-500',
    demoType: 'gaze'
  },
  {
    id: 'engagement',
    title: 'Çoklu Faktör Katılım',
    description: 'Tüm verileri birleştirerek kapsamlı katılım skoru hesaplar',
    features: [
      '12 farklı parametre analizi',
      'Dinamik ağırlıklandırma',
      'Gerçek zamanlı skorlama',
      'Öğrenci profil oluşturma'
    ],
    metrics: {
      factors: '12 Parametre',
      update: 'Anlık',
      accuracy: '89.3%',
      profiles: 'Kişisel'
    },
    icon: <BarChart3 className="w-8 h-8" />,
    color: 'from-orange-500 to-red-500',
    demoType: 'engagement'
  }
];
const techSpecs = [
  {
    icon: <Camera className="w-6 h-6" />,
    title: 'WebRTC Video İşleme',
    description: 'Düşük gecikme ile video akışı',
    specs: ['1080p @ 30fps', 'Adaptif kalite', 'Bandwidth optimizasyonu']
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: 'ONNX Model Optimizasyonu',
    description: 'Hızlı AI model çıkarımı',
    specs: ['Cross-platform', 'GPU acceleration', 'Quantized models']
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: 'Gerçek Zamanlı İşleme',
    description: 'Minimum gecikme garantisi',
    specs: ['< 50ms end-to-end', 'WebSocket iletişimi', 'Async processing']
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: 'Gizlilik ve Güvenlik',
    description: 'KVKK uyumlu veri koruması',
    specs: ['Local processing', 'Encrypted transmission', 'No data storage']
  }
];
export function AICapabilitiesSection() {
  const { state, actions } = useAI();
  const [activeDemo, setActiveDemo] = useState<string | null>(null);
  const [hoveredCapability, setHoveredCapability] = useState<string | null>(null);
  const sectionRef = useRef(null);
  const isInView = useInView(sectionRef, { once: true });
  const startDemo = async (demoType: string) => {
    setActiveDemo(demoType);
    if (!state.isAnalyzing) {
      await actions.startAnalysis();
    }
  };
  return (
    <section ref={sectionRef} className="py-20 bg-gradient-to-b from-black to-gray-900 relative overflow-hidden">
      {}
      <div className="absolute inset-0">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-blue-900/10 via-purple-900/10 to-cyan-900/10"
          animate={{
            backgroundPosition: ["0% 0%", "100% 100%"],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
        {}
        <svg className="absolute inset-0 w-full h-full opacity-5" viewBox="0 0 1000 1000">
          <defs>
            <radialGradient id="aiGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.8"/>
              <stop offset="100%" stopColor="#1E40AF" stopOpacity="0"/>
            </radialGradient>
          </defs>
          {}
          {Array.from({ length: 30 }).map((_, i) => (
            <motion.circle
              key={i}
              cx={50 + (i % 6) * 150}
              cy={100 + Math.floor(i / 6) * 120}
              r="3"
              fill="url(#aiGradient)"
              animate={{
                opacity: [0.2, 1, 0.2],
                scale: [1, 1.5, 1],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                delay: i * 0.15,
              }}
            />
          ))}
        </svg>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <motion.h2 
            className="text-4xl md:text-5xl font-bold text-white mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Yapay Zeka
            </span>
            <br />
            <span className="text-white">Analiz Motorları</span>
          </motion.h2>
          <motion.p 
            className="text-xl text-gray-300 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Ders Lens, en güncel AI modellerini ve algoritmaları kullanarak 
            eğitim ortamınızda derinlemesine analiz ve öngörü sağlar.
          </motion.p>
        </motion.div>
        {}
        <div className="grid lg:grid-cols-2 gap-8 mb-16">
          {aiCapabilities.map((capability, index) => (
            <motion.div
              key={capability.id}
              className="bg-white/5 backdrop-blur-md rounded-3xl border border-white/10 overflow-hidden group"
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.6 + index * 0.2 }}
              onHoverStart={() => setHoveredCapability(capability.id)}
              onHoverEnd={() => setHoveredCapability(null)}
              whileHover={{ y: -5, scale: 1.02 }}
            >
              <div className="p-8">
                {}
                <div className="flex items-start justify-between mb-6">
                  <motion.div
                    className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${capability.color} p-3 shadow-2xl`}
                    whileHover={{ rotate: 5, scale: 1.1 }}
                  >
                    <div className="text-white w-full h-full flex items-center justify-center">
                      {capability.icon}
                    </div>
                  </motion.div>
                  <motion.button
                    onClick={() => startDemo(capability.demoType)}
                    className={`px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-300 ${
                      activeDemo === capability.demoType
                        ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                        : 'bg-white/10 text-white border border-white/20 hover:bg-white/20'
                    }`}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {activeDemo === capability.demoType ? 'Demo Aktif' : 'Demo Başlat'}
                  </motion.button>
                </div>
                {}
                <h3 className="text-2xl font-bold text-white mb-3">{capability.title}</h3>
                <p className="text-gray-300 text-sm leading-relaxed mb-6">
                  {capability.description}
                </p>
                {}
                <div className="space-y-2 mb-6">
                  {capability.features.map((feature, i) => (
                    <motion.div
                      key={i}
                      className="flex items-center text-sm text-gray-300"
                      initial={{ opacity: 0, x: -20 }}
                      animate={hoveredCapability === capability.id ? { opacity: 1, x: 0 } : { opacity: 0.7, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                    >
                      <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${capability.color} mr-3`} />
                      {feature}
                    </motion.div>
                  ))}
                </div>
                {}
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(capability.metrics).map(([key, value], i) => (
                    <motion.div
                      key={key}
                      className="bg-white/5 rounded-xl p-3 border border-white/10"
                      initial={{ opacity: 0, y: 20 }}
                      animate={hoveredCapability === capability.id ? { opacity: 1, y: 0 } : { opacity: 0.8, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                    >
                      <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">
                        {key === 'accuracy' ? 'Doğruluk' :
                         key === 'speed' ? 'Hız' :
                         key === 'emotions' ? 'Duygu' :
                         key === 'dataset' ? 'Veri Seti' :
                         key === 'tracking' ? 'Takip' :
                         key === 'latency' ? 'Gecikme' :
                         key === 'range' ? 'Menzil' :
                         key === 'calibration' ? 'Kalibrasyon' :
                         key === 'points' ? 'Nokta' :
                         key === 'factors' ? 'Faktör' :
                         key === 'update' ? 'Güncelleme' :
                         key === 'profiles' ? 'Profil' : key}
                      </div>
                      <div className={`font-bold bg-gradient-to-r ${capability.color} bg-clip-text text-transparent`}>
                        {value}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
              {}
              {activeDemo === capability.demoType && (
                <motion.div
                  className="bg-green-500/10 border-t border-green-500/20 p-4"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <div className="flex items-center justify-center text-green-400">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-3 animate-pulse" />
                    <span className="text-sm font-medium">Demo çalışıyor - Gerçek zamanlı analiz aktif</span>
                  </div>
                </motion.div>
              )}
            </motion.div>
          ))}
        </div>
        {}
        <motion.div
          className="bg-white/5 backdrop-blur-md rounded-3xl p-8 border border-white/10"
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 1.4 }}
        >
          <h3 className="text-2xl font-bold text-white mb-8 text-center">
            Teknik Altyapı ve Performans
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {techSpecs.map((spec, index) => (
              <motion.div
                key={spec.title}
                className="bg-white/5 rounded-2xl p-6 border border-white/10 text-center"
                initial={{ opacity: 0, y: 30 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: 1.6 + index * 0.1 }}
                whileHover={{ y: -5, backgroundColor: "rgba(255,255,255,0.1)" }}
              >
                <motion.div
                  className="w-12 h-12 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-xl p-2.5 mx-auto mb-4 border border-blue-500/30"
                  whileHover={{ rotate: 5, scale: 1.1 }}
                >
                  <div className="text-blue-400 w-full h-full flex items-center justify-center">
                    {spec.icon}
                  </div>
                </motion.div>
                <h4 className="text-white font-semibold mb-2">{spec.title}</h4>
                <p className="text-gray-400 text-sm mb-4">{spec.description}</p>
                <div className="space-y-1">
                  {spec.specs.map((item, i) => (
                    <div key={i} className="text-xs text-gray-500">
                      {item}
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
        {}
        <motion.div
          className="mt-12 text-center"
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 2 }}
        >
          <div className="inline-flex items-center bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full px-8 py-4 border border-white/20">
            <motion.div
              className="w-3 h-3 bg-green-400 rounded-full mr-4"
              animate={{ scale: [1, 1.2, 1], opacity: [1, 0.8, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="text-white font-medium">
              Tüm AI modelleri %90+ doğrulukla gerçek zamanlı çalışıyor
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
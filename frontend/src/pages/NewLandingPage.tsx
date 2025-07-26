import {
    ArrowRightIcon,
    BeakerIcon,
    BoltIcon,
    CameraIcon,
    ChartBarIcon,
    CheckCircleIcon,
    ClockIcon,
    CpuChipIcon,
    EyeIcon,
    FaceSmileIcon,
    PlayIcon,
    ShieldCheckIcon,
    SparklesIcon,
    UserGroupIcon
} from '@heroicons/react/24/outline';
import { AnimatePresence, motion, useScroll, useTransform } from 'framer-motion';
import React, { useEffect, useState } from 'react';
const NewLandingPage: React.FC = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeDemo, setActiveDemo] = useState('attention');
  const { scrollY } = useScroll();
  const heroY = useTransform(scrollY, [0, 300], [0, -50]);
  const heroOpacity = useTransform(scrollY, [0, 300], [1, 0.8]);
  useEffect(() => {
    setIsLoaded(true);
  }, []);
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };
  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.6, ease: "easeOut" }
    }
  };
  const floatingVariants = {
    animate: {
      y: [0, -10, 0],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };
  const features = [
    {
      icon: <EyeIcon className="w-8 h-8" />,
      title: "Dikkat Takibi",
      description: "Yüz yönelimi ve varlık tespiti ile öğrenci dikkat seviyesini anlık olarak izler ve analiz eder.",
      stats: "98% Doğruluk",
      gradient: "from-blue-500 via-blue-600 to-cyan-500",
      delay: 0
    },
    {
      icon: <ChartBarIcon className="w-8 h-8" />,
      title: "Katılım Tahmini",
      description: "Hareket, ekran etkileşimi ve duruş analizine dayalı olarak öğrenci katılım seviyesini ölçer.",
      stats: "Gerçek Zamanlı",
      gradient: "from-purple-500 via-purple-600 to-pink-500",
      delay: 0.1
    },
    {
      icon: <FaceSmileIcon className="w-8 h-8" />,
      title: "Duygu Tanıma",
      description: "Yüzsel mikro-ifadeleri kullanarak karışıklık, sıkılma ve heyecan gibi duyguları sınıflandırır.",
      stats: "7 Duygu Tipi",
      gradient: "from-green-500 via-emerald-500 to-teal-500",
      delay: 0.2
    },
    {
      icon: <CameraIcon className="w-8 h-8" />,
      title: "Bakış Haritalaması",
      description: "Göz takibi teknolojisiyle öğrencinin ekranın hangi bölümüne odaklandığını tespit eder.",
      stats: "Milisaniye Hassasiyet",
      gradient: "from-orange-500 via-red-500 to-pink-500",
      delay: 0.3
    }
  ];
  const techFeatures = [
    {
      icon: <CpuChipIcon className="w-6 h-6" />,
      title: "Yapay Zeka Destekli",
      description: "Gelişmiş makine öğrenmesi algoritmaları"
    },
    {
      icon: <BoltIcon className="w-6 h-6" />,
      title: "Gerçek Zamanlı",
      description: "Anlık analiz ve değerlendirme"
    },
    {
      icon: <ShieldCheckIcon className="w-6 h-6" />,
      title: "KVKK Uyumlu",
      description: "Veri güvenliği ve gizlilik odaklı"
    },
    {
      icon: <BeakerIcon className="w-6 h-6" />,
      title: "Araştırma Tabanlı",
      description: "Bilimsel yöntemlerle geliştirilmiş"
    }
  ];
  const stats = [
    { value: "98%", label: "Doğruluk Oranı", icon: <CheckCircleIcon className="w-6 h-6" /> },
    { value: "50ms", label: "Analiz Süresi", icon: <ClockIcon className="w-6 h-6" /> },
    { value: "24/7", label: "Çalışma Süresi", icon: <BoltIcon className="w-6 h-6" /> },
    { value: "500+", label: "Aktif Kullanıcı", icon: <UserGroupIcon className="w-6 h-6" /> }
  ];
  const demoTabs = [
    { id: 'attention', label: 'Dikkat Analizi', icon: <EyeIcon className="w-5 h-5" /> },
    { id: 'emotion', label: 'Duygu Tanıma', icon: <FaceSmileIcon className="w-5 h-5" /> },
    { id: 'engagement', label: 'Katılım Ölçümü', icon: <ChartBarIcon className="w-5 h-5" /> },
    { id: 'gaze', label: 'Bakış Takibi', icon: <CameraIcon className="w-5 h-5" /> }
  ];
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {}
      <motion.nav 
        className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/20"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {}
            <motion.div 
              className="flex items-center space-x-3"
              whileHover={{ scale: 1.05 }}
            >
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                <img 
                  src="/derslens-logo.png" 
                  alt="Ders Lens" 
                  className="w-8 h-8 object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                    (e.target as HTMLImageElement).parentElement!.innerHTML = '<div class="text-white font-bold text-lg">DL</div>';
                  }}
                />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                Ders Lens
              </span>
            </motion.div>
            {}
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-gray-700 hover:text-blue-600 transition-colors font-medium">
                Özellikler
              </a>
              <a href="#demo" className="text-gray-700 hover:text-blue-600 transition-colors font-medium">
                Demo
              </a>
              <a href="#technology" className="text-gray-700 hover:text-blue-600 transition-colors font-medium">
                Teknoloji
              </a>
              <motion.button
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-2 rounded-lg font-medium hover:shadow-lg transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Demoyu Başlat
              </motion.button>
            </div>
          </div>
        </div>
      </motion.nav>
      {}
      <section className="relative pt-24 pb-16 overflow-hidden">
        <motion.div 
          style={{ y: heroY, opacity: heroOpacity }}
          className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
        >
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {}
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="space-y-8"
            >
              {}
              <motion.div
                variants={itemVariants}
                className="inline-flex items-center space-x-2 bg-blue-50 border border-blue-200 rounded-full px-4 py-2"
              >
                <SparklesIcon className="w-5 h-5 text-blue-600" />
                <span className="text-sm font-medium text-blue-700">Yapay Zeka Destekli Eğitim Analizi</span>
              </motion.div>
              {}
              <motion.div variants={itemVariants} className="space-y-4">
                <h1 className="text-5xl lg:text-6xl font-bold leading-tight">
                  <span className="bg-gradient-to-r from-gray-900 via-blue-800 to-purple-800 bg-clip-text text-transparent">
                    Öğrenci Odağını
                  </span>
                  <br />
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    AI ile Anlayın
                  </span>
                </h1>
                <p className="text-xl text-gray-600 max-w-2xl leading-relaxed">
                  Ders Lens, öğrenci dikkat, duygusal durum, bakış yönü ve katılımını gerçek zamanlı olarak 
                  yakalayıp görselleştiren AI destekli platform ile eğitimcilerin öğrenmeyi optimize etmesine yardımcı olur.
                </p>
              </motion.div>
              {}
              <motion.div 
                variants={itemVariants}
                className="flex flex-col sm:flex-row gap-4"
              >
                <motion.button
                  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-xl transition-all inline-flex items-center justify-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <PlayIcon className="w-6 h-6" />
                  <span>Demoyu İzle</span>
                </motion.button>
                <motion.button
                  className="border-2 border-gray-300 text-gray-700 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-gray-50 transition-all inline-flex items-center justify-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>Daha Fazla Bilgi</span>
                  <ArrowRightIcon className="w-6 h-6" />
                </motion.button>
              </motion.div>
              {}
              <motion.div 
                variants={itemVariants}
                className="grid grid-cols-2 gap-4"
              >
                {techFeatures.map((feature, index) => (
                  <motion.div
                    key={index}
                    className="flex items-center space-x-3 p-3 bg-white/50 rounded-lg backdrop-blur-sm"
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="p-2 bg-gradient-to-r from-blue-100 to-purple-100 rounded-lg">
                      {feature.icon}
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-900 text-sm">{feature.title}</h4>
                      <p className="text-gray-600 text-xs">{feature.description}</p>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            </motion.div>
            {}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative"
            >
              {}
              <motion.div
                variants={floatingVariants}
                animate="animate"
                className="bg-white rounded-2xl shadow-2xl border border-gray-200 p-6 relative z-10"
              >
                {}
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                  </div>
                  <span className="text-sm font-medium text-gray-500">Canlı Dashboard</span>
                </div>
                {}
                <div className="flex space-x-1 mb-6 bg-gray-100 rounded-lg p-1">
                  {demoTabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveDemo(tab.id)}
                      className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
                        activeDemo === tab.id
                          ? 'bg-white text-blue-600 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      {tab.icon}
                      <span className="hidden sm:inline">{tab.label}</span>
                    </button>
                  ))}
                </div>
                {}
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeDemo}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    {activeDemo === 'attention' && (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Dikkat Seviyesi</span>
                          <span className="text-lg font-bold text-green-600">87%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <motion.div
                            className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: '87%' }}
                            transition={{ duration: 1, delay: 0.5 }}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-4 mt-4">
                          <div className="bg-blue-50 p-3 rounded-lg">
                            <div className="text-xs text-blue-600 font-medium">Yüz Yönelimi</div>
                            <div className="text-lg font-bold text-blue-700">Merkez</div>
                          </div>
                          <div className="bg-green-50 p-3 rounded-lg">
                            <div className="text-xs text-green-600 font-medium">Göz Durumu</div>
                            <div className="text-lg font-bold text-green-700">Açık</div>
                          </div>
                        </div>
                      </div>
                    )}
                    {activeDemo === 'emotion' && (
                      <div className="space-y-4">
                        <div className="text-center">
                          <div className="text-4xl mb-2">😊</div>
                          <div className="text-lg font-bold text-gray-800">Mutlu</div>
                          <div className="text-sm text-gray-600">Güven: 92%</div>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="text-center p-2 bg-blue-50 rounded">
                            <div>😊 Mutlu</div>
                            <div className="font-bold text-blue-600">92%</div>
                          </div>
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div>😐 Nötr</div>
                            <div className="font-bold text-gray-600">5%</div>
                          </div>
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div>😕 Üzgün</div>
                            <div className="font-bold text-gray-600">3%</div>
                          </div>
                        </div>
                      </div>
                    )}
                    {activeDemo === 'engagement' && (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Katılım Skoru</span>
                          <span className="text-lg font-bold text-purple-600">94%</span>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span>Hareket Analizi</span>
                            <span className="font-medium">85%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div className="bg-purple-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span>Ekran Etkileşimi</span>
                            <span className="font-medium">96%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div className="bg-purple-500 h-2 rounded-full" style={{ width: '96%' }}></div>
                          </div>
                        </div>
                      </div>
                    )}
                    {activeDemo === 'gaze' && (
                      <div className="space-y-4">
                        <div className="bg-gray-100 rounded-lg p-4 relative">
                          <div className="text-xs text-gray-600 mb-2">Ekran Bölgeleri</div>
                          <div className="grid grid-cols-3 gap-1 h-20">
                            <div className="bg-gray-200 rounded opacity-30"></div>
                            <div className="bg-blue-400 rounded"></div>
                            <div className="bg-gray-200 rounded opacity-30"></div>
                            <div className="bg-gray-200 rounded opacity-50"></div>
                            <div className="bg-green-400 rounded"></div>
                            <div className="bg-gray-200 rounded opacity-30"></div>
                            <div className="bg-gray-200 rounded opacity-30"></div>
                            <div className="bg-gray-200 rounded opacity-30"></div>
                            <div className="bg-gray-200 rounded opacity-30"></div>
                          </div>
                          <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-red-500 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm font-medium text-gray-700">Mevcut Odak</div>
                          <div className="text-lg font-bold text-blue-600">Merkez Üst</div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </motion.div>
              {}
              <motion.div
                className="absolute -top-4 -right-4 w-20 h-20 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full opacity-20"
                animate={{
                  scale: [1, 1.2, 1],
                  rotate: [0, 180, 360]
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
              <motion.div
                className="absolute -bottom-6 -left-6 w-16 h-16 bg-gradient-to-r from-green-400 to-blue-400 rounded-full opacity-20"
                animate={{
                  scale: [1.2, 1, 1.2],
                  rotate: [360, 180, 0]
                }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </motion.div>
          </div>
        </motion.div>
        {}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-200 rounded-full filter blur-3xl opacity-20"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-200 rounded-full filter blur-3xl opacity-20"></div>
        </div>
      </section>
      {}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="grid grid-cols-2 lg:grid-cols-4 gap-8"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                className="text-center"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-r from-blue-100 to-purple-100 rounded-xl mb-4">
                  {stat.icon}
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>
      {}
      <section id="features" className="py-20 bg-gradient-to-b from-white to-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {}
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                Gelişmiş AI ile
              </span>
              <br />
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Eğitimi Dönüştürün
              </span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Ders Lens, yapay zeka destekli görüntü analizi ile öğrenci davranışlarını anlık olarak 
              değerlendirerek eğitimcilere derinlemesine içgörüler sağlar.
            </p>
          </motion.div>
          {}
          <div className="grid lg:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="group relative bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-100"
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: feature.delay }}
                viewport={{ once: true }}
                whileHover={{ y: -5 }}
              >
                {}
                <div className={`absolute inset-0 bg-gradient-to-r ${feature.gradient} opacity-0 group-hover:opacity-5 rounded-2xl transition-opacity duration-300`}></div>
                {}
                <div className={`inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r ${feature.gradient} rounded-xl mb-6 relative`}>
                  <div className="text-white">
                    {feature.icon}
                  </div>
                </div>
                {}
                <div className="relative">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-2xl font-bold text-gray-900">{feature.title}</h3>
                    <span className={`px-3 py-1 bg-gradient-to-r ${feature.gradient} text-white text-xs font-semibold rounded-full`}>
                      {feature.stats}
                    </span>
                  </div>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </div>
                {}
                <motion.div
                  className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  whileHover={{ scale: 1.1 }}
                >
                  <ArrowRightIcon className="w-6 h-6 text-gray-400" />
                </motion.div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {}
      <section id="technology" className="py-20 bg-gray-900 text-white relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Teknoloji Stack
              </span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Modern web teknolojileri ve gelişmiş AI modelleri ile güçlendirilmiş platform
            </p>
          </motion.div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { title: "Frontend", tech: "React + TypeScript", icon: "⚛️" },
              { title: "Backend", tech: "FastAPI + Python", icon: "🐍" },
              { title: "AI Modelleri", tech: "Computer Vision", icon: "🤖" },
              { title: "Database", tech: "PostgreSQL", icon: "🗄️" },
              { title: "Real-time", tech: "WebSocket", icon: "⚡" },
              { title: "Deployment", tech: "Docker", icon: "🐳" }
            ].map((item, index) => (
              <motion.div
                key={index}
                className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-blue-500 transition-all"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05 }}
              >
                <div className="text-4xl mb-4">{item.icon}</div>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-gray-400">{item.tech}</p>
              </motion.div>
            ))}
          </div>
        </div>
        {}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0)',
            backgroundSize: '20px 20px'
          }}></div>
        </div>
      </section>
      {}
      <section className="py-20 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              Sınıfınızı Hiç Olmadığı Gibi
              <br />
              <span className="text-blue-200">Anlamaya Başlayın</span>
            </h2>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Ders Lens ile eğitim deneyimini dönüştürün. Öğrencilerinizin gerçek zamanlı 
              davranışlarını analiz ederek daha etkili öğretim stratejileri geliştirin.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-xl transition-all inline-flex items-center justify-center space-x-2"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <PlayIcon className="w-6 h-6" />
                <span>Ücretsiz Demo</span>
              </motion.button>
              <motion.button
                className="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-white hover:text-blue-600 transition-all inline-flex items-center justify-center space-x-2"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span>İletişime Geç</span>
                <ArrowRightIcon className="w-6 h-6" />
              </motion.button>
            </div>
          </motion.div>
        </div>
        {}
        <div className="absolute inset-0">
          <motion.div
            className="absolute top-1/4 left-1/4 w-32 h-32 bg-white rounded-full opacity-10"
            animate={{
              scale: [1, 1.5, 1],
              rotate: [0, 180, 360]
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <motion.div
            className="absolute bottom-1/4 right-1/4 w-24 h-24 bg-blue-200 rounded-full opacity-20"
            animate={{
              scale: [1.5, 1, 1.5],
              rotate: [360, 180, 0]
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </div>
      </section>
      {}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8">
            {}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                  <span className="text-white font-bold text-lg">DL</span>
                </div>
                <span className="text-xl font-bold">Ders Lens</span>
              </div>
              <p className="text-gray-400">
                AI destekli eğitim analizi ile öğrenme deneyimini optimize eden gelecek nesil platform.
              </p>
            </div>
            {}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Bağlantılar</h3>
              <div className="space-y-2">
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Özellikler</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Demo</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Teknoloji</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">GitHub</a>
              </div>
            </div>
            {}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Yasal</h3>
              <div className="space-y-2">
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Gizlilik Politikası</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">Kullanım Şartları</a>
                <a href="#" className="block text-gray-400 hover:text-white transition-colors">İletişim</a>
                <div className="flex items-center space-x-2 text-gray-400">
                  <ShieldCheckIcon className="w-4 h-4" />
                  <span className="text-sm">KVKK Uyumlu</span>
                </div>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 Ders Lens. Tüm hakları saklıdır.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default NewLandingPage;
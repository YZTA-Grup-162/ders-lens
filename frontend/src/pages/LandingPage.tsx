import {
    ArrowRightIcon,
    BeakerIcon,
    BoltIcon,
    CameraIcon,
    ChartBarIcon,
    CpuChipIcon,
    EyeIcon,
    FaceSmileIcon,
    GlobeEuropeAfricaIcon,
    PlayIcon,
    ShieldCheckIcon
} from '@heroicons/react/24/outline';
import { motion, useScroll, useTransform } from 'framer-motion';
import React, { useEffect, useState } from 'react';
import '../styles/landing.css';
const LandingPage: React.FC = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeDemo, setActiveDemo] = useState('attention');
  const { scrollY } = useScroll();
  const heroY = useTransform(scrollY, [0, 300], [0, -50]);
  const heroOpacity = useTransform(scrollY, [0, 300], [1, 0.8]);
  const fadeInUpTransition = {
    duration: 0.6,
    ease: "easeOut"
  };
  const staggerContainer = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
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
      title: "KVKV Uyumlu",
      description: "Veri güvenliği ve gizlilik odaklı"
    },
    {
      icon: <BeakerIcon className="w-6 h-6" />,
      title: "Araştırma Tabanlı",
      description: "Bilimsel yöntemlerle geliştirilmiş"
    },
    {
      icon: <CameraIcon className="h-8 w-8" />,
      title: "Bakış Haritası",
      description: "Göz takibiyle öğrencinin ekranın hangi bölümüne odaklandığını görür.",
      gradient: "from-orange-500 to-red-500"
    }
  ];
  const dashboardMetrics = [
    { label: "Ortalama Dikkat", value: "87%", color: "text-green-600" },
    { label: "Aktif Öğrenci", value: "24/28", color: "text-blue-600" },
    { label: "Katılım Skoru", value: "92%", color: "text-purple-600" },
    { label: "Duygu Durumu", value: "Pozitif", color: "text-emerald-600" }
  ];
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <img 
                src="/derslens-logo.png" 
                alt="Ders Lens Logo" 
                className="h-10 w-10 object-contain"
              />
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Ders Lens
              </span>
            </div>
            <nav className="hidden md:flex space-x-8">
              <a href="#features" className="text-gray-700 hover:text-blue-600 transition-colors">Özellikler</a>
              <a href="#dashboard" className="text-gray-700 hover:text-blue-600 transition-colors">Dashboard</a>
              <a href="#tech" className="text-gray-700 hover:text-blue-600 transition-colors">Teknoloji</a>
            </nav>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-2 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
            >
              Demo Başlat
            </motion.button>
          </div>
        </div>
      </header>
      {}
      <section className="pt-16 pb-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.h1 
              className="text-5xl md:text-7xl font-bold text-gray-900 mb-6"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-cyan-600 bg-clip-text text-transparent">
                Öğrenci Odağını
              </span>
              <br />
              <span className="text-gray-900">
                AI ile Anlayın
              </span>
            </motion.h1>
            <motion.p 
              className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              Ders Lens, öğrenci dikkatini, duygusal durumunu, bakış yönünü ve katılımını gerçek zamanlı olarak 
              yakalayıp görselleştirerek eğitimcilerin öğrenmeyi optimize etmesine yardımcı olur.
            </motion.p>
            <motion.div 
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-xl hover:shadow-2xl transition-all duration-300 flex items-center space-x-2"
              >
                <PlayIcon className="h-6 w-6" />
                <span>Sınıfınızı Keşfetmeye Başlayın</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="border-2 border-gray-300 text-gray-700 px-8 py-4 rounded-xl font-semibold text-lg hover:border-blue-500 hover:text-blue-600 transition-all duration-300 flex items-center space-x-2"
              >
                <span>Demo İzle</span>
                <ArrowRightIcon className="h-5 w-5" />
              </motion.button>
            </motion.div>
          </motion.div>
          {}
          <motion.div 
            className="mt-16 relative"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
          >
            <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-3xl p-8 backdrop-blur-sm border border-white/20">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[...Array(8)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="bg-white/50 rounded-xl p-4 text-center"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 1 + i * 0.1 }}
                  >
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full mx-auto mb-2 flex items-center justify-center">
                      <div className="w-6 h-6 bg-white rounded-full"></div>
                    </div>
                    <div className="text-sm text-gray-600">Öğrenci {i + 1}</div>
                    <div className="text-xs text-green-600 font-semibold">Aktif</div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      {}
      <section id="features" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            transition={fadeInUpTransition}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Güçlü <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">AI Özellikleri</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Gelişmiş yapay zeka teknolojileriyle sınıfınızı hiç olmadığı kadar derinlemesine anlayın
            </p>
          </motion.div>
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
            variants={staggerContainer}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 60 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="relative group"
              >
                <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-100 group-hover:border-transparent group-hover:ring-2 group-hover:ring-blue-500/20">
                  <div className={`w-16 h-16 bg-gradient-to-r ${feature.gradient} rounded-2xl flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>
      {}
      <section id="dashboard" className="py-20 bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            transition={fadeInUpTransition}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Gerçek Zamanlı</span> Dashboard
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Sınıfınızın anlık durumunu tek bakışta görün ve veri odaklı kararlar alın
            </p>
          </motion.div>
          <motion.div 
            className="bg-white rounded-3xl shadow-2xl p-8 border border-gray-200"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {}
            <div className="flex justify-between items-center mb-8">
              <h3 className="text-2xl font-bold text-gray-900">Sınıf Analizi</h3>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600">Canlı</span>
              </div>
            </div>
            {}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
              {dashboardMetrics.map((metric, index) => (
                <motion.div
                  key={index}
                  className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 text-center"
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className={`text-3xl font-bold ${metric.color} mb-2`}>{metric.value}</div>
                  <div className="text-sm text-gray-600">{metric.label}</div>
                </motion.div>
              ))}
            </div>
            {}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Dikkat Isı Haritası</h4>
              <div className="grid grid-cols-8 gap-2">
                {[...Array(32)].map((_, i) => (
                  <motion.div
                    key={i}
                    className={`h-8 rounded ${
                      Math.random() > 0.3 
                        ? 'bg-gradient-to-t from-green-400 to-green-300' 
                        : Math.random() > 0.6 
                        ? 'bg-gradient-to-t from-yellow-400 to-yellow-300'
                        : 'bg-gradient-to-t from-red-400 to-red-300'
                    }`}
                    initial={{ opacity: 0, height: 0 }}
                    whileInView={{ opacity: 1, height: 32 }}
                    transition={{ delay: i * 0.05 }}
                    viewport={{ once: true }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      {}
      <section id="tech" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            transition={fadeInUpTransition}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Gelişmiş</span> Teknoloji
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              En son AI ve bilgisayar görüsü teknolojileriyle güçlendirilmiş
            </p>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div 
              className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-8 text-center"
              initial={{ opacity: 0, y: 60 }}
              animate={{ opacity: 1, y: 0 }}
              transition={fadeInUpTransition}
            >
              <BoltIcon className="h-16 w-16 text-blue-600 mx-auto mb-4" />
              <h3 className="text-xl font-bold text-gray-900 mb-2">AI Destekli</h3>
              <p className="text-gray-600">Sinir ağları ve makine öğrenmesi</p>
            </motion.div>
            <motion.div 
              className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-8 text-center"
              initial={{ opacity: 0, y: 60 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <ShieldCheckIcon className="h-16 w-16 text-green-600 mx-auto mb-4" />
              <h3 className="text-xl font-bold text-gray-900 mb-2">GDPR Uyumlu</h3>
              <p className="text-gray-600">Tam veri gizliliği koruması</p>
            </motion.div>
            <motion.div 
              className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-8 text-center"
              initial={{ opacity: 0, y: 60 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <GlobeEuropeAfricaIcon className="h-16 w-16 text-purple-600 mx-auto mb-4" />
              <h3 className="text-xl font-bold text-gray-900 mb-2">Gerçek Zamanlı</h3>
              <p className="text-gray-600">Anlık analiz ve geri bildirim</p>
            </motion.div>
          </div>
        </div>
      </section>
      {}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Sınıfınızı Hiç Olmadığı Kadar Keşfedin
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Ders Lens ile öğrencilerinizin gerçek ihtiyaçlarını anlayın ve eğitim deneyimini optimize edin
            </p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-white text-blue-600 px-8 py-4 rounded-xl font-bold text-lg shadow-xl hover:shadow-2xl transition-all duration-300"
            >
              Ücretsiz Demo Başlat
            </motion.button>
          </motion.div>
        </div>
      </section>
      {}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <img 
                  src="/derslens-logo.png" 
                  alt="Ders Lens Logo" 
                  className="h-8 w-8 object-contain"
                />
                <span className="text-xl font-bold">Ders Lens</span>
              </div>
              <p className="text-gray-400">
                AI destekli eğitim analizi platformu
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Ürün</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Özellikler</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Fiyatlandırma</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Demo</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Destek</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Dokümantasyon</a></li>
                <li><a href="#" className="hover:text-white transition-colors">İletişim</a></li>
                <li><a href="#" className="hover:text-white transition-colors">GitHub</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Yasal</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Gizlilik Politikası</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Kullanım Şartları</a></li>
                <li><a href="#" className="hover:text-white transition-colors">GDPR</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2025 Ders Lens. Tüm hakları saklıdır.</p>
            <p className="mt-2 text-sm">React • Node.js • AI • Bilgisayar Görüsü ile geliştirilmiştir</p>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default LandingPage;
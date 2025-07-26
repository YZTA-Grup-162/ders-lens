import {
    AcademicCapIcon,
    ArrowRightIcon,
    ChartBarIcon,
    CheckCircleIcon,
    EyeIcon,
    HeartIcon,
    PlayIcon,
    SparklesIcon,
    UserGroupIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
const SimpleLandingPage: React.FC = () => {
  const [activeDemo, setActiveDemo] = useState('attention');
  const demoTabs = [
    { id: 'attention', name: 'Dikkat Analizi', icon: EyeIcon },
    { id: 'emotion', name: 'Duygu Analizi', icon: HeartIcon },
    { id: 'engagement', name: 'Katılım Takibi', icon: ChartBarIcon }
  ];
  const features = [
    {
      icon: <EyeIcon className="h-6 w-6" />,
      title: "Gerçek Zamanlı Dikkat Analizi",
      description: "AI ile öğrenci dikkatini anlık olarak ölçer ve raporlar.",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      icon: <HeartIcon className="h-6 w-6" />,
      title: "Duygu Durumu Takibi",
      description: "Öğrencilerin duygusal durumlarını analiz ederek öğrenme kalitesini artırır.",
      gradient: "from-pink-500 to-rose-500"
    },
    {
      icon: <ChartBarIcon className="h-6 w-6" />,
      title: "Katılım Metrikleri",
      description: "Detaylı katılım raporları ve öğrenci başarı analizi sunar.",
      gradient: "from-purple-500 to-indigo-500"
    }
  ];
  const stats = [
    { label: "Ortalama Dikkat", value: "87%", color: "text-emerald-600" },
    { label: "Aktif Öğrenci", value: "24/28", color: "text-blue-600" },
    { label: "Katılım Skoru", value: "92%", color: "text-purple-600" }
  ];
  const techStack = [
    { name: "TensorFlow", description: "Deep Learning Framework" },
    { name: "OpenCV", description: "Bilgisayarlı Görü" },
    { name: "PyTorch", description: "Neural Networks" },
    { name: "React", description: "Modern UI Framework" }
  ];
  return (
    <div className="min-h-screen bg-white">
      {}
      <nav className="bg-white/95 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <AcademicCapIcon className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">Ders Lens</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#home" className="text-blue-600 font-medium">Anasayfa</a>
              <Link to="/student-dashboard" className="text-gray-700 hover:text-blue-600 transition-colors">Gelişmiş Öğrenci Dashboard</Link>
              <Link to="/teacher-dashboard" className="text-gray-700 hover:text-blue-600 transition-colors">Eğitmen Gelişmiş Dashboard</Link>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Başla
            </motion.button>
          </div>
        </div>
      </nav>
      {}
      <section className="relative bg-gradient-to-br from-blue-50 via-white to-purple-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="space-y-8"
            >
              <div className="space-y-4">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="inline-flex items-center space-x-2 bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-medium"
                >
                  <SparklesIcon className="h-4 w-4" />
                  <span>AI Destekli Öğrenci Analizi</span>
                </motion.div>
                <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 leading-tight">
                  Öğrenci Dikkatini{' '}
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    AI ile Ölçün
                  </span>
                </h1>
                <p className="text-xl text-gray-600 leading-relaxed">
                  Yapay zeka destekli dikkat analizi ile öğrencilerin katılımını gerçek zamanlı olarak 
                  takip edin ve öğrenme deneyimini optimize edin.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-4">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2"
                >
                  <PlayIcon className="h-5 w-5" />
                  <span>Demo İzle</span>
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="border-2 border-gray-300 text-gray-700 px-8 py-4 rounded-xl font-semibold hover:border-blue-600 hover:text-blue-600 transition-colors flex items-center justify-center space-x-2"
                >
                  <span>Daha Fazla Bilgi</span>
                  <ArrowRightIcon className="h-5 w-5" />
                </motion.button>
              </div>
            </motion.div>
            {}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="relative"
            >
              <div className="bg-white rounded-2xl shadow-2xl p-6 space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Gerçek Zamanlı Dashboard</h3>
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  {stats.map((stat, index) => (
                    <motion.div
                      key={stat.label}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      className="text-center p-4 rounded-xl bg-gray-50"
                    >
                      <div className={`text-2xl font-bold ${stat.color}`}>
                        {stat.value}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {stat.label}
                      </div>
                    </motion.div>
                  ))}
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Sınıf Katılımı</span>
                    <span className="text-emerald-600 font-medium">Yüksek</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: "87%" }}
                      transition={{ delay: 1, duration: 1.5 }}
                      className="bg-gradient-to-r from-emerald-500 to-blue-500 h-2 rounded-full"
                    ></motion.div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>
      {}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Canlı Demo
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Ders Lens'in güçlü özelliklerini keşfedin ve nasıl çalıştığını görün
            </p>
          </div>
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            {}
            <div className="border-b border-gray-200">
              <div className="flex space-x-0">
                {demoTabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveDemo(tab.id)}
                    className={`flex items-center space-x-2 px-6 py-4 font-medium transition-colors relative ${
                      activeDemo === tab.id
                        ? 'text-blue-600 bg-blue-50'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                  >
                    <tab.icon className="h-5 w-5" />
                    <span>{tab.name}</span>
                    {activeDemo === tab.id && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"
                      />
                    )}
                  </button>
                ))}
              </div>
            </div>
            {}
            <div className="p-8">
              <motion.div
                key={activeDemo}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="grid lg:grid-cols-2 gap-8 items-center"
              >
                <div className="space-y-4">
                  <h3 className="text-2xl font-bold text-gray-900">
                    {demoTabs.find(tab => tab.id === activeDemo)?.name}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {activeDemo === 'attention' && "Öğrencilerin dikkat seviyelerini gerçek zamanlı olarak takip edin. AI algoritmaları ile %95 doğrulukla dikkat dağınıklığını tespit eder."}
                    {activeDemo === 'emotion' && "Öğrencilerin yüz ifadelerinden duygu durumlarını analiz eder. Mutluluk, üzüntü, şaşkınlık gibi duyguları tanır."}
                    {activeDemo === 'engagement' && "Sınıf katılımını ölçer ve öğretmenlere detaylı raporlar sunar. Hangi öğrencinin ne kadar aktif olduğunu gösterir."}
                  </p>
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <div className="flex items-center space-x-1">
                      <CheckCircleIcon className="h-4 w-4 text-emerald-500" />
                      <span>Gerçek Zamanlı</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <CheckCircleIcon className="h-4 w-4 text-emerald-500" />
                      <span>%95 Doğruluk</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <CheckCircleIcon className="h-4 w-4 text-emerald-500" />
                      <span>AI Destekli</span>
                    </div>
                  </div>
                </div>
                <div className="relative">
                  <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-6 text-white">
                    <div className="flex items-center justify-between mb-4">
                      <div className="text-sm text-gray-400">Demo Arayüzü</div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Aktif Öğrenci:</span>
                        <span className="text-blue-400 font-mono">24/28</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Ortalama Dikkat:</span>
                        <span className="text-emerald-400 font-mono">87.3%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Duygu Durumu:</span>
                        <span className="text-purple-400 font-mono">Pozitif</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </section>
      {}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Güçlü Özellikler
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Ders Lens ile eğitim deneyiminizi bir üst seviyeye taşıyın
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="group"
              >
                <div className="bg-white p-8 rounded-2xl border border-gray-200 hover:shadow-xl transition-all duration-300 group-hover:border-blue-200">
                  <div className={`w-12 h-12 bg-gradient-to-r ${feature.gradient} rounded-xl flex items-center justify-center text-white mb-6`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Modern Teknoloji Yığını
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              En güncel AI ve web teknolojileri ile geliştirildi
            </p>
          </div>
          <div className="grid md:grid-cols-4 gap-6">
            {techStack.map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-xl text-center shadow-sm hover:shadow-lg transition-shadow"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mx-auto mb-4 flex items-center justify-center text-white font-bold text-lg">
                  {tech.name.charAt(0)}
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">{tech.name}</h3>
                <p className="text-sm text-gray-600">{tech.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-white">
              Eğitimde AI Devrimini Yaşayın
            </h2>
            <p className="text-xl text-blue-100 leading-relaxed">
              Ders Lens ile öğrenci dikkatini ve katılımını artırın. 
              Modern eğitimin geleceğini bugün deneyimleyin.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-colors flex items-center justify-center space-x-2"
              >
                <span>Ücretsiz Deneyin</span>
                <ArrowRightIcon className="h-5 w-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold hover:bg-white hover:text-blue-600 transition-colors flex items-center justify-center space-x-2"
              >
                <UserGroupIcon className="h-5 w-5" />
                <span>Demo Talep Et</span>
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>
      {}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <AcademicCapIcon className="h-5 w-5 text-white" />
                </div>
                <span className="text-xl font-bold">Ders Lens</span>
              </div>
              <p className="text-gray-400">
                AI destekli öğrenci analizi ile eğitimin geleceğini şekillendiriyoruz.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Ürün</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Özellikler</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Fiyatlandırma</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Demo</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Şirket</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Hakkımızda</a></li>
                <li><a href="#" className="hover:text-white transition-colors">İletişim</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Kariyer</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Destek</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Dokümantasyon</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Yardım Merkezi</a></li>
                <li><a href="#" className="hover:text-white transition-colors">İletişim</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
            <p>&copy; 2025 Ders Lens. Tüm hakları saklıdır.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default SimpleLandingPage;
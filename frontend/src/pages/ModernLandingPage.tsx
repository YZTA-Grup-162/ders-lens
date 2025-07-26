import { motion, useAnimation } from 'framer-motion';
import {
    ArrowRight,
    Brain,
    Clock,
    Cpu,
    Eye,
    Github,
    Globe,
    Heart,
    Mail,
    Monitor,
    Network,
    Play,
    Shield,
    Target,
    TrendingUp,
    Users,
    Zap
} from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ParticleBackground from '../components/ParticleBackground';
import ThemeToggle from '../components/ThemeToggle';
import { useTheme } from '../contexts/ThemeContext';
const ModernLandingPage: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const controls = useAnimation();
  const navigate = useNavigate();
  const { isDarkMode } = useTheme();
  useEffect(() => {
    setIsVisible(true);
    controls.start({ opacity: 1, y: 0 });
  }, [controls]);
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2
      }
    }
  };
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };
  return (
    <div className={`min-h-screen ${isDarkMode 
      ? 'bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900' 
      : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100'
    }`}>
      <ParticleBackground isDarkMode={isDarkMode} />
      {}
      <nav className={`fixed top-0 w-full ${isDarkMode 
        ? 'bg-slate-900/80 border-slate-700/60' 
        : 'bg-white/80 border-slate-200/60'
      } backdrop-blur-lg border-b z-50`}>
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <img 
                src="/derslens-logo.png" 
                alt="Ders Lens Logo" 
                className="h-10 w-auto"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Ders Lens
              </span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <a href="#ozellikler" className={`${isDarkMode ? 'text-slate-300 hover:text-blue-400' : 'text-slate-600 hover:text-blue-600'} transition-colors`}>
                Özellikler
              </a>
              <a href="#teknoloji" className={`${isDarkMode ? 'text-slate-300 hover:text-blue-400' : 'text-slate-600 hover:text-blue-600'} transition-colors`}>
                Teknoloji
              </a>
              <a href="#demo" className={`${isDarkMode ? 'text-slate-300 hover:text-blue-400' : 'text-slate-600 hover:text-blue-600'} transition-colors`}>
                Demo
              </a>
              <div className="flex space-x-2">
                <button 
                  onClick={() => navigate('/student-dashboard')}
                  className={`${isDarkMode ? 'text-slate-300 hover:text-blue-400' : 'text-slate-600 hover:text-blue-600'} transition-colors px-3 py-1 rounded`}
                >
                  Öğrenci
                </button>
                <button 
                  onClick={() => navigate('/teacher-dashboard')}
                  className={`${isDarkMode ? 'text-slate-300 hover:text-blue-400' : 'text-slate-600 hover:text-blue-600'} transition-colors px-3 py-1 rounded`}
                >
                  Eğitmen
                </button>
              </div>
              <ThemeToggle />
              <button className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-2 rounded-lg hover:shadow-lg transition-all duration-300">
                Başlayın
              </button>
            </div>
          </div>
        </div>
      </nav>
      {}
      <motion.section 
        className="relative pt-24 pb-16 px-6 lg:px-8"
        style={{ zIndex: 10 }}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div variants={itemVariants} className="text-center lg:text-left">
              <motion.div 
                className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-medium mb-6"
                whileHover={{ scale: 1.05 }}
              >
                <Zap className="w-4 h-4 mr-2" />
                AI Destekli Eğitim Analizi
              </motion.div>
              <motion.h1 
                className={`text-5xl lg:text-6xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'} leading-tight mb-6`}
                variants={itemVariants}
              >
                Öğrenci Odağını{' '}
                <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  AI ile Anlayın
                </span>
              </motion.h1>
              <motion.p 
                className={`text-xl ${isDarkMode ? 'text-slate-300' : 'text-slate-600'} mb-8 leading-relaxed`}
                variants={itemVariants}
              >
                Ders Lens, öğrenci dikkatini, duygusal durumunu, göz hareketlerini ve 
                katılımını gerçek zamanlı olarak yakalayıp görselleştirerek eğitmenlerin 
                öğrenmeyi optimize etmesine yardımcı olur.
              </motion.p>
              <motion.div 
                className="flex flex-col sm:flex-row gap-4"
                variants={itemVariants}
              >
                <motion.button 
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center group"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => navigate('/demo')}
                >
                  <Play className="w-5 h-5 mr-2 group-hover:translate-x-1 transition-transform" />
                  Demo
                </motion.button>
                <motion.button 
                  className="bg-gradient-to-r from-green-600 to-emerald-600 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center group"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => navigate('/teacher-dashboard')}
                >
                  <Users className="w-5 h-5 mr-2 group-hover:translate-x-1 transition-transform" />
                  Eğitmen Dashboard'ı
                </motion.button>
                <motion.button 
                  className="border-2 border-slate-300 text-slate-700 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-slate-50 transition-all duration-300 flex items-center justify-center group"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => navigate('/student-dashboard')}
                >
                  Öğrenci Dashboard'ı
                  <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                </motion.button>
              </motion.div>
            </motion.div>
            {}
            <motion.div 
              className="relative"
              variants={itemVariants}
            >
              <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-3xl p-8 shadow-2xl">
                <div className="bg-white rounded-2xl p-6">
                  {}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-slate-800">Canlı Sınıf Analizi</h3>
                      <div className="flex items-center text-green-600">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                        Aktif
                      </div>
                    </div>
                    {}
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { name: "Başak Avcı", attention: 94 },
                        { name: "Öğrenci 2", attention: 87 },
                        { name: "Öğrenci 3", attention: 89 },
                        { name: "Öğrenci 4", attention: 91 }
                      ].map((student, index) => (
                        <motion.div 
                          key={index}
                          className="bg-slate-50 rounded-lg p-3 border"
                          whileHover={{ scale: 1.02 }}
                        >
                          <div className="flex items-center mb-2">
                            <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full mr-2"></div>
                            <span className="text-sm font-medium">{student.name}</span>
                          </div>
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span>Dikkat</span>
                              <span className="text-green-600">%{student.attention}</span>
                            </div>
                            <div className="w-full bg-slate-200 rounded-full h-1.5">
                              <div 
                                className="bg-gradient-to-r from-green-400 to-blue-500 h-1.5 rounded-full"
                                style={{ width: `${student.attention}%` }}
                              ></div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                    {}
                    <div className="grid grid-cols-3 gap-2 mt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">89%</div>
                        <div className="text-xs text-slate-600">Ortalama Dikkat</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">92%</div>
                        <div className="text-xs text-slate-600">Katılım</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-indigo-600">4.2</div>
                        <div className="text-xs text-slate-600">Duygu Skoru</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              {}
              <motion.div 
                className="absolute -top-4 -right-4 bg-white rounded-full p-3 shadow-lg"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Brain className="w-6 h-6 text-blue-600" />
              </motion.div>
              <motion.div 
                className="absolute -bottom-4 -left-4 bg-white rounded-full p-3 shadow-lg"
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 2.5, repeat: Infinity }}
              >
                <Eye className="w-6 h-6 text-indigo-600" />
              </motion.div>
            </motion.div>
          </div>
        </div>
      </motion.section>
      {}
      <section id="ozellikler" className={`relative py-20 ${isDarkMode ? 'bg-slate-800/50' : 'bg-white'}`} style={{ zIndex: 10 }}>
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className={`text-4xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'} mb-4`}>
              Özellikler
            </h2>
            <p className={`text-xl ${isDarkMode ? 'text-slate-300' : 'text-slate-600'} max-w-3xl mx-auto`}>
              AI destekli görsel analiz ile eğitim deneyimini yeni boyutlara taşıyın
            </p>
          </motion.div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: Target,
                title: "Dikkat Takibi",
                description: "Yüz yönelimi ve varlık tespiti ile dikkat seviyesini analiz eder",
                color: "from-red-500 to-pink-500"
              },
              {
                icon: TrendingUp,
                title: "Katılım Ölçümü",
                description: "Hareket, ekran etkileşimi ve duruş bazlı katılım değerlendirmesi",
                color: "from-blue-500 to-indigo-500"
              },
              {
                icon: Heart,
                title: "Duygu Tanıma",
                description: "Yüz mikro ifadeleri ile duygu sınıflandırması (karışıklık, sıkılma, heyecan)",
                color: "from-green-500 to-emerald-500"
              },
              {
                icon: Eye,
                title: "Göz Haritalama",
                description: "Öğrencinin ekranın hangi bölümüne odaklandığını tespit eden göz takibi",
                color: "from-purple-500 to-violet-500"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className={`${isDarkMode ? 'bg-slate-700/50 border-slate-600' : 'bg-white border-slate-100'} rounded-2xl p-8 shadow-lg border hover:shadow-xl transition-all duration-300`}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ y: -5 }}
              >
                <div className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center mb-6`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-slate-900'} mb-3`}>{feature.title}</h3>
                <p className={`${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {}
      <section id="teknoloji" className="py-20 bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-slate-900 mb-4">
              Gelişmiş Teknoloji Yığını
            </h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              Modern web teknolojileri ve AI altyapısı ile güçlendirilmiş
            </p>
          </motion.div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Globe,
                title: "React Frontend",
                description: "Modern, responsive kullanıcı arayüzü",
                tech: "TypeScript, Tailwind CSS"
              },
              {
                icon: Network,
                title: "Node.js Backend",
                description: "Ölçeklenebilir API ve gerçek zamanlı iletişim",
                tech: "Express, WebSocket"
              },
              {
                icon: Brain,
                title: "AI Modülleri",
                description: "Gelişmiş makine öğrenmesi algoritmaları",
                tech: "Computer Vision, Deep Learning"
              },
              {
                icon: Monitor,
                title: "Docker Konteynerization",
                description: "Kolay dağıtım ve ölçeklendirme",
                tech: "Docker, Docker Compose"
              },
              {
                icon: Shield,
                title: "GDPR Uyumlu",
                description: "Veri gizliliği ve güvenlik öncelikli",
                tech: "Şifreli veri işleme"
              },
              {
                icon: Cpu,
                title: "Gerçek Zamanlı İşleme",
                description: "Düşük gecikme ile canlı analiz",
                tech: "WebRTC, GPU İvmesi"
              }
            ].map((tech, index) => (
              <motion.div
                key={index}
                className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.02 }}
              >
                <tech.icon className="w-8 h-8 text-blue-600 mb-4" />
                <h3 className="text-lg font-bold text-slate-900 mb-2">{tech.title}</h3>
                <p className="text-slate-600 mb-3">{tech.description}</p>
                <span className="text-sm text-blue-600 font-medium">{tech.tech}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {}
      <section id="demo" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-slate-900 mb-4">
              Canlı Dashboard Önizlemesi
            </h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              Gerçek zamanlı sınıf analitiği ve öğrenci metrikleri
            </p>
          </motion.div>
          <motion.div 
            className="bg-gradient-to-br from-slate-900 to-blue-900 rounded-3xl p-8 shadow-2xl"
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <div className="bg-white rounded-2xl p-6">
              {}
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h3 className="text-2xl font-bold text-slate-900">Sınıf Analiz Dashboard'u</h3>
                  <p className="text-slate-600">Matematik 101 - 24 Öğrenci Aktif</p>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center text-green-600">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                    Canlı
                  </div>
                  <Clock className="w-5 h-5 text-slate-400" />
                  <span className="text-slate-600">14:32</span>
                </div>
              </div>
              {}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                {[
                  { label: "Ortalama Dikkat", value: "87%", color: "text-blue-600", bg: "bg-blue-50" },
                  { label: "Sınıf Katılımı", value: "92%", color: "text-green-600", bg: "bg-green-50" },
                  { label: "Duygu Skoru", value: "4.2/5", color: "text-purple-600", bg: "bg-purple-50" },
                  { label: "Aktif Öğrenci", value: "24/26", color: "text-orange-600", bg: "bg-orange-50" }
                ].map((metric, index) => (
                  <motion.div
                    key={index}
                    className={`${metric.bg} rounded-xl p-6 text-center`}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <div className={`text-3xl font-bold ${metric.color} mb-2`}>{metric.value}</div>
                    <div className="text-slate-600 text-sm">{metric.label}</div>
                  </motion.div>
                ))}
              </div>
              {}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {Array.from({ length: 12 }, (_, i) => (
                  <motion.div
                    key={i}
                    className="bg-slate-50 rounded-lg p-3 border border-slate-200"
                    initial={{ opacity: 0, scale: 0.8 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.05 }}
                    viewport={{ once: true }}
                    whileHover={{ scale: 1.05 }}
                  >
                    <div className="flex items-center mb-2">
                      <div 
                        className={`w-6 h-6 rounded-full mr-2 ${
                          i % 3 === 0 ? 'bg-green-400' : 
                          i % 3 === 1 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                      ></div>
                      <span className="text-xs font-medium">Öğr. {i + 1}</span>
                    </div>
                    <div className="text-xs text-slate-600">
                      %{Math.floor(Math.random() * 30) + 70} dikkat
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-1 mt-1">
                      <div 
                        className={`h-1 rounded-full ${
                          i % 3 === 0 ? 'bg-green-400' : 
                          i % 3 === 1 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                        style={{ width: `${Math.floor(Math.random() * 30) + 70}%` }}
                      ></div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      {}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-indigo-600">
        <div className="max-w-4xl mx-auto text-center px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-6">
              Sınıfınızı Hiç Olmadığı Gibi Anlamaya Başlayın
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              AI destekli analiz ile öğrenci katılımını artırın ve eğitim kalitesini yükseltin
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button 
                className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate('/teacher-dashboard')}
              >
                Eğitmen Dashboard'ını Deneyin
              </motion.button>
              <motion.button 
                className="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-white hover:text-blue-600 transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate('/student-dashboard')}
              >
                Öğrenci Görünümü
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>
      {}
      <footer className="bg-slate-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <div className="flex items-center space-x-3 mb-6">
                <img 
                  src="/derslens-logo.png" 
                  alt="Ders Lens Logo" 
                  className="h-8 w-auto"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
                <span className="text-2xl font-bold">Ders Lens</span>
              </div>
              <p className="text-slate-400 mb-6 max-w-md">
                AI destekli öğrenci analizi ile eğitim deneyimini yeni boyutlara taşıyan 
                gelişmiş platform. Öğrenci dikkatini, katılımını ve duygusal durumunu 
                gerçek zamanlı olarak analiz eder.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-slate-400 hover:text-white transition-colors">
                  <Github className="w-6 h-6" />
                </a>
                <a href="#" className="text-slate-400 hover:text-white transition-colors">
                  <Mail className="w-6 h-6" />
                </a>
                <a href="#" className="text-slate-400 hover:text-white transition-colors">
                  <Globe className="w-6 h-6" />
                </a>
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Ürün</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Özellikler</a></li>
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Fiyatlandırma</a></li>
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Demo</a></li>
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Dokümantasyon</a></li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Destek</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Gizlilik Politikası</a></li>
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Kullanım Şartları</a></li>
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">İletişim</a></li>
                <li><a href="#" className="text-slate-400 hover:text-white transition-colors">GitHub</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-slate-800 mt-12 pt-8 text-center">
            <p className="text-slate-400">
              &copy; 2025 Ders Lens. Tüm hakları saklıdır. 
              <span className="ml-2 text-blue-400">AI Destekli • GDPR Uyumlu • Docker ile Güçlendirilmiş</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default ModernLandingPage;
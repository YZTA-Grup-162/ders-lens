import { motion } from 'framer-motion';
import {
    ArrowRight,
    BarChart3,
    Brain,
    Camera,
    Eye,
    Moon,
    Play,
    Shield,
    Sun,
    Target,
    Users,
    Zap
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
const HomePage = () => {
  const navigate = useNavigate();
  const { isDarkMode, toggleTheme } = useTheme();
  const features = [
    {
      icon: Eye,
      title: 'Dikkat Takibi',
      description: 'Yüz yönü ve varlık tespiti ile öğrenci dikkatini gerçek zamanlı takip edin.',
      gradient: 'from-blue-500 to-purple-600'
    },
    {
      icon: Users,
      title: 'Katılım Analizi',
      description: 'Hareket, ekran etkileşimi ve duruş analizi ile katılım seviyesini ölçün.',
      gradient: 'from-green-500 to-teal-600'
    },
    {
      icon: Brain,
      title: 'Duygu Tanıma',
      description: 'Mikro ifadelerle duygu sınıflandırması (sıkılma, kafa karışıklığı vb.).',
      gradient: 'from-orange-500 to-red-600'
    },
    {
      icon: Target,
      title: 'Bakış Haritalama',
      description: 'Göz izleme ve ekran odak tespiti ile dikkat haritaları oluşturun.',
      gradient: 'from-pink-500 to-violet-600'
    }
  ];
  const stats = [
    { label: 'Dikkat Doğruluğu', value: '95%' },
    { label: 'Duygu Tanıma', value: '92%' },
    { label: 'Gerçek Zamanlı', value: '<100ms' },
    { label: 'Öğrenci Kapasitesi', value: '50+' }
  ];
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {}
      <header className="relative z-50 px-6 py-4">
        <nav className="flex items-center justify-between max-w-7xl mx-auto">
          <motion.div 
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <img 
              src="/derslens-logo.png" 
              alt="Ders Lens Logo" 
              className="w-10 h-10 rounded-xl"
            />
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Ders Lens
            </h1>
          </motion.div>
          <motion.div 
            className="flex items-center space-x-4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-white/20 dark:bg-gray-800/20 backdrop-blur-sm border border-white/20 dark:border-gray-700/20 hover:bg-white/30 dark:hover:bg-gray-700/30 transition-all duration-300"
            >
              {isDarkMode ? (
                <Sun className="w-5 h-5 text-yellow-500" />
              ) : (
                <Moon className="w-5 h-5 text-gray-600" />
              )}
            </button>
            <button
              onClick={() => navigate('/demo')}
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transform hover:scale-105 transition-all duration-300"
            >
              Demo'ya Başla
            </button>
          </motion.div>
        </nav>
      </header>
      {}
      <section className="relative px-6 py-20">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-5xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              AI Destekli Öğrenci
              <br />
              Dikkat ve Katılım Analizi
            </h2>
            <p className="text-xl lg:text-2xl text-gray-600 dark:text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed">
              Ders Lens, öğrencilerin dikkatini, duygularını, bakış yönünü ve derse katılımını 
              gerçek zamanlı olarak analiz eder.
            </p>
          </motion.div>
          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <button
              onClick={() => navigate('/demo')}
              className="group flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
            >
              <Play className="w-5 h-5 mr-2" />
              Demo'ya Başla
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </button>
            <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
              <Shield className="w-4 h-4 mr-2" />
              GDPR Uyumlu • Gizlilik Odaklı
            </div>
          </motion.div>
          {}
          <motion.div
            className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-20"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            {stats.map((stat, index) => (
              <div
                key={stat.label}
                className="p-6 bg-white/20 dark:bg-gray-800/20 backdrop-blur-sm rounded-2xl border border-white/20 dark:border-gray-700/20"
              >
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600 dark:text-gray-300 text-sm">
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>
        </div>
        {}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-4 -left-4 w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
          <div className="absolute -top-4 -right-4 w-96 h-96 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="absolute bottom-8 left-20 w-96 h-96 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>
      </section>
      {}
      <section className="px-6 py-20 bg-white/30 dark:bg-gray-800/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-4xl lg:text-5xl font-bold mb-6 text-gray-900 dark:text-white">
              Gelişmiş AI Özellikleri
            </h3>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Son teknoloji yapay zeka modelleri ile öğrenci davranışlarını anında analiz edin
            </p>
          </motion.div>
          <div className="grid lg:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                className="group p-8 bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl border border-white/20 dark:border-gray-700/20 hover:shadow-2xl transition-all duration-500"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ y: -5 }}
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.gradient} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h4 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  {feature.title}
                </h4>
                <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {}
      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-4xl lg:text-5xl font-bold mb-6 text-gray-900 dark:text-white">
              Gerçek Zamanlı Dashboard
            </h3>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Öğrenci davranışlarını anında izleyin ve analiz edin
            </p>
          </motion.div>
          <motion.div
            className="relative bg-gradient-to-r from-blue-600/10 to-purple-600/10 rounded-3xl p-8 border border-blue-200/20 dark:border-blue-700/20"
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <div className="grid lg:grid-cols-3 gap-8">
              {}
              <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  <div className="ml-3">
                    <h4 className="font-semibold text-gray-900 dark:text-white">Öğrenci #1</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300">Aktif</p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-300">Dikkat</span>
                    <span className="text-sm font-semibold text-green-600">92%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-300">Katılım</span>
                    <span className="text-sm font-semibold text-blue-600">88%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-300">Duygu</span>
                    <span className="text-sm font-semibold text-orange-600">Odaklı</span>
                  </div>
                </div>
              </div>
              {}
              <div className="lg:col-span-2 bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Sınıf Analizi</h4>
                  <BarChart3 className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">85%</div>
                    <div className="text-xs text-gray-600 dark:text-gray-300">Ortalama Dikkat</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">78%</div>
                    <div className="text-xs text-gray-600 dark:text-gray-300">Katılım Oranı</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">12</div>
                    <div className="text-xs text-gray-600 dark:text-gray-300">Aktif Öğrenci</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      {}
      <section className="px-6 py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              Sınıfınızı Gerçek Zamanlı
              <br />
              Anlamaya Başlayın
            </h3>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Modern AI teknolojileri ile öğrenci başarısını artırın ve eğitim kalitesini yükseltin.
            </p>
            <motion.button
              onClick={() => navigate('/demo')}
              className="group inline-flex items-center px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Zap className="w-5 h-5 mr-2" />
              Demo'ya Başla
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </motion.div>
        </div>
      </section>
      {}
      <footer className="px-6 py-12 bg-gray-900 dark:bg-black">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col lg:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-6 lg:mb-0">
              <img 
                src="/derslens-logo.png" 
                alt="Ders Lens Logo" 
                className="w-10 h-10 rounded-xl"
              />
              <h1 className="text-2xl font-bold text-white">Ders Lens</h1>
            </div>
            <div className="flex space-x-8 text-gray-400">
              <a href="#" className="hover:text-white transition-colors">Gizlilik Politikası</a>
              <a href="#" className="hover:text-white transition-colors">GitHub</a>
              <a href="#" className="hover:text-white transition-colors">İletişim</a>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400">
            <p>&copy; 2025 Ders Lens. Tüm hakları saklıdır. • Powered by AI</p>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default HomePage;
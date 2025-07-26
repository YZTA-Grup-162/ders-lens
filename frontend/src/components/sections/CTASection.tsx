import { motion, useInView } from 'framer-motion';
import { ArrowRight, BookOpen, CheckCircle, Download, ExternalLink, Play } from 'lucide-react';
import React, { useRef, useState } from 'react';
import { useAI } from '../../stores/aiStore';
export function CTASection() {
  const { state, actions } = useAI();
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const sectionRef = useRef(null);
  const isInView = useInView(sectionRef, { once: true });
  const handleStartDemo = async () => {
    if (!state.isAnalyzing) {
      await actions.startAnalysis();
    }
  };
  const handleEmailSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      setIsSubmitted(true);
      setTimeout(() => {
        setIsSubmitted(false);
        setEmail('');
      }, 3000);
    }
  };
  return (
    <section ref={sectionRef} className="py-20 bg-gradient-to-b from-gray-900 to-black relative overflow-hidden">
      {}
      <div className="absolute inset-0">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-cyan-600/10"
          animate={{
            backgroundPosition: ["0% 0%", "100% 100%"],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
        {}
        <div className="absolute inset-0">
          {Array.from({ length: 50 }).map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-blue-400/30 rounded-full"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, -100, 0],
                opacity: [0, 1, 0],
              }}
              transition={{
                duration: Math.random() * 10 + 10,
                repeat: Infinity,
                delay: Math.random() * 10,
              }}
            />
          ))}
        </div>
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
            className="text-4xl md:text-6xl font-bold text-white mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <span className="text-white">Eğitiminizde</span>
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              AI Devrimini
            </span>
            <br />
            <span className="text-white">Başlatın</span>
          </motion.h2>
          <motion.p 
            className="text-xl text-gray-300 max-w-3xl mx-auto mb-12"
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Ders Lens ile öğrencilerinizin dikkat, katılım ve duygusal durumlarını 
            gerçek zamanlı olarak analiz edin. Geleceğin eğitim teknolojisini bugün deneyimleyin.
          </motion.p>
          {}
          <motion.div 
            className="flex flex-col sm:flex-row gap-6 justify-center mb-12"
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <motion.button
              onClick={handleStartDemo}
              className="group bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-8 py-4 rounded-2xl font-bold text-lg shadow-2xl hover:shadow-blue-500/25 transition-all duration-300 flex items-center justify-center"
              whileHover={{ 
                scale: 1.05, 
                boxShadow: "0 25px 50px -12px rgba(59, 130, 246, 0.5)" 
              }}
              whileTap={{ scale: 0.95 }}
            >
              <Play className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform" />
              Canlı Demo'yu Başlat
              <ArrowRight className="w-6 h-6 ml-3 group-hover:translate-x-1 transition-transform" />
            </motion.button>
            <motion.button
              className="group bg-white/10 backdrop-blur-md text-white px-8 py-4 rounded-2xl font-bold text-lg border border-white/20 hover:bg-white/20 transition-all duration-300 flex items-center justify-center"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Download className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform" />
              Sistem İndir
            </motion.button>
          </motion.div>
          {}
          {state.isAnalyzing && (
            <motion.div
              className="inline-flex items-center bg-green-500/20 border border-green-500/30 rounded-full px-6 py-3 mb-8"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <motion.div
                className="w-3 h-3 bg-green-400 rounded-full mr-3"
                animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="text-green-300 font-semibold">
                Demo aktif - AI analizi çalışıyor
              </span>
            </motion.div>
          )}
        </motion.div>
        {}
        <motion.div
          className="grid md:grid-cols-3 gap-8 mb-16"
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          {[
            {
              title: 'Anında Kurulum',
              description: 'Docker ile tek komutla tüm sistemi kurun',
              icon: <CheckCircle className="w-8 h-8" />,
              features: ['Otomatik dependency yönetimi', 'Cross-platform uyumluluk', '5 dakikada hazır']
            },
            {
              title: 'Kapsamlı Dokümantasyon',
              description: 'Detaylı kurulum ve kullanım kılavuzu',
              icon: <BookOpen className="w-8 h-8" />,
              features: ['Adım adım kurulum', 'API referansları', 'Video eğitimler']
            },
            {
              title: 'Açık Kaynak',
              description: 'Tamamen ücretsiz ve özelleştirilebilir',
              icon: <ExternalLink className="w-8 h-8" />,
              features: ['MIT lisansı', 'GitHub deposu', 'Topluluk desteği']
            }
          ].map((feature, index) => (
            <motion.div
              key={feature.title}
              className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 text-center"
              initial={{ opacity: 0, y: 30 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 1 + index * 0.1 }}
              whileHover={{ y: -5, backgroundColor: "rgba(255,255,255,0.1)" }}
            >
              <motion.div
                className="w-16 h-16 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-2xl p-3 mx-auto mb-4 border border-blue-500/30"
                whileHover={{ rotate: 5, scale: 1.1 }}
              >
                <div className="text-blue-400 w-full h-full flex items-center justify-center">
                  {feature.icon}
                </div>
              </motion.div>
              <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
              <p className="text-gray-300 text-sm mb-4">{feature.description}</p>
              <div className="space-y-2">
                {feature.features.map((item, i) => (
                  <div key={i} className="flex items-center text-sm text-gray-400">
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full mr-3" />
                    {item}
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </motion.div>
        {}
        <motion.div
          className="bg-white/5 backdrop-blur-md rounded-3xl p-8 border border-white/10 text-center"
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 1.3 }}
        >
          <h3 className="text-2xl font-bold text-white mb-4">
            Güncellemelerden Haberdar Olun
          </h3>
          <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
            Yeni özellikler, performans iyileştirmeleri ve eğitim teknolojilerindeki 
            gelişmeler hakkında bilgi almak için e-posta listemize katılın.
          </p>
          <motion.form
            onSubmit={handleEmailSubmit}
            className="max-w-md mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 1.5 }}
          >
            <div className="flex gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="E-posta adresiniz"
                className="flex-1 bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-400 focus:bg-white/15 transition-all"
                required
              />
              <motion.button
                type="submit"
                className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-6 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                disabled={isSubmitted}
              >
                {isSubmitted ? 'Teşekkürler!' : 'Katıl'}
              </motion.button>
            </div>
            {isSubmitted && (
              <motion.div
                className="mt-4 text-green-400 font-medium"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                ✓ E-posta adresiniz kaydedildi!
              </motion.div>
            )}
          </motion.form>
        </motion.div>
        {}
        <motion.div
          className="text-center mt-16"
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 1.7 }}
        >
          <p className="text-gray-400 text-lg italic">
            "Bu teknoloji gerçekten geleceğin eğitim sistemi!"
          </p>
          <p className="text-gray-500 text-sm mt-2">
            - Eğitim Teknolojileri Uzmanları
          </p>
        </motion.div>
      </div>
    </section>
  );
}
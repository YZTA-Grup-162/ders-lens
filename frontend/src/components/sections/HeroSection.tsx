import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { useAI } from '../../stores/aiStore';
export function HeroSection() {
  const { actions } = useAI();
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end start"]
  });
  const y = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  return (
    <motion.section 
      ref={sectionRef}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
      style={{ y, opacity }}
    >
      {}
      <div className="absolute inset-0">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-cyan-600/20"
          animate={{
            backgroundPosition: ["0% 50%", "100% 50%"],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
        {}
        <svg className="absolute inset-0 w-full h-full opacity-20" viewBox="0 0 1000 1000">
          <defs>
            <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.6"/>
              <stop offset="100%" stopColor="#06B6D4" stopOpacity="0.6"/>
            </linearGradient>
          </defs>
          {}
          {Array.from({ length: 20 }).map((_, i) => (
            <motion.circle
              key={i}
              cx={100 + (i % 5) * 200}
              cy={200 + Math.floor(i / 5) * 150}
              r="4"
              fill="url(#neuralGradient)"
              animate={{
                opacity: [0.3, 1, 0.3],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.1,
              }}
            />
          ))}
          {}
          {Array.from({ length: 15 }).map((_, i) => (
            <motion.line
              key={i}
              x1={100 + (i % 4) * 200}
              y1={200 + Math.floor(i / 4) * 150}
              x2={300 + (i % 4) * 200}
              y2={200 + Math.floor(i / 4) * 150}
              stroke="url(#neuralGradient)"
              strokeWidth="1"
              animate={{
                opacity: [0.2, 0.8, 0.2],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                delay: i * 0.2,
              }}
            />
          ))}
        </svg>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
        >
          {}
          <motion.h1 
            className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
          >
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              AI Destekli
            </span>
            <br />
            <span className="text-white">
              Öğrenci Dikkat ve
            </span>
            <br />
            <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Katılım Analizi
            </span>
          </motion.h1>
          {}
          <motion.p 
            className="text-xl md:text-2xl text-blue-100 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.4 }}
          >
            Ders Lens, öğrencilerin dikkat seviyesini, duygusal durumunu, bakış yönünü ve derse katılımını 
            <span className="text-cyan-300 font-semibold"> gerçek zamanlı olarak analiz ederek</span> eğitimcilere 
            öğrenme sürecini optimize etme imkanı sunar.
          </motion.p>
          {}
          <motion.div 
            className="flex flex-wrap justify-center gap-4 mb-10"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.6 }}
          >
            {[
              { icon: '🧠', label: '8 Duygu Tanıma' },
              { icon: '👁️', label: 'Dikkat Takibi' },
              { icon: '📊', label: 'Katılım Analizi' },
              { icon: '🎯', label: 'Bakış Haritalama' }
            ].map((feature, index) => (
              <motion.div
                key={feature.label}
                className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20"
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255,255,255,0.15)" }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.8 + index * 0.1 }}
              >
                <span className="text-2xl mr-2">{feature.icon}</span>
                <span className="text-white font-medium">{feature.label}</span>
              </motion.div>
            ))}
          </motion.div>
          {}
          <motion.div 
            className="flex flex-col sm:flex-row gap-4 justify-center"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1 }}
          >
            <motion.button
              onClick={actions.startAnalysis}
              className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-2xl hover:shadow-blue-500/25 transition-all duration-300"
              whileHover={{ 
                scale: 1.05, 
                boxShadow: "0 25px 50px -12px rgba(59, 130, 246, 0.5)" 
              }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center justify-center">
                <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd"/>
                </svg>
                Canlı Demo'yu İzle
              </span>
            </motion.button>
            <motion.button
              className="bg-white/10 backdrop-blur-md text-white px-8 py-4 rounded-xl font-bold text-lg border border-white/20 hover:bg-white/20 transition-all duration-300"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center justify-center">
                <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Teknik Dokümantasyon
              </span>
            </motion.button>
          </motion.div>
          {}
          <motion.div 
            className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1.2 }}
          >
            {[
              { value: '94.2%', label: 'Duygu Tanıma Doğruluğu' },
              { value: '<23ms', label: 'İşlem Hızı' },
              { value: '30 FPS', label: 'Gerçek Zamanlı Analiz' },
              { value: '8 Duygu', label: 'FER2013+ Modeli' }
            ].map((metric, index) => (
              <motion.div
                key={metric.label}
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 1.4 + index * 0.1 }}
              >
                <div className="text-2xl md:text-3xl font-bold text-cyan-400 mb-2">
                  {metric.value}
                </div>
                <div className="text-sm text-blue-200">
                  {metric.label}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </div>
      {}
      <motion.div
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <div className="w-6 h-10 border-2 border-white/30 rounded-full p-1">
          <motion.div
            className="w-1 h-3 bg-white/60 rounded-full mx-auto"
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </motion.section>
  );
}
import { motion } from 'framer-motion';
import { Camera, CheckCircle, Play, Zap } from 'lucide-react';
interface CallToActionProps {
  onStartCamera: () => void;
  className?: string;
}
export function CameraTestCallToAction({ onStartCamera, className = '' }: CallToActionProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`bg-gradient-to-br from-blue-600/20 via-purple-600/20 to-pink-600/20 backdrop-blur-sm rounded-2xl border border-blue-500/30 p-8 text-center ${className}`}
    >
      {}
      <motion.div
        animate={{ 
          rotate: [0, 5, -5, 0],
          scale: [1, 1.1, 1]
        }}
        transition={{ 
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6"
      >
        <Camera className="w-10 h-10 text-white" />
      </motion.div>
      {}
      <motion.h2
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-3xl font-bold text-white mb-4"
      >
        Canlı Kamera Analizi
      </motion.h2>
      {}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="text-lg text-blue-200 mb-6 max-w-2xl mx-auto leading-relaxed"
      >
        Gerçek zamanlı analiz yapın. Duygu analizi, dikkat takibi ve katılım ölçümü
        özelliklerini kullanarak veri toplayın.
      </motion.p>
      {}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"
      >
        {[
          {
            title: 'Duygu Analizi',
            description: 'FER2013+ modeli ile 7 farklı duygu tespiti',
            icon: 'emotion'
          },
          {
            title: 'Dikkat Takibi',
            description: 'DAISEE modeli ile gerçek zamanlı dikkat ölçümü',
            icon: 'eye'
          },
          {
            title: 'Katılım Analizi',
            description: 'Mendeley modeli ile öğrenme katılımı tespiti',
            icon: 'chart'
          }
        ].map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 + index * 0.1 }}
            className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10"
          >
            <div className="text-2xl mb-2">{feature.icon}</div>
            <h3 className="font-semibold text-white mb-1">{feature.title}</h3>
            <p className="text-sm text-blue-200">{feature.description}</p>
          </motion.div>
        ))}
      </motion.div>
      {}
      <motion.button
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        whileHover={{ 
          scale: 1.05,
          boxShadow: "0 10px 25px rgba(59, 130, 246, 0.3)"
        }}
        whileTap={{ scale: 0.95 }}
        onClick={onStartCamera}
        className="group relative bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-10 py-4 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg"
      >
        <span className="flex items-center gap-3">
          <Play className="w-6 h-6 group-hover:scale-110 transition-transform" />
          Hemen Test Et
          <Zap className="w-5 h-5 group-hover:rotate-12 transition-transform" />
        </span>
        {}
        <motion.div
          animate={{
            rotate: 360
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 opacity-20 blur-sm"
        />
      </motion.button>
      {}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="mt-6 flex items-center justify-center gap-6 text-sm text-blue-200"
      >
        <div className="flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-green-400" />
          <span>Gerçek zamanlı</span>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-green-400" />
          <span>Yüksek doğruluk</span>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-green-400" />
          <span>Kolay kullanım</span>
        </div>
      </motion.div>
      {}
      <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            animate={{
              y: [0, -100, 0],
              x: [0, 50, 0],
              opacity: [0, 1, 0]
            }}
            transition={{
              duration: 4 + i,
              repeat: Infinity,
              delay: i * 0.8
            }}
            className="absolute w-2 h-2 bg-blue-400/30 rounded-full"
            style={{
              left: `${10 + i * 15}%`,
              bottom: '-10px'
            }}
          />
        ))}
      </div>
    </motion.div>
  );
}
import { AnimatePresence, motion } from 'framer-motion';
import { BarChart3, Brain, Camera, Eye, Globe, Shield, Target, Zap } from 'lucide-react';
import React, { useState } from 'react';
interface Feature {
  id: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  details: string;
  metrics: { label: string; value: string }[];
  color: string;
}
const features: Feature[] = [
  {
    id: 'attention',
    icon: <Eye className="w-8 h-8" />,
    title: 'Dikkat Takibi',
    description: 'Öğrencilerin derse odaklanma seviyesini gerçek zamanlı olarak izler ve analiz eder.',
    details: 'Gelişmiş bilgisayar görü algoritmaları kullanarak göz hareketleri, göz kapağı pozisyonu ve baş yönelimi analiz edilir. DAISEE veri seti ile eğitilmiş modeller sayesinde %92.5 doğrulukla dikkat seviyesi tespit edilir.',
    metrics: [
      { label: 'Doğruluk Oranı', value: '%92.5' },
      { label: 'İşlem Hızı', value: '30 FPS' },
      { label: 'Latans', value: '<16ms' }
    ],
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'emotion',
    icon: <Brain className="w-8 h-8" />,
    title: 'Duygu Tanıma',
    description: 'FER2013+ modeli ile 8 farklı duygu durumunu yüksek doğrulukla tespit eder.',
    details: 'Nötr, mutluluk, şaşırma, üzüntü, öfke, iğrenme, korku ve küçümseme olmak üzere 8 temel duyguyu gerçek zamanlı olarak tanır. Derin öğrenme modelleri ile yüz ifadeleri analiz edilir.',
    metrics: [
      { label: 'Duygu Sayısı', value: '8 Farklı' },
      { label: 'Model Doğruluğu', value: '%94.2' },
      { label: 'Veri Seti', value: 'FER2013+' }
    ],
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'engagement',
    icon: <BarChart3 className="w-8 h-8" />,
    title: 'Katılım Analizi',
    description: 'Sınıf genelinde katılım seviyelerini ölçer ve detaylı raporlar sunar.',
    details: 'Birden fazla faktörü birleştirerek genel katılım skoru hesaplar. Dikkat, duygu durumu ve davranış analizini entegre ederek kapsamlı bir katılım profili oluşturur.',
    metrics: [
      { label: 'Faktör Sayısı', value: '12 Parametre' },
      { label: 'Güncelleme', value: 'Anlık' },
      { label: 'Öğrenci Takibi', value: 'Sınırsız' }
    ],
    color: 'from-green-500 to-teal-500'
  },
  {
    id: 'gaze',
    icon: <Target className="w-8 h-8" />,
    title: 'Bakış Haritalama',
    description: 'Öğrencilerin hangi alanlara baktığını tespit ederek odak noktalarını belirler.',
    details: 'MPIIGaze veri seti ile eğitilmiş modellerle bakış yönü ve odak noktaları belirlenir. Sınıf içindeki ilgi alanları haritalanarak eğitim materyallerinin etkinliği değerlendirilir.',
    metrics: [
      { label: 'Hassasiyet', value: '±2.5°' },
      { label: 'Kapsama Alanı', value: '360°' },
      { label: 'Kalibrasyon', value: 'Otomatik' }
    ],
    color: 'from-orange-500 to-red-500'
  }
];
export function FeaturesSection() {
  const [activeFeature, setActiveFeature] = useState<string | null>(null);
  return (
    <section className="py-20 bg-gradient-to-b from-gray-900 to-black relative overflow-hidden">
      {}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20" />
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#grid)" />
        </svg>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <motion.h2 
            className="text-4xl md:text-5xl font-bold text-white mb-6"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              AI Destekli
            </span>
            <br />
            <span className="text-white">Analiz Yetenekleri</span>
          </motion.h2>
          <motion.p 
            className="text-xl text-gray-300 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Ders Lens'in gelişmiş yapay zeka modelleri ile eğitim ortamınızı 
            daha verimli ve etkileşimli hale getirin.
          </motion.p>
        </motion.div>
        {}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.id}
              className="relative group cursor-pointer"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              onClick={() => setActiveFeature(activeFeature === feature.id ? null : feature.id)}
            >
              {}
              <motion.div
                className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 h-full transition-all duration-300 hover:bg-white/10 hover:border-white/20"
                whileHover={{ y: -5, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {}
                <motion.div
                  className={`w-16 h-16 rounded-xl bg-gradient-to-r ${feature.color} p-3 mb-6 shadow-lg`}
                  whileHover={{ rotate: 5, scale: 1.1 }}
                >
                  <div className="text-white w-full h-full flex items-center justify-center">
                    {feature.icon}
                  </div>
                </motion.div>
                {}
                <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                <p className="text-gray-300 text-sm leading-relaxed mb-4">
                  {feature.description}
                </p>
                {}
                <div className="space-y-2">
                  {feature.metrics.slice(0, 2).map((metric, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-gray-400">{metric.label}</span>
                      <span className="text-cyan-400 font-semibold">{metric.value}</span>
                    </div>
                  ))}
                </div>
                {}
                <motion.div
                  className="mt-4 flex items-center text-blue-400 text-sm font-medium"
                  whileHover={{ x: 5 }}
                >
                  <span>Detayları Gör</span>
                  <motion.svg
                    className="w-4 h-4 ml-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    animate={{ rotate: activeFeature === feature.id ? 90 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </motion.svg>
                </motion.div>
              </motion.div>
              {}
              <AnimatePresence>
                {activeFeature === feature.id && (
                  <motion.div
                    className="absolute top-full left-0 right-0 mt-4 bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 z-20"
                    initial={{ opacity: 0, y: -20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -20, scale: 0.95 }}
                    transition={{ duration: 0.3 }}
                  >
                    <h4 className="text-lg font-bold text-white mb-3">Teknik Detaylar</h4>
                    <p className="text-gray-300 text-sm mb-4 leading-relaxed">
                      {feature.details}
                    </p>
                    <div className="grid grid-cols-1 gap-3">
                      {feature.metrics.map((metric, i) => (
                        <motion.div
                          key={i}
                          className="flex justify-between items-center p-3 bg-white/5 rounded-lg"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1 }}
                        >
                          <span className="text-gray-300 text-sm">{metric.label}</span>
                          <span className={`font-bold bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`}>
                            {metric.value}
                          </span>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
        {}
        <motion.div
          className="mt-20 grid md:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
        >
          {[
            { icon: <Camera className="w-6 h-6" />, title: 'WebRTC Entegrasyonu', description: 'Düşük gecikmeli video akışı' },
            { icon: <Zap className="w-6 h-6" />, title: 'ONNX Optimizasyonu', description: 'Hızlı model çıkarımı' },
            { icon: <Globe className="w-6 h-6" />, title: 'Çoklu Platform', description: 'Windows, macOS, Linux desteği' },
            { icon: <Shield className="w-6 h-6" />, title: 'Gizlilik Koruması', description: 'KVKK uyumlu veri işleme' }
          ].map((item, index) => (
            <motion.div
              key={item.title}
              className="text-center p-4"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 + index * 0.1 }}
              viewport={{ once: true }}
            >
              <motion.div
                className="w-12 h-12 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-full p-3 mx-auto mb-3 border border-blue-500/30"
                whileHover={{ scale: 1.1, rotate: 5 }}
              >
                <div className="text-blue-400 w-full h-full flex items-center justify-center">
                  {item.icon}
                </div>
              </motion.div>
              <h4 className="text-white font-semibold text-sm mb-2">{item.title}</h4>
              <p className="text-gray-400 text-xs">{item.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
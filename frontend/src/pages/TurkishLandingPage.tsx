import { motion, useInView } from 'framer-motion';
import {
  BarChart3,
  Brain,
  Camera,
  CheckCircle,
  Eye,
  Lightbulb,
  Monitor,
  Play,
  Shield,
  Sparkles,
  Target,
  TrendingUp,
  Users,
  Zap
} from 'lucide-react';
import React, { useRef } from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/layout/Header';
import { Layout } from '../components/layout/layout';
import AnimatedBackground from '../components/ui/animated-background';
import { GlassCard } from '../components/ui/glass-card';
import { NeonButton } from '../components/ui/neon-button';

const TurkishLandingPage: React.FC = () => {
  const coreFeatures = [
    {
      icon: Eye,
      title: 'Dikkat Takibi',
      description: 'Yüz yönelimi ve varlık tespiti ile dikkat seviyesini ölçer',
      details: [
        'Gerçek zamanlı yüz algılama',
        'Dikkat seviyesi skoru',
        'Dikkatsizlik uyarıları',
        'Tarihsel trend analizi'
      ],
      color: 'from-primary-500 to-accent-cyan',
      demo: 'Canlı dikkat gauge ve durum göstergesi'
    },
    {
      icon: BarChart3,
      title: 'Katılım Analizi',
      description: 'Hareket, ekran etkileşimi ve duruş analiziyle katılımı değerlendirir',
      details: [
        'Baş hareketi analizi',
        'Göz kontağı ölçümü',
        'Yüz ifadesi takibi',
        'Duruş değerlendirmesi'
      ],
      color: 'from-accent-cyan to-accent-purple',
      demo: 'Gerçek zamanlı katılım timeline grafiği'
    },
    {
      icon: Brain,
      title: 'Duygu Tanıma',
      description: 'Yüz mikroifadelerini analiz ederek duygu durumunu tespit eder',
      details: [
        'Mutluluk, odaklanma, şaşkınlık',
        'Yorgunluk ve stres tespiti',
        'Duygu dağılım analizi',
        'Valens ve uyarılma ölçümü'
      ],
      color: 'from-accent-purple to-accent-emerald',
      demo: 'İnteraktif duygu radar grafiği'
    },
    {
      icon: Target,
      title: 'Bakış Haritalama',
      description: 'Öğrencilerin ekranın hangi bölümüne odaklandığını takip eder',
      details: [
        'Gaze point tracking',
        'Ekran ısı haritası',
        'Bakış yönü analizi',
        'Odaklanma bölgeleri'
      ],
      color: 'from-accent-emerald to-primary-500',
      demo: 'Gerçek zamanlı gaze heatmap'
    }
  ];

  const technicalSpecs = [
    {
      icon: Camera,
      title: 'Gelişmiş Kamera Sistemi',
      specs: ['1280x720 HD çözünürlük', '30 FPS gerçek zamanlı analiz', 'Düşük ışık desteği']
    },
    {
      icon: Zap,
      title: 'Hızlı AI İşlemci',
      specs: ['<100ms analiz süresi', 'GPU hızlandırma desteği', 'Çoklu model ensemble']
    },
    {
      icon: Shield,
      title: 'Gizlilik ve Güvenlik',
      specs: ['Lokal işleme', 'Veri şifreleme', 'GDPR uyumlu']
    }
  ];

  const demoScenarios = [
    { key: 'Standart Ders', icon: '📚', description: 'Normal sınıf ortamı simülasyonu' },
    { key: 'Etkileşimli Ders', icon: '💬', description: 'Aktif katılım ve soru-cevap' },
    { key: 'Zorlu İçerik', icon: '🧮', description: 'Matematik ve fen dersleri' },
    { key: 'Grup Çalışması', icon: '👥', description: 'Takım tabanlı öğrenme' },
    { key: 'Sunum', icon: '📊', description: 'Öğrenci sunumları ve değerlendirme' }
  ];

  return (
    <Layout>
      <Header />
      <AnimatedBackground variant="neural" intensity="medium" color="rainbow" />
      
      <main className="relative z-10">
        {/* Hero Section with Turkish Focus */}
        <section className="relative pt-20 pb-16 overflow-hidden">
          <div className="container mx-auto px-4">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center max-w-6xl mx-auto"
            >
              {/* Status Badge */}
              <motion.div
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-primary-600/20 to-accent-cyan/20 border border-primary-500/30 mb-8 backdrop-blur-sm"
              >
                <Sparkles className="w-5 h-5 text-accent-cyan mr-2 animate-pulse" />
                <span className="text-primary-200 font-medium">
                  Türkiye'nin İlk AI Destekli Eğitim Platformu
                </span>
              </motion.div>

              {/* Main Title */}
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="text-5xl md:text-7xl font-bold text-white mb-6"
              >
                <span className="bg-gradient-to-r from-primary-400 via-accent-cyan to-accent-purple bg-clip-text text-transparent">
                  DersLens
                </span>
                <br />
                <span className="text-3xl md:text-4xl text-gray-300">
                  Gelişmiş AI Eğitim Analizi
                </span>
              </motion.h1>

              {/* Subtitle */}
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                className="text-xl md:text-2xl text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed"
              >
                Öğrenci dikkatini, katılımını, duygularını ve bakış yönünü gerçek zamanda ölçen 
                <span className="text-accent-cyan font-semibold"> yapay zeka teknolojisi</span>. 
                Eğitimin geleceğini bugün deneyimleyin.
              </motion.p>

              {/* Call to Action Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
                className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-16"
              >
                <Link to="/demo">
                  <NeonButton
                    size="lg"
                    variant="primary"
                    className="text-xl px-12 py-6 text-white"
                  >
                    <Eye className="w-6 h-6 mr-3" />
                    Demo'yu Dene
                  </NeonButton>
                </Link>

                <Link to="/dashboard">
                  <NeonButton
                    size="lg"
                    variant="accent"
                    className="text-xl px-12 py-6"
                  >
                    <TrendingUp className="w-6 h-6 mr-3" />
                    Öğretmen Paneli
                  </NeonButton>
                </Link>
              </motion.div>

              {/* Key Stats */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
                className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center"
              >
                <div>
                  <div className="text-3xl font-bold text-accent-cyan">%98</div>
                  <div className="text-sm text-gray-400">Doğruluk Oranı</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-accent-purple">~50ms</div>
                  <div className="text-sm text-gray-400">Analiz Hızı</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-accent-emerald">4 AI</div>
                  <div className="text-sm text-gray-400">Model Teknolojisi</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-primary-400">GDPR</div>
                  <div className="text-sm text-gray-400">Uyumlu</div>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Core Features Section - Turkish Focus */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                <span className="bg-gradient-to-r from-primary-400 to-accent-cyan bg-clip-text text-transparent">
                  Temel Özellikler
                </span>
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Türk eğitim sistemine özel olarak tasarlanmış AI teknolojileri
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12">
              {coreFeatures.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                >
                  <GlassCard neonAccent padding="lg" className="h-full">
                    <div className="flex items-start gap-6">
                      <div className={`w-20 h-20 bg-gradient-to-r ${feature.color} rounded-2xl flex items-center justify-center flex-shrink-0`}>
                        <feature.icon className="w-10 h-10 text-white" />
                      </div>
                      
                      <div className="flex-1">
                        <h3 className="text-2xl font-bold text-white mb-3">
                          {feature.title}
                        </h3>
                        <p className="text-gray-300 mb-6 text-lg">
                          {feature.description}
                        </p>
                        
                        <div className="space-y-3 mb-6">
                          {feature.details.map((detail, i) => (
                            <div key={i} className="flex items-center text-gray-300">
                              <CheckCircle className="w-5 h-5 text-accent-cyan mr-3 flex-shrink-0" />
                              <span>{detail}</span>
                            </div>
                          ))}
                        </div>
                        
                        <div className="bg-dark-800/50 rounded-lg p-4 border border-gray-600/30">
                          <div className="text-sm text-accent-cyan font-medium mb-1">
                            Demo Özelliği:
                          </div>
                          <div className="text-sm text-gray-300">
                            {feature.demo}
                          </div>
                        </div>
                      </div>
                    </div>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Technical Specifications */}
        <section className="py-20 bg-dark-900/30">
          <div className="container mx-auto px-4">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                <span className="bg-gradient-to-r from-accent-cyan to-accent-purple bg-clip-text text-transparent">
                  Teknik Özellikler
                </span>
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Endüstri lideri performans ve güvenlik standartları
              </p>
            </motion.div>

            <div className="grid md:grid-cols-3 gap-8">
              {technicalSpecs.map((spec, index) => (
                <motion.div
                  key={spec.title}
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                >
                  <GlassCard neonAccent className="text-center h-full">
                    <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-accent-cyan rounded-full flex items-center justify-center mx-auto mb-6">
                      <spec.icon className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-4">
                      {spec.title}
                    </h3>
                    <div className="space-y-2">
                      {spec.specs.map((item, i) => (
                        <div key={i} className="text-gray-300 text-sm">
                          {item}
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Demo Scenarios */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                <span className="bg-gradient-to-r from-accent-purple to-accent-emerald bg-clip-text text-transparent">
                  Demo Senaryoları
                </span>
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-12">
                Farklı eğitim durumları için özelleştirilmiş AI analizleri
              </p>
            </motion.div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-12">
              {demoScenarios.map((scenario, index) => (
                <motion.div
                  key={scenario.key}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                >
                  <GlassCard className="text-center p-6 hover:scale-105 transition-transform cursor-pointer">
                    <div className="text-3xl mb-3">{scenario.icon}</div>
                    <div className="text-white font-semibold mb-2 text-sm">
                      {scenario.key}
                    </div>
                    <div className="text-xs text-gray-300">
                      {scenario.description}
                    </div>
                  </GlassCard>
                </motion.div>
              ))}
            </div>

            {/* Final CTA */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center"
            >
              <GlassCard neonAccent padding="lg" className="max-w-4xl mx-auto">
                <h3 className="text-3xl font-bold text-white mb-6">
                  DersLens AI Teknolojisini Hemen Deneyin
                </h3>
                <p className="text-xl text-gray-300 mb-8">
                  Gerçek zamanlı analiz, türkçe arayüz ve güvenli veri işleme
                </p>
                
                <div className="flex flex-col sm:flex-row gap-6 justify-center">
                  <Link to="/demo">
                    <NeonButton
                      size="lg"
                      variant="primary"
                      className="text-lg px-8 py-4"
                    >
                      <Zap className="w-5 h-5 mr-2" />
                      Demo Başlat
                    </NeonButton>
                  </Link>
                </div>
                
                <div className="mt-8 text-sm text-gray-400">
                  ✅ Ücretsiz deneme • ✅ Kurulum gerektirmez • ✅ GDPR uyumlu
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </section>
      </main>
    </Layout>
  );
};

export default TurkishLandingPage;

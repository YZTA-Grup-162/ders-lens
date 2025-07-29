import { motion } from 'framer-motion';
import {
    AlertCircle,
    Award,
    BarChart3,
    Brain,
    CheckCircle,
    Clock,
    Eye,
    Heart,
    Lightbulb,
    Target,
    TrendingUp
} from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CameraDetector } from '../components/ai/camera-detector';
import { Header } from '../components/layout/header';
import { Layout } from '../components/layout/layout';
import { GlassCard } from '../components/ui/glass-card';

// Simple demo chart components
const EngagementChart: React.FC = () => {
  const [data, setData] = useState<number[]>(Array.from({length: 20}, () => Math.random() * 100));
  
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => [...prev.slice(1), Math.random() * 100]);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-48 flex items-end gap-1">
      {data.map((value, index) => (
        <div
          key={index}
          className="flex-1 bg-gradient-to-t from-primary-500 to-accent-cyan rounded-t opacity-70 transition-all duration-500"
          style={{ height: `${value}%` }}
        />
      ))}
    </div>
  );
};

// Real-time emotion chart that updates with AI data
const EmotionChart: React.FC<{ currentEmotion?: any }> = ({ currentEmotion }) => {
  const [emotionHistory, setEmotionHistory] = useState<{[key: string]: number[]}>({
    'Happy': Array.from({length: 20}, () => Math.random() * 40 + 20),
    'Focused': Array.from({length: 20}, () => Math.random() * 30 + 40),
    'Neutral': Array.from({length: 20}, () => Math.random() * 20 + 10),
    'Confused': Array.from({length: 20}, () => Math.random() * 15 + 5)
  });

  // Update emotion history when new data comes from camera
  useEffect(() => {
    if (currentEmotion?.emotions) {
      setEmotionHistory(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(key => {
          const newValue = currentEmotion.emotions[key] * 100 || Math.random() * 20;
          updated[key] = [...updated[key].slice(1), newValue];
        });
        return updated;
      });
    } else {
      // Simulate data when no camera input
      const interval = setInterval(() => {
        setEmotionHistory(prev => {
          const updated = { ...prev };
          Object.keys(updated).forEach(key => {
            const newValue = Math.random() * 30 + (key === 'Focused' ? 40 : 20);
            updated[key] = [...updated[key].slice(1), newValue];
          });
          return updated;
        });
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [currentEmotion]);

  const emotionColors = {
    'Happy': 'rgb(34, 197, 94)',
    'Focused': 'rgb(59, 130, 246)', 
    'Neutral': 'rgb(156, 163, 175)',
    'Confused': 'rgb(249, 115, 22)'
  };

  return (
    <div className="h-48 relative">
      <svg width="100%" height="100%" className="overflow-visible">
        {Object.entries(emotionHistory).map(([emotion, data], emotionIndex) => {
          const pathData = data.map((value, index) => {
            const x = (index / (data.length - 1)) * 100;
            const y = 100 - (value / 100) * 80; // Scale to 80% of height
            return `${index === 0 ? 'M' : 'L'} ${x}% ${y}%`;
          }).join(' ');

          return (
            <g key={emotion}>
              <path
                d={pathData}
                fill="none"
                stroke={emotionColors[emotion as keyof typeof emotionColors]}
                strokeWidth="2"
                className="drop-shadow-sm"
              />
              {/* Data points */}
              {data.map((value, index) => {
                if (index === data.length - 1) { // Only show last point
                  const x = (index / (data.length - 1)) * 100;
                  const y = 100 - (value / 100) * 80;
                  return (
                    <circle
                      key={index}
                      cx={`${x}%`}
                      cy={`${y}%`}
                      r="3"
                      fill={emotionColors[emotion as keyof typeof emotionColors]}
                      className="drop-shadow-sm animate-pulse"
                    />
                  );
                }
                return null;
              })}
            </g>
          );
        })}
      </svg>
      
      {/* Legend */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-center space-x-4 text-xs">
        {Object.entries(emotionColors).map(([emotion, color]) => (
          <div key={emotion} className="flex items-center space-x-1">
            <div 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: color }}
            />
            <span className="text-gray-400">{emotion}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Demo Student Dashboard - Real-time AI-powered classroom analytics for students
 */
export const DemoStudentDashboard: React.FC = () => {
  const { t } = useTranslation();
  const [isLive, setIsLive] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [aiAnalysis, setAiAnalysis] = useState<any>(null);
  const [personalStats, setPersonalStats] = useState({
    attentionScore: 85,
    engagementLevel: 92,
    participationPoints: 247,
    streakDays: 12
  });

  // Handle AI analysis updates from camera
  const handleAnalysisUpdate = (result: any) => {
    setAiAnalysis(result);
    
    // Update personal stats based on AI analysis
    if (result.attention) {
      setPersonalStats(prev => ({
        ...prev,
        attentionScore: Math.round(result.attention.score * 100)
      }));
    }
    
    if (result.engagement) {
      setPersonalStats(prev => ({
        ...prev,
        engagementLevel: Math.round(result.engagement.score * 100)
      }));
    }
  };

  // Simulate real-time updates only when no AI data
  useEffect(() => {
    if (!aiAnalysis) {
      const timer = setInterval(() => {
        setCurrentTime(new Date());
        setPersonalStats(prev => ({
          attentionScore: Math.max(70, Math.min(100, prev.attentionScore + (Math.random() - 0.5) * 10)),
          engagementLevel: Math.max(60, Math.min(100, prev.engagementLevel + (Math.random() - 0.5) * 8)),
          participationPoints: prev.participationPoints + Math.floor(Math.random() * 3),
          streakDays: prev.streakDays
        }));
      }, 5000);
      return () => clearInterval(timer);
    } else {
      setCurrentTime(new Date());
    }
  }, [aiAnalysis]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400';
    if (score >= 60) return 'text-yellow-400';
    if (score >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getEngagementIcon = (level: number) => {
    if (level >= 80) return <CheckCircle className="w-5 h-5 text-emerald-400" />;
    if (level >= 60) return <Eye className="w-5 h-5 text-yellow-400" />;
    return <AlertCircle className="w-5 h-5 text-red-400" />;
  };

  return (
    <Layout>
      {/* Animated Background */}
      <div className="animated-background" />
      
      <Header />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <GlassCard>
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">
                  {t('demo.student.welcome', 'Hoş Geldin, Öğrenci!')}
                </h1>
                <p className="text-gray-400">
                  {t('demo.student.subtitle', 'AI destekli öğrenme deneyimini keşfet')}
                </p>
                <div className="flex items-center space-x-2 mt-2">
                  <Clock className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-500 text-sm">
                    {currentTime.toLocaleTimeString('tr-TR')}
                  </span>
                  {isLive && (
                    <>
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-red-400 text-sm">CANLI</span>
                    </>
                  )}
                </div>
              </div>
              <div className="text-right">
                <div className="flex items-center space-x-4">
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${getScoreColor(personalStats.attentionScore)}`}>
                      {personalStats.attentionScore}
                    </div>
                    <div className="text-xs text-gray-400">Dikkat</div>
                  </div>
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${getScoreColor(personalStats.engagementLevel)}`}>
                      {personalStats.engagementLevel}
                    </div>
                    <div className="text-xs text-gray-400">Katılım</div>
                  </div>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>

        {/* AI Camera Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <CameraDetector 
            onAnalysisUpdate={handleAnalysisUpdate}
            className="mb-8"
          />
        </motion.div>

        {/* Real-time Analytics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Emotion Analysis Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <GlassCard>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Heart className="w-5 h-5 text-pink-400" />
                  <h3 className="text-lg font-semibold text-white">
                    {t('demo.emotion.title', 'Duygu Analizi')}
                  </h3>
                </div>
                <div className="text-xs text-gray-400">Gerçek Zamanlı</div>
              </div>
              <EmotionChart currentEmotion={aiAnalysis?.emotion} />
            </GlassCard>
          </motion.div>

          {/* Engagement Trend */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <GlassCard>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">
                    {t('demo.engagement.title', 'Katılım Trendi')}
                  </h3>
                </div>
                <div className="text-xs text-gray-400">Son 10 dakika</div>
              </div>
              <EngagementChart />
            </GlassCard>
          </motion.div>
        </div>

        {/* Personal Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Attention Score */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <GlassCard>
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <Eye className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-gray-400">Dikkat Skoru</span>
                  </div>
                  <div className={`text-2xl font-bold ${getScoreColor(personalStats.attentionScore)}`}>
                    {personalStats.attentionScore}%
                  </div>
                </div>
                {getEngagementIcon(personalStats.attentionScore)}
              </div>
            </GlassCard>
          </motion.div>

          {/* Engagement Level */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <GlassCard>
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-gray-400">Katılım Seviyesi</span>
                  </div>
                  <div className={`text-2xl font-bold ${getScoreColor(personalStats.engagementLevel)}`}>
                    {personalStats.engagementLevel}%
                  </div>
                </div>
                <TrendingUp className="w-5 h-5 text-emerald-400" />
              </div>
            </GlassCard>
          </motion.div>

          {/* Participation Points */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <GlassCard>
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <Target className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm text-gray-400">Katılım Puanı</span>
                  </div>
                  <div className="text-2xl font-bold text-emerald-400">
                    {personalStats.participationPoints}
                  </div>
                </div>
                <Award className="w-5 h-5 text-yellow-400" />
              </div>
            </GlassCard>
          </motion.div>

          {/* Learning Streak */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.7 }}
          >
            <GlassCard>
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <CheckCircle className="w-4 h-4 text-orange-400" />
                    <span className="text-sm text-gray-400">Öğrenme Serisi</span>
                  </div>
                  <div className="text-2xl font-bold text-orange-400">
                    {personalStats.streakDays} gün
                  </div>
                </div>
                <Lightbulb className="w-5 h-5 text-yellow-400" />
              </div>
            </GlassCard>
          </motion.div>
        </div>

        {/* AI Recommendations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <GlassCard>
            <div className="flex items-center space-x-2 mb-4">
              <Brain className="w-5 h-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">
                {t('demo.ai.recommendations', 'AI Önerileri')}
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-300 font-medium">Çok İyi!</span>
                </div>
                <p className="text-gray-300 text-sm">
                  Dikkat seviyeniz harika! Bu performansı sürdürmek için ara ara göz egzersizleri yapın.
                </p>
              </div>
              <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Lightbulb className="w-4 h-4 text-blue-400" />
                  <span className="text-blue-300 font-medium">İpucu</span>
                </div>
                <p className="text-gray-300 text-sm">
                  {aiAnalysis?.emotion?.dominant === 'Confused' 
                    ? 'Anlamadığınız bir konu var gibi görünüyor. Öğretmeninize soru sormaktan çekinmeyin!'
                    : 'Öğrenme motivasyonunuzu artırmak için hedefler belirleyebilirsiniz.'
                  }
                </p>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      </main>
    </Layout>
  );
};

export default DemoStudentDashboard;

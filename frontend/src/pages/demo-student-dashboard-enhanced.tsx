import { motion } from 'framer-motion';
import {
  Activity,
  AlertCircle,
  Award,
  Brain,
  Camera,
  Clock,
  Eye,
  Frown,
  Heart,
  Meh,
  Smile,
  Star,
  Target,
  ThumbsUp,
  Zap
} from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CameraDetector } from '../components/ai/camera-detector';
import { Header } from '../components/layout/header';
import { Layout } from '../components/layout/layout';
import { GlassCard } from '../components/ui/glass-card';

// Real-time emotion display component
const EmotionMeter: React.FC<{ emotion: string; confidence: number; color: string }> = ({ 
  emotion, 
  confidence, 
  color 
}) => {
  const getEmotionIcon = (emotion: string) => {
    const emotionLower = emotion.toLowerCase();
    switch (emotionLower) {
      case 'happy':
      case 'mutlu': 
        return <Smile className="w-6 h-6" />;
      case 'sad':
      case 'üzgün':
        return <Frown className="w-6 h-6" />;
      case 'angry':
      case 'kızgın':
        return <AlertCircle className="w-6 h-6" />;
      case 'confused':
      case 'şaşkın':
        return <Meh className="w-6 h-6" />;
      case 'focused':
      case 'odaklı':
        return <Eye className="w-6 h-6" />;
      case 'engaged':
      case 'katılım':
        return <ThumbsUp className="w-6 h-6" />;
      case 'fear':
      case 'korkmuş':
        return <AlertCircle className="w-6 h-6" />;
      case 'disgust':
      case 'iğrenmiş':
        return <Frown className="w-6 h-6" />;
      case 'neutral':
      case 'nötr':
        return <Heart className="w-6 h-6" />;
      default: 
        return <Heart className="w-6 h-6" />;
    }
  };

  return (
    <motion.div
      className="flex items-center space-x-3 p-3 rounded-lg bg-gray-800/30 border border-gray-700/50"
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className={`${color} p-2 rounded-full bg-gray-800/50`}>
        {getEmotionIcon(emotion)}
      </div>
      <div className="flex-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-white font-medium capitalize">{emotion}</span>
          <span className={`text-sm font-bold ${color}`}>{Math.round(confidence * 100)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <motion.div
            className={`h-2 rounded-full bg-gradient-to-r ${color.includes('emerald') ? 'from-emerald-500 to-emerald-400' : 
              color.includes('blue') ? 'from-blue-500 to-blue-400' :
              color.includes('yellow') ? 'from-yellow-500 to-yellow-400' :
              color.includes('red') ? 'from-red-500 to-red-400' :
              'from-purple-500 to-purple-400'}`}
            style={{ width: `${confidence * 100}%` }}
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>
    </motion.div>
  );
};

// Live attention tracker
const AttentionTracker: React.FC<{ attentionScore: number; isLive: boolean }> = ({ 
  attentionScore, 
  isLive 
}) => {
  const getAttentionLevel = (score: number) => {
    if (score >= 80) return { level: 'Çok Odaklı', color: 'text-emerald-400', icon: Eye };
    if (score >= 60) return { level: 'Odaklı', color: 'text-blue-400', icon: Eye };
    if (score >= 40) return { level: 'Kısmen Odaklı', color: 'text-yellow-400', icon: AlertCircle };
    return { level: 'Dağınık', color: 'text-red-400', icon: AlertCircle };
  };

  const attention = getAttentionLevel(attentionScore);
  const Icon = attention.icon;

  return (
    <motion.div
      className="relative"
      animate={isLive ? { scale: [1, 1.02, 1] } : {}}
      transition={{ duration: 2, repeat: Infinity }}
    >
      <div className="text-center space-y-3">
        <div className="relative">
          <div className={`w-24 h-24 mx-auto rounded-full border-4 ${attention.color.replace('text-', 'border-')} flex items-center justify-center relative`}>
            <Icon className={`w-8 h-8 ${attention.color}`} />
            {isLive && (
              <div className={`absolute -top-1 -right-1 w-4 h-4 ${attention.color.replace('text-', 'bg-')} rounded-full animate-pulse`} />
            )}
          </div>
          <div className={`text-3xl font-bold ${attention.color} mt-2`}>
            {attentionScore}%
          </div>
        </div>
        <div>
          <div className="text-white font-medium">{attention.level}</div>
          <div className="text-gray-400 text-sm">Dikkat Seviyesi</div>
        </div>
      </div>
    </motion.div>
  );
};

// Engagement pulse indicator
const EngagementPulse: React.FC<{ level: number; isActive: boolean }> = ({ level, isActive }) => {
  const pulseColor = level >= 80 ? 'bg-emerald-500' : 
                   level >= 60 ? 'bg-blue-500' : 
                   level >= 40 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="relative flex items-center justify-center">
      <div className={`w-16 h-16 ${pulseColor} rounded-full flex items-center justify-center`}>
        <Activity className="w-8 h-8 text-white" />
      </div>
      {isActive && (
        <>
          <div className={`absolute w-16 h-16 ${pulseColor} rounded-full animate-ping opacity-30`} />
          <div className={`absolute w-20 h-20 ${pulseColor} rounded-full animate-ping opacity-20 animation-delay-300`} />
        </>
      )}
      <div className="absolute -bottom-2 text-white font-bold text-sm">
        {level}%
      </div>
    </div>
  );
};

/**
 * Enhanced Demo Student Dashboard - Highly personalized AI-powered experience
 */
export const DemoStudentDashboardEnhanced: React.FC = () => {
  const { t } = useTranslation();
  const [isLive, setIsLive] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [aiAnalysis, setAiAnalysis] = useState<any>(null);
  const [currentEmotion, setCurrentEmotion] = useState<any>(null);
  const [attentionHistory, setAttentionHistory] = useState<number[]>([]);
  const [faceDetected, setFaceDetected] = useState(false);
  const [lastFaceDetection, setLastFaceDetection] = useState<Date | null>(null);
  const [personalStats, setPersonalStats] = useState({
    attentionScore: 75,
    engagementLevel: 85,
    participationPoints: 247,
    streakDays: 12,
    todayFocus: 0,
    emotionStability: 0
  });

  // Handle AI analysis updates from camera
  const handleAnalysisUpdate = (result: any) => {
    console.log('AI Analysis Update:', result);
    
    // Force debug display
    if (result && typeof result === 'object') {
      console.log('Face detected status:', result.faceDetected);
      console.log('Full result object:', JSON.stringify(result, null, 2));
    }
    
    setAiAnalysis(result);
    
    // Track face detection status
    const isFaceDetected = result.faceDetected || false;
    console.log(`Face detection changing from ${faceDetected} to ${isFaceDetected}`);
    
    // Add temporary alert for debugging
    if (isFaceDetected !== faceDetected) {
      console.log(`🔥 FACE DETECTION STATE CHANGE: ${faceDetected} → ${isFaceDetected}`);
    }
    
    setFaceDetected(isFaceDetected);
    
    if (isFaceDetected) {
      setLastFaceDetection(new Date());
    }
    
    // Only process emotion/attention data if face is detected
    if (isFaceDetected && result.emotion) {
      // Update current emotion with real AI data
      const emotions = result.emotion.emotions || {};
      const dominantEmotion = result.emotion.dominant || 'neutral';
      const confidence = result.emotion.confidence || 0;
      
      setCurrentEmotion({
        name: dominantEmotion,
        confidence: confidence,
        all: emotions
      });
      
      // Update attention score
      if (result.attention?.score !== undefined) {
        const newScore = Math.round(result.attention.score * 100);
        setPersonalStats(prev => ({
          ...prev,
          attentionScore: newScore
        }));
        
        // Update attention history
        setAttentionHistory(prev => [...prev.slice(-19), newScore]);
      }
      
      // Update engagement level
      if (result.engagement?.score !== undefined) {
        setPersonalStats(prev => ({
          ...prev,
          engagementLevel: Math.round(result.engagement.score * 100)
        }));
      }
      
      // Calculate today's focus time and emotion stability
      setPersonalStats(prev => ({
        ...prev,
        todayFocus: prev.todayFocus + (result.attention?.score > 0.7 ? 1 : 0),
        emotionStability: currentEmotion ? Math.round(currentEmotion.confidence * 100) : prev.emotionStability
      }));
    }
  };

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Simulate some data when no AI input
  useEffect(() => {
    if (!aiAnalysis) {
      const interval = setInterval(() => {
        const mockAnalysis = {
          emotions: {
            'Focused': Math.random() * 0.4 + 0.6,
            'Happy': Math.random() * 0.3 + 0.2,
            'Neutral': Math.random() * 0.2 + 0.1,
            'Confused': Math.random() * 0.1
          },
          attention: { score: Math.random() * 0.3 + 0.7 },
          engagement: { score: Math.random() * 0.2 + 0.8 }
        };
        handleAnalysisUpdate(mockAnalysis);
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [aiAnalysis]);

  const getMotivationalMessage = () => {
    if (!faceDetected) {
      return "👀 Lütfen kameranın görüş alanına geç - AI analizi için yüzünü görebilmeliyim!";
    }
    
    if (!currentEmotion) return "AI sizi analiz ediyor...";
    
    const { name, confidence } = currentEmotion;
    
    if ((name === 'Happy' || name === 'mutlu') && confidence > 0.7) {
      return "🎉 Harika bir ruh halinde görünüyorsun! Bu pozitif enerjiyi sürdür!";
    } else if ((name === 'Focused' || name === 'odaklı') && confidence > 0.8) {
      return "🎯 Süper odaklısın! Bu konsantrasyonla her şeyi başarabilirsin!";
    } else if ((name === 'Confused' || name === 'şaşkın') && confidence > 0.6) {
      return "🤔 Anlamadığın bir şey var mı? Öğretmenine soru sormaktan çekinme!";
    } else if ((name === 'Fear' || name === 'korkmuş') && confidence > 0.6) {
      return "😊 Rahat ol! Öğrenmek eğlenceli olmalı, endişelenme!";
    } else if ((name === 'Sad' || name === 'üzgün') && confidence > 0.6) {
      return "💪 Her zorluk yeni bir öğrenme fırsatı! Sen yapabilirsin!";
    } else if (personalStats.attentionScore > 85) {
      return "⭐ Dikkat seviyeN mükemmel! Böyle devam et!";
    } else if (personalStats.engagementLevel > 80) {
      return "🚀 Katılımın çok iyi! Aktif öğrenci olmaya devam et!";
    }
    
    return "💪 Her dakika yeni bir öğrenme fırsatı! Devam et!";
  };

  const formatTime = (date: Date) => {
    return {
      time: date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
      seconds: date.getSeconds().toString().padStart(2, '0')
    };
  };

  const timeFormatted = formatTime(currentTime);

  return (
    <Layout>
      <Header />
      
      <main className={`container mx-auto px-4 py-8 space-y-6 transition-all duration-1000 ${
        !faceDetected && isLive ? 'shadow-[0_0_50px_rgba(239,68,68,0.4)] border border-red-500/30 rounded-xl' : ''
      }`}>
        
        {/* DEBUG INFO */}
        <div className="fixed top-0 right-0 bg-black/80 text-white p-2 text-xs z-50">
          <div>Face Detected: {String(faceDetected)}</div>
          <div>AI Analysis: {aiAnalysis ? 'Yes' : 'No'}</div>
          <div>Live: {String(isLive)}</div>
          {aiAnalysis && <div>Emotion: {aiAnalysis.emotion?.dominant}</div>}
        </div>
        
        {/* Hero Section with Live Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <GlassCard className="relative overflow-hidden">
            {/* Animated background effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-emerald-500/10 animate-pulse" />
            
            <div className="relative flex flex-col lg:flex-row items-center justify-between space-y-4 lg:space-y-0">
              <div className="text-center lg:text-left">
                <motion.h1 
                  className="text-4xl lg:text-5xl font-bold text-white mb-2"
                  animate={{ scale: [1, 1.02, 1] }}
                  transition={{ duration: 3, repeat: Infinity }}
                >
                  Merhaba! 👋
                </motion.h1>
                <p className="text-gray-300 text-lg mb-2">AI destekli öğrenme seansın başladı</p>
                <div className="flex items-center justify-center lg:justify-start space-x-4">
                  <div className="flex items-center space-x-2">
                    <Clock className="w-5 h-5 text-blue-400" />
                    <span className="text-blue-300 font-mono text-xl">
                      {timeFormatted.time}
                      <span className="text-blue-500">:{timeFormatted.seconds}</span>
                    </span>
                  </div>
                  {isLive && (
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-red-400 font-medium">CANLI ANALİZ</span>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Live attention tracker */}
              {faceDetected && (
                <div className="flex-shrink-0">
                  <AttentionTracker 
                    attentionScore={personalStats.attentionScore} 
                    isLive={isLive}
                  />
                </div>
              )}
            </div>
          </GlassCard>
        </motion.div>

        {/* Face Detection Status Warning */}
        {isLive && !faceDetected && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
          >
            <GlassCard className="relative overflow-hidden border-red-500/50 bg-red-900/20">
              <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 via-red-600/10 to-red-500/20 animate-pulse" />
              <div className="relative flex items-center justify-center space-x-3 py-4">
                <AlertCircle className="w-6 h-6 text-red-400 animate-pulse" />
                <span className="text-red-300 font-medium text-lg">
                  Kamera görüş alanında yüz algılanamadı - Lütfen kameranın önüne geçin
                </span>
              </div>
            </GlassCard>
          </motion.div>
        )}

        {/* Camera and Current Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Feed */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="lg:col-span-2"
          >
            <GlassCard>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Camera className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">Canlı Kamera Analizi</h3>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-400">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span>Aktif</span>
                </div>
              </div>
              <CameraDetector 
                onAnalysisUpdate={handleAnalysisUpdate}
                className="rounded-lg overflow-hidden"
              />
            </GlassCard>
          </motion.div>

          {/* Current Emotion & Engagement */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="space-y-4"
          >
            {/* Current Emotion */}
            {faceDetected ? (
              <GlassCard>
                <div className="text-center space-y-4">
                  <div className="flex items-center justify-center space-x-2 mb-3">
                    <Heart className="w-5 h-5 text-pink-400" />
                    <h3 className="text-lg font-semibold text-white">Şu Anki Duygum</h3>
                  </div>
                  
                  {currentEmotion ? (
                    <motion.div
                      key={currentEmotion.name}
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.3 }}
                      className="space-y-3"
                    >
                      <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center ${
                        currentEmotion.name === 'Happy' ? 'bg-emerald-500/20 text-emerald-400' :
                        currentEmotion.name === 'Focused' ? 'bg-blue-500/20 text-blue-400' :
                        currentEmotion.name === 'Confused' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-purple-500/20 text-purple-400'
                      }`}>
                        {currentEmotion.name === 'Happy' && <Smile className="w-10 h-10" />}
                        {currentEmotion.name === 'Focused' && <Eye className="w-10 h-10" />}
                        {currentEmotion.name === 'Confused' && <Meh className="w-10 h-10" />}
                        {!['Happy', 'Focused', 'Confused'].includes(currentEmotion.name) && <Heart className="w-10 h-10" />}
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-white capitalize">
                          {currentEmotion.name === 'Happy' ? 'Mutlu' :
                           currentEmotion.name === 'Focused' ? 'Odaklı' :
                           currentEmotion.name === 'Confused' ? 'Kafası Karışık' :
                           currentEmotion.name}
                        </div>
                        <div className="text-sm text-gray-400">
                          %{Math.round(currentEmotion.confidence * 100)} güven
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="text-gray-400">Analiz ediliyor...</div>
                  )}
                </div>
              </GlassCard>
            ) : (
              <GlassCard className="border-red-500/30 bg-red-900/10">
                <div className="text-center space-y-4">
                  <div className="flex items-center justify-center space-x-2 mb-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <h3 className="text-lg font-semibold text-red-300">Duygu Algılanamıyor</h3>
                  </div>
                  <div className="text-sm text-red-400">
                    Yüz algılandığında duygusal analiz görünür
                  </div>
                </div>
              </GlassCard>
            )}

            {/* Engagement Pulse */}
            {faceDetected ? (
              <GlassCard>
                <div className="text-center space-y-4">
                  <div className="flex items-center justify-center space-x-2 mb-3">
                    <Zap className="w-5 h-5 text-yellow-400" />
                    <h3 className="text-lg font-semibold text-white">Katılım Nabzım</h3>
                  </div>
                  <EngagementPulse 
                    level={personalStats.engagementLevel} 
                    isActive={isLive}
                  />
                  <div className="text-sm text-gray-400">
                    {personalStats.engagementLevel >= 80 ? 'Çok Aktif!' :
                     personalStats.engagementLevel >= 60 ? 'İyi Katılım' :
                     personalStats.engagementLevel >= 40 ? 'Orta Seviye' : 'Pasif'}
                  </div>
                </div>
              </GlassCard>
            ) : (
              <GlassCard className="border-red-500/30 bg-red-900/10">
                <div className="text-center space-y-4">
                  <div className="flex items-center justify-center space-x-2 mb-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <h3 className="text-lg font-semibold text-red-300">Katılım Algılanamıyor</h3>
                  </div>
                  <div className="text-sm text-red-400">
                    Yüz algılandığında katılım seviyen görünür
                  </div>
                </div>
              </GlassCard>
            )}
          </motion.div>
        </div>

        {/* Motivational Message */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <GlassCard className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-purple-500/20">
            <div className="text-center">
              <div className="flex items-center justify-center space-x-2 mb-3">
                <Star className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-white">AI Motivasyon Mesajı</h3>
                <Star className="w-5 h-5 text-yellow-400" />
              </div>
              <motion.p 
                key={getMotivationalMessage()}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-lg text-gray-200 font-medium"
              >
                {getMotivationalMessage()}
              </motion.p>
            </div>
          </GlassCard>
        </motion.div>

        {/* Personal Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Today's Focus Time */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <GlassCard className="text-center">
              <div className="space-y-2">
                <Eye className="w-8 h-8 text-blue-400 mx-auto" />
                <div className="text-2xl font-bold text-blue-400">
                  {Math.floor(personalStats.todayFocus / 2)}dk
                </div>
                <div className="text-sm text-gray-400">Bugün Odaklanma</div>
              </div>
            </GlassCard>
          </motion.div>

          {/* Participation Points */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <GlassCard className="text-center">
              <div className="space-y-2">
                <Award className="w-8 h-8 text-yellow-400 mx-auto" />
                <div className="text-2xl font-bold text-yellow-400">
                  {personalStats.participationPoints}
                </div>
                <div className="text-sm text-gray-400">Katılım Puanı</div>
              </div>
            </GlassCard>
          </motion.div>

          {/* Learning Streak */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <GlassCard className="text-center">
              <div className="space-y-2">
                <Target className="w-8 h-8 text-emerald-400 mx-auto" />
                <div className="text-2xl font-bold text-emerald-400">
                  {personalStats.streakDays}
                </div>
                <div className="text-sm text-gray-400">Günlük Seri</div>
              </div>
            </GlassCard>
          </motion.div>

          {/* Emotion Stability */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.7 }}
          >
            <GlassCard className="text-center">
              <div className="space-y-2">
                <Brain className="w-8 h-8 text-purple-400 mx-auto" />
                <div className="text-2xl font-bold text-purple-400">
                  {personalStats.emotionStability}%
                </div>
                <div className="text-sm text-gray-400">Duygu Stabilitesi</div>
              </div>
            </GlassCard>
          </motion.div>
        </div>

        {/* Current Emotions Breakdown */}
        {currentEmotion?.all && faceDetected && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <GlassCard>
              <div className="flex items-center space-x-2 mb-4">
                <Activity className="w-5 h-5 text-cyan-400" />
                <h3 className="text-lg font-semibold text-white">Detaylı Duygu Analizi</h3>
                <span className="text-xs text-gray-400">Şu anda</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(currentEmotion.all as {[key: string]: number})
                  .sort(([,a], [,b]) => (b as number) - (a as number))
                  .slice(0, 6)
                  .map(([emotion, confidence]) => (
                    <EmotionMeter
                      key={emotion}
                      emotion={emotion === 'Happy' ? 'Mutlu' : 
                              emotion === 'Focused' ? 'Odaklı' :
                              emotion === 'Confused' ? 'Kafası Karışık' :
                              emotion === 'Neutral' ? 'Nötr' :
                              emotion === 'Surprised' ? 'Şaşkın' :
                              emotion}
                      confidence={confidence as number}
                      color={
                        emotion === 'Happy' ? 'text-emerald-400' :
                        emotion === 'Focused' ? 'text-blue-400' :
                        emotion === 'Confused' ? 'text-yellow-400' :
                        emotion === 'Sad' ? 'text-red-400' :
                        'text-purple-400'
                      }
                    />
                  ))}
              </div>
            </GlassCard>
          </motion.div>
        )}
      </main>
    </Layout>
  );
};

export default DemoStudentDashboardEnhanced;

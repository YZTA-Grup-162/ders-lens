import { motion } from 'framer-motion';
import { RotateCcw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { LiveCameraAnalysis } from '../components/camera/LiveCameraAnalysis';

interface AnalysisData {
  attention: number;
  engagement: number;
  emotion: string;
  gaze: { x: number; y: number };
}

const DemoPage = () => {
  const [analysisData, setAnalysisData] = useState<AnalysisData>({
    attention: 0,
    engagement: 0,
    emotion: 'neutral',
    gaze: { x: 50, y: 50 }
  });

  // Debug: Component mount durumunu logla
  useEffect(() => {
    console.log('🎯 DemoPage mounted!');
    return () => console.log('🎯 DemoPage unmounted!');
  }, []);

  const resetDemo = () => {
    console.log('🔄 Reset demo clicked');
    setAnalysisData({
      attention: 0,
      engagement: 0,
      emotion: 'neutral',
      gaze: { x: 50, y: 50 }
    });
  };

  const handleAnalysisResult = (result: any) => {
    console.log('📊 Demo page received analysis result:', result);
    
    // Transform the result to match our demo format
    const newData = {
      attention: result.attention?.attention_score ? Math.round(result.attention.attention_score * 100) : Math.round(Math.random() * 100),
      engagement: result.engagement?.engagement_score ? Math.round(result.engagement.engagement_score * 100) : Math.round(Math.random() * 100),
      emotion: result.emotion?.dominant_emotion || 'happy',
      gaze: {
        x: result.gaze?.gaze_x || Math.random() * 100,
        y: result.gaze?.gaze_y || Math.random() * 100
      }
    };
    
    console.log('📈 Setting new analysis data:', newData);
    setAnalysisData(newData);
  };

  return (
    <div className="min-h-screen pt-20 bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            DersLens Old Demo
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Klasik yapay zeka analizi sistemi (eski versiyon)
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Live Camera Analysis Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
          >
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                Canlı Kamera Analizi
              </h2>
              <button
                onClick={resetDemo}
                className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                Sıfırla
              </button>
            </div>
            
            <LiveCameraAnalysis 
              onAnalysisResult={handleAnalysisResult}
              analysisInterval={2000}
            />
          </motion.div>

          {/* Analysis Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="space-y-6"
          >
            {/* Attention Meter */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Dikkat Seviyesi
              </h3>
              <div className="relative">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <motion.div
                    className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${analysisData.attention}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-2xl font-bold text-gray-900 dark:text-white mt-2 block">
                  %{analysisData.attention}
                </span>
              </div>
            </div>

            {/* Engagement Meter */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Katılım Seviyesi
              </h3>
              <div className="relative">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <motion.div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${analysisData.engagement}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-2xl font-bold text-gray-900 dark:text-white mt-2 block">
                  %{analysisData.engagement}
                </span>
              </div>
            </div>

            {/* Emotion Display */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Tespit Edilen Duygu
              </h3>
              <div className="text-center">
                <motion.div
                  key={analysisData.emotion}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3 }}
                  className="text-4xl mb-2"
                >
                  {getEmotionEmoji(analysisData.emotion)}
                </motion.div>
                <span className="text-xl font-medium text-gray-900 dark:text-white capitalize">
                  {translateEmotion(analysisData.emotion)}
                </span>
              </div>
            </div>

            {/* Gaze Tracking */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Bakış Haritası
              </h3>
              <div className="relative bg-gray-100 dark:bg-gray-800 rounded-lg h-32">
                <motion.div
                  className="absolute w-3 h-3 bg-red-500 rounded-full transform -translate-x-1/2 -translate-y-1/2"
                  animate={{
                    left: `${analysisData.gaze.x}%`,
                    top: `${analysisData.gaze.y}%`
                  }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Kırmızı nokta: Mevcut bakış odağı
              </p>
            </div>
          </motion.div>
        </div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-8 text-center"
        >
          <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Nasıl Kullanılır?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600 dark:text-gray-400">
              <div>
                <span className="font-medium">1. Kamerayı Başlatın</span>
                <p>Sol panelde "Kamerayı Başlat" butonuna tıklayın</p>
              </div>
              <div>
                <span className="font-medium">2. Kameraya Bakın</span>
                <p>Sistemi test etmek için kameraya doğrudan bakın</p>
              </div>
              <div>
                <span className="font-medium">3. Sonuçları İzleyin</span>
                <p>Sağ panelde gerçek zamanlı analiz sonuçlarını görün</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

// Helper functions
const getEmotionEmoji = (emotion: string): string => {
  const emojiMap: { [key: string]: string } = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    surprise: '😲',
    fear: '😨',
    disgust: '🤢',
    neutral: '😐',
    focused: '🤔',
    confused: '😕'
  };
  return emojiMap[emotion] || '😐';
};

const translateEmotion = (emotion: string): string => {
  const translationMap: { [key: string]: string } = {
    happy: 'Mutlu',
    sad: 'Üzgün',
    angry: 'Kızgın',
    surprise: 'Şaşırmış',
    fear: 'Korkmuş',
    disgust: 'İğrenmiş',
    neutral: 'Nötr',
    focused: 'Odaklanmış',
    confused: 'Kafası Karışık'
  };
  return translationMap[emotion] || 'Bilinmiyor';
};

export default DemoPage;

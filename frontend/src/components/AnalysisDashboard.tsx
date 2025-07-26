import { motion } from 'framer-motion';
import { BarChart3, Brain, Eye, Target, TrendingUp, Users } from 'lucide-react';
import { AnalysisResult } from '../services/apiService';
interface AnalysisDashboardProps {
  analysisData: AnalysisResult | null;
  className?: string;
}
const AnalysisDashboard = ({ analysisData, className = '' }: AnalysisDashboardProps) => {
  console.log('AnalysisDashboard render - analysisData:', analysisData);
  const getEmotionColor = (emotion: string) => {
    const colors: { [key: string]: string } = {
      'Odaklı': 'text-green-600 dark:text-green-400',
      'Meraklı': 'text-blue-600 dark:text-blue-400',
      'Anlıyor': 'text-purple-600 dark:text-purple-400',
      'Düşünüyor': 'text-yellow-600 dark:text-yellow-400',
      'Kararsız': 'text-orange-600 dark:text-orange-400',
      'Sıkılmış': 'text-red-600 dark:text-red-400'
    };
    return colors[emotion] || 'text-gray-600 dark:text-gray-400';
  };
  const getAttentionColor = (attention: number) => {
    if (attention >= 80) return 'text-green-600 dark:text-green-400';
    if (attention >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };
  const getGaugeColor = (value: number) => {
    if (value >= 80) return 'from-green-500 to-green-600';
    if (value >= 60) return 'from-yellow-500 to-orange-500';
    return 'from-red-500 to-red-600';
  };
  const CircularGauge = ({ 
    value, 
    label, 
    icon: Icon, 
    gradient 
  }: { 
    value: number; 
    label: string; 
    icon: any; 
    gradient: string;
  }) => {
    const circumference = 2 * Math.PI * 45;
    const strokeDasharray = circumference;
    const strokeDashoffset = circumference - (value / 100) * circumference;
    return (
      <div className="flex flex-col items-center">
        <div className="relative w-24 h-24">
          <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
            {}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              className="text-gray-200 dark:text-gray-700"
            />
            {}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="url(#gradient)"
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
            />
            {}
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" className={gradient.split(' ')[0].replace('from-', 'stop-')} />
                <stop offset="100%" className={gradient.split(' ')[2].replace('to-', 'stop-')} />
              </linearGradient>
            </defs>
          </svg>
          {}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className={`w-8 h-8 bg-gradient-to-r ${gradient} rounded-full flex items-center justify-center`}>
              <Icon className="w-4 h-4 text-white" />
            </div>
          </div>
        </div>
        {}
        <div className="text-center mt-2">
          <div className={`text-2xl font-bold ${getAttentionColor(value)}`}>
            {Math.round(value)}%
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">{label}</div>
        </div>
      </div>
    );
  };
  if (!analysisData) {
    return (
      <div className={`bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-white/20 dark:border-gray-700/20 ${className}`}>
        <div className="text-center py-12">
          <div className="relative">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 text-blue-400 animate-pulse" />
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full blur-lg opacity-20 animate-pulse"></div>
          </div>
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Analiz Başlıyor...
          </h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            Kameranızı açın ve analiz sonuçlarını görün
          </p>
          <div className="mt-4 flex justify-center">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  return (
    <motion.div
      className={`bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-white/20 dark:border-gray-700/20 ${className}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white">Gerçek Zamanlı Analiz</h3>
        <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
          <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
          Canlı
        </div>
      </div>
      {}
      <div className="grid grid-cols-2 gap-8 mb-8">
        <CircularGauge
          value={analysisData.attention}
          label="Dikkat"
          icon={Eye}
          gradient={getGaugeColor(analysisData.attention)}
        />
        <CircularGauge
          value={analysisData.engagement}
          label="Katılım"
          icon={Users}
          gradient={getGaugeColor(analysisData.engagement)}
        />
      </div>
      {}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {}
        <div className="p-4 bg-white/30 dark:bg-gray-700/30 rounded-2xl">
          <div className="flex items-center mb-3">
            <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-red-600 rounded-xl flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div className="ml-3">
              <h4 className="font-semibold text-gray-900 dark:text-white">Duygu Durumu</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">AI Duygu Tanıma</p>
            </div>
          </div>
          <div className={`text-lg font-bold ${getEmotionColor(analysisData.emotion)} mb-1`}>
            {analysisData.emotion}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            {Math.round(analysisData.emotionConfidence)}% güvenilirlik
          </div>
        </div>
        {}
        <div className="p-4 bg-white/30 dark:bg-gray-700/30 rounded-2xl">
          <div className="flex items-center mb-3">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <div className="ml-3">
              <h4 className="font-semibold text-gray-900 dark:text-white">Bakış Yönü</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">Göz İzleme</p>
            </div>
          </div>
          <div className="text-lg font-bold text-purple-600 dark:text-purple-400 mb-1">
            {analysisData.gazeDirection}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            {analysisData.faceDetected ? 'Yüz tespit edildi' : 'Yüz tespit edilemedi'}
          </div>
        </div>
      </div>
      {}
      <div className="mt-6 pt-6 border-t border-gray-200/20 dark:border-gray-700/20">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center text-gray-600 dark:text-gray-400">
            <TrendingUp className="w-4 h-4 mr-2" />
            Son güncelleme: {new Date(analysisData.timestamp).toLocaleTimeString('tr-TR')}
          </div>
          <div className="flex space-x-4">
            <div className="text-green-600 dark:text-green-400">
              FER2013 • DAiSEE • ONNX
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
export default AnalysisDashboard;
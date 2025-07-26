import { motion } from 'framer-motion';
import {
    ArrowLeft,
    BarChart3,
    TrendingUp,
    Zap
} from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AnalysisDashboard from '../components/AnalysisDashboard';
import LiveCamera from '../components/LiveCamera';
import { useTheme } from '../contexts/ThemeContext';
import { AnalysisResult } from '../services/apiService';
interface StudentData {
  id: string;
  name: string;
  attention: number;
  engagement: number;
  emotion: string;
  gazeDirection: string;
  isActive: boolean;
}
const DemoPage = () => {
  const navigate = useNavigate();
  const { isDarkMode } = useTheme();
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [classData, setClassData] = useState<StudentData[]>([]);
  const [overallStats, setOverallStats] = useState({
    averageAttention: 0,
    averageEngagement: 0,
    activeStudents: 0,
    totalStudents: 0
  });
  const triggerTestAnalysis = useCallback(() => {
    console.log('Test analysis button clicked!');
    const mockResult: AnalysisResult = {
      attention: Math.round(Math.random() * 30 + 70),
      engagement: Math.round(Math.random() * 25 + 75),
      emotion: ['Odaklı', 'Meraklı', 'Anlıyor', 'Düşünüyor'][Math.floor(Math.random() * 4)],
      emotionConfidence: Math.round(Math.random() * 15 + 85),
      gazeDirection: ['Merkez', 'Sol', 'Sağ'][Math.floor(Math.random() * 3)],
      faceDetected: true,
      timestamp: Date.now()
    };
    console.log('Generated mock result:', mockResult);
    handleAnalysisResult(mockResult);
  }, []);
  useEffect(() => {
    const mockStudents: StudentData[] = Array.from({ length: 12 }, (_, i) => ({
      id: `student-${i + 1}`,
      name: `Öğrenci ${i + 1}`,
      attention: Math.random() * 40 + 60, 
      engagement: Math.random() * 30 + 70, 
      emotion: ['Odaklı', 'Meraklı', 'Anlıyor', 'Düşünüyor', 'Kararsız'][Math.floor(Math.random() * 5)],
      gazeDirection: ['Merkez', 'Sol', 'Sağ', 'Yukarı'][Math.floor(Math.random() * 4)],
      isActive: Math.random() > 0.1
    }));
    setClassData(mockStudents);
    const activeStudents = mockStudents.filter(s => s.isActive);
    const avgAttention = activeStudents.reduce((sum, s) => sum + s.attention, 0) / activeStudents.length;
    const avgEngagement = activeStudents.reduce((sum, s) => sum + s.engagement, 0) / activeStudents.length;
    setOverallStats({
      averageAttention: Math.round(avgAttention),
      averageEngagement: Math.round(avgEngagement),
      activeStudents: activeStudents.length,
      totalStudents: mockStudents.length
    });
    setTimeout(() => {
      console.log('Auto-triggering test analysis on page load...');
      triggerTestAnalysis();
    }, 2000);
  }, [triggerTestAnalysis]);
  const handleAnalysisResult = (result: AnalysisResult) => {
    console.log('Received analysis result in DemoPage:', result);
    setAnalysisData(result);
  };
  const getEmotionColor = (emotion: string) => {
    const colors: { [key: string]: string } = {
      'Odaklı': 'text-green-600',
      'Meraklı': 'text-blue-600',
      'Anlıyor': 'text-purple-600',
      'Düşünüyor': 'text-yellow-600',
      'Kararsız': 'text-orange-600',
      'Sıkılmış': 'text-red-600'
    };
    return colors[emotion] || 'text-gray-600';
  };
  const getAttentionColor = (attention: number) => {
    if (attention >= 80) return 'text-green-600';
    if (attention >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {}
      <header className="px-6 py-4 border-b border-gray-200/20 dark:border-gray-700/20 backdrop-blur-sm">
        <nav className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 rounded-lg bg-white/20 dark:bg-gray-800/20 backdrop-blur-sm border border-white/20 dark:border-gray-700/20 hover:bg-white/30 dark:hover:bg-gray-700/30 transition-all duration-300"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div className="flex items-center space-x-3">
              <img 
                src="/derslens-logo.png" 
                alt="Ders Lens Logo" 
                className="w-10 h-10 rounded-xl"
              />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Ders Lens Demo
              </h1>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={triggerTestAnalysis}
              className="px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-200 font-semibold shadow-lg"
            >
              🧪 Test Analizi Yap
            </button>
          </div>
        </nav>
      </header>
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {}
          <div className="lg:col-span-2 space-y-6">
            {}
            <LiveCamera onAnalysisResult={handleAnalysisResult} />
            {}
            <AnalysisDashboard analysisData={analysisData} />
          </div>
          {}
          <div className="space-y-6">
            {}
            <motion.div
              className="bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-white/20 dark:border-gray-700/20"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Sınıf Özeti</h3>
                <BarChart3 className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 dark:text-gray-300">Ortalama Dikkat</span>
                  <span className={`font-bold ${getAttentionColor(overallStats.averageAttention)}`}>
                    {overallStats.averageAttention}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 dark:text-gray-300">Ortalama Katılım</span>
                  <span className={`font-bold ${getAttentionColor(overallStats.averageEngagement)}`}>
                    {overallStats.averageEngagement}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 dark:text-gray-300">Aktif Öğrenci</span>
                  <span className="font-bold text-blue-600">
                    {overallStats.activeStudents}/{overallStats.totalStudents}
                  </span>
                </div>
              </div>
            </motion.div>
            {}
            <motion.div
              className="bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-white/20 dark:border-gray-700/20 max-h-96 overflow-y-auto"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Öğrenci Listesi</h3>
              <div className="space-y-3">
                {classData.slice(0, 8).map((student) => (
                  <div
                    key={student.id}
                    className="flex items-center justify-between p-3 bg-white/30 dark:bg-gray-700/30 rounded-xl"
                  >
                    <div className="flex items-center">
                      <div className={`w-3 h-3 rounded-full mr-3 ${student.isActive ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                      <span className="font-medium text-gray-900 dark:text-white text-sm">
                        {student.name}
                      </span>
                    </div>
                    <div className="text-right text-xs">
                      <div className={`font-semibold ${getAttentionColor(student.attention)}`}>
                        {Math.round(student.attention)}%
                      </div>
                      <div className={`${getEmotionColor(student.emotion)}`}>
                        {student.emotion}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
            {}
            <motion.div
              className="bg-gradient-to-r from-blue-600/10 to-purple-600/10 backdrop-blur-sm rounded-3xl p-6 border border-blue-200/20 dark:border-blue-700/20"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <div className="flex items-center mb-4">
                <Zap className="w-5 h-5 text-blue-600 mr-2" />
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">AI Modelleri</h3>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">Duygu Tanıma</span>
                  <span className="text-green-600 font-semibold">FER2013</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">Dikkat Analizi</span>
                  <span className="text-green-600 font-semibold">DAiSEE</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">Katılım Modeli</span>
                  <span className="text-green-600 font-semibold">Mendeley</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">ONNX Runtime</span>
                  <span className="text-green-600 font-semibold">Aktif</span>
                </div>
              </div>
            </motion.div>
            {}
            <motion.div
              className="bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-white/20 dark:border-gray-700/20"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <div className="flex items-center mb-4">
                <TrendingUp className="w-5 h-5 text-gray-600 dark:text-gray-300 mr-2" />
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">Performans</h3>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">Analiz Hızı</span>
                  <span className="text-green-600 font-semibold">&lt;100ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">Doğruluk Oranı</span>
                  <span className="text-green-600 font-semibold">95%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-300">CPU Kullanımı</span>
                  <span className="text-yellow-600 font-semibold">Orta</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default DemoPage;
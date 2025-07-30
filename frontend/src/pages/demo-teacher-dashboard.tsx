import { motion } from 'framer-motion';
import {
  AlertTriangle,
  Brain,
  Clock,
  Download,
  Eye,
  Heart,
  MessageSquare,
  Users,
  Video
} from 'lucide-react';
import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { useTranslation } from 'react-i18next';
import { Header } from '../components/layout/header';
import { Layout } from '../components/layout/layout';
import { GlassCard } from '../components/ui/glass-card';
import { NeonButton } from '../components/ui/neon-button';

// Demo chart components for teacher view
const ClassOverviewChart: React.FC = () => {
  const [data, setData] = useState<number[]>(Array.from({length: 15}, () => 60 + Math.random() * 40));
  
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const updated = [...prev];
        updated.shift();
        updated.push(60 + Math.random() * 40);
        return updated;
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-32 flex items-end gap-1">
      {data.map((value, index) => (
        <div
          key={index}
          className="flex-1 bg-gradient-to-t from-accent-cyan to-primary-500 rounded-t opacity-80 transition-all duration-500"
          style={{ height: `${value}%` }}
        />
      ))}
    </div>
  );
};

const StudentGrid: React.FC = () => {
  const students = [
    { name: 'Ahmet K.', attention: 95, engagement: 88, status: 'excellent' },
    { name: 'Zeynep A.', attention: 78, engagement: 92, status: 'good' },
    { name: 'Mehmet S.', attention: 65, engagement: 70, status: 'attention' },
    { name: 'Ayşe D.', attention: 89, engagement: 85, status: 'good' },
    { name: 'Fatma Y.', attention: 92, engagement: 95, status: 'excellent' },
    { name: 'Ali R.', attention: 58, engagement: 62, status: 'warning' },
    { name: 'Elif M.', attention: 87, engagement: 83, status: 'good' },
    { name: 'Burak T.', attention: 74, engagement: 78, status: 'attention' }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'border-emerald-500 bg-emerald-500/20';
      case 'good': return 'border-blue-500 bg-blue-500/20';
      case 'attention': return 'border-yellow-500 bg-yellow-500/20';
      case 'warning': return 'border-red-500 bg-red-500/20';
      default: return 'border-gray-500 bg-gray-500/20';
    }
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {students.map((student, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.1 }}
          className={`p-3 rounded-lg border-2 ${getStatusColor(student.status)} transition-all duration-300 hover:scale-105`}
        >
          <h4 className="text-white text-sm font-medium mb-2">{student.name}</h4>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Dikkat:</span>
              <span className="text-white">{student.attention}%</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Katılım:</span>
              <span className="text-white">{student.engagement}%</span>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

/**
 * Demo Teacher Dashboard - Comprehensive classroom management and analytics
 */
export const DemoTeacherDashboard: React.FC = () => {
  const { t } = useTranslation();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [classStats, setClassStats] = useState({
    totalStudents: 28,
    activeStudents: 26,
    avgAttention: 82,
    avgEngagement: 78,
    alerts: 3,
    questions: 12
  });

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      
      // Simulate fluctuating class stats
      setClassStats(prev => ({
        ...prev,
        avgAttention: Math.max(70, Math.min(95, prev.avgAttention + (Math.random() - 0.5) * 5)),
        avgEngagement: Math.max(65, Math.min(90, prev.avgEngagement + (Math.random() - 0.5) * 4)),
        questions: prev.questions + (Math.random() > 0.9 ? 1 : 0)
      }));
    }, 3000);

    return () => clearInterval(timer);
  }, []);

  const classInfo = {
    subject: "İleri Matematik - Diferansiyel Denklemler",
    duration: "50 dakika",
    startTime: "14:00",
    currentTopic: "Birinci Derece Lineer Diferansiyel Denklemler",
    nextTopic: "Değişkenlerine Ayrılabilen Denklemler"
  };

  const alerts = [
    { type: 'attention', message: 'Ali R. dikkat seviyesi düşük', time: '14:23' },
    { type: 'engagement', message: 'Mehmet S. katılım azaldı', time: '14:20' },
    { type: 'question', message: 'Yeni soru: "Türev nasıl alınır?"', time: '14:25' }
  ];

  const suggestions = [
    "Sınıf ortalaması çok iyi! Bu konuyu derinleştirmeyi düşünebilirsiniz.",
    "Bazı öğrenciler dikkat seviyesi düşük. Kısa bir mola verebilirsiniz.",
    "Katılım oranı yüksek. Bu fırsatı değerlendirerek soru sorabilirsiniz.",
    "Görsel materyaller öğrenci ilgisini artırabilir."
  ];

  return (
    <Layout>
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-8"
        >
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              Öğretmen Paneli
            </h1>
            <p className="text-gray-300">
              {classInfo.subject} - Canlı Sınıf Yönetimi
            </p>
          </div>
          
          <div className="flex items-center gap-4 mt-4 lg:mt-0">
            <div className="flex items-center gap-2 text-emerald-400">
              <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-sm font-medium">Ders Devam Ediyor</span>
            </div>
            <div className="text-gray-400 text-sm">
              {currentTime.toLocaleTimeString('tr-TR')}
            </div>
          </div>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8"
        >
          <GlassCard neonAccent className="p-4 text-center">
            <Users className="w-6 h-6 text-primary-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{classStats.totalStudents}</div>
            <div className="text-xs text-gray-400">Toplam Öğrenci</div>
          </GlassCard>

          <GlassCard neonAccent className="p-4 text-center">
            <Video className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{classStats.activeStudents}</div>
            <div className="text-xs text-gray-400">Aktif</div>
          </GlassCard>

          <GlassCard neonAccent className="p-4 text-center">
            <Eye className="w-6 h-6 text-accent-cyan mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{Math.round(classStats.avgAttention)}%</div>
            <div className="text-xs text-gray-400">Ort. Dikkat</div>
          </GlassCard>

          <GlassCard neonAccent className="p-4 text-center">
            <Heart className="w-6 h-6 text-accent-purple mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{Math.round(classStats.avgEngagement)}%</div>
            <div className="text-xs text-gray-400">Ort. Katılım</div>
          </GlassCard>

          <GlassCard neonAccent className="p-4 text-center">
            <AlertTriangle className="w-6 h-6 text-yellow-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{classStats.alerts}</div>
            <div className="text-xs text-gray-400">Uyarı</div>
          </GlassCard>

          <GlassCard neonAccent className="p-4 text-center">
            <MessageSquare className="w-6 h-6 text-accent-emerald mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{classStats.questions}</div>
            <div className="text-xs text-gray-400">Soru</div>
          </GlassCard>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Main Charts */}
          <div className="lg:col-span-2 space-y-8">
            {/* Class Overview */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <GlassCard neonAccent className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-white">
                    Sınıf Genel Durumu
                  </h3>
                  <div className="flex items-center gap-2 text-emerald-400">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                    <span className="text-sm">Canlı</span>
                  </div>
                </div>
                <ClassOverviewChart />
                <div className="mt-4 text-center text-gray-400 text-sm">
                  Son 45 dakika - Sınıf ortalama dikkat seviyesi
                </div>
              </GlassCard>
            </motion.div>

            {/* Student Grid */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <GlassCard neonAccent className="p-6">
                <h3 className="text-xl font-semibold text-white mb-6">
                  Öğrenci Durumu
                </h3>
                <StudentGrid />
              </GlassCard>
            </motion.div>
          </div>

          {/* Right Column - Controls & Info */}
          <div className="space-y-6">
            {/* Class Controls */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <GlassCard neonAccent className="p-6">
                <h3 className="text-xl font-semibold text-white mb-4">
                  Ders Kontrolleri
                </h3>
                <div className="space-y-3">
                  <NeonButton 
                    variant="primary" 
                    className="w-full justify-center"
                    onClick={() => toast('Mola özelliği yakında!')}
                  >
                    <Clock className="w-4 h-4 mr-2" />
                    Mola Ver
                  </NeonButton>
                  
                  <NeonButton 
                    variant="secondary" 
                    className="w-full justify-center"
                    onClick={() => toast('Anket özelliği yakında!')}
                  >
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Hızlı Anket
                  </NeonButton>

                  <NeonButton 
                    variant="secondary" 
                    className="w-full justify-center"
                    onClick={() => toast('Rapor özelliği yakında!')}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Rapor İndir
                  </NeonButton>
                </div>
              </GlassCard>
            </motion.div>

            {/* Alerts */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <GlassCard neonAccent className="p-6">
                <h3 className="text-xl font-semibold text-white mb-4">
                  Anlık Uyarılar
                </h3>
                <div className="space-y-3">
                  {alerts.map((alert, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 * index }}
                      className="p-3 rounded-lg bg-gradient-to-r from-orange-600/20 to-red-600/20 border border-orange-500/30"
                    >
                      <div className="flex items-start justify-between">
                        <p className="text-gray-300 text-sm flex-1">{alert.message}</p>
                        <span className="text-gray-400 text-xs ml-2">{alert.time}</span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </GlassCard>
            </motion.div>

            {/* AI Suggestions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <GlassCard neonAccent className="p-6">
                <h3 className="text-xl font-semibold text-white mb-4">
                  AI Öğretim Önerileri
                </h3>
                <div className="space-y-3">
                  {suggestions.map((suggestion, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 * index }}
                      className="flex items-start gap-3 p-3 rounded-lg bg-gradient-to-r from-primary-600/20 to-accent-cyan/20 border border-primary-500/30"
                    >
                      <Brain className="w-5 h-5 text-accent-cyan mt-0.5 flex-shrink-0" />
                      <p className="text-gray-300 text-sm">{suggestion}</p>
                    </motion.div>
                  ))}
                </div>
              </GlassCard>
            </motion.div>

            {/* Lesson Info */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <GlassCard neonAccent className="p-6">
                <h3 className="text-xl font-semibold text-white mb-4">
                  Ders Programı
                </h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-400">Şu Anki Konu:</span>
                    <p className="text-white mt-1">{classInfo.currentTopic}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Sıradaki Konu:</span>
                    <p className="text-white mt-1">{classInfo.nextTopic}</p>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t border-gray-600">
                    <span className="text-gray-400">Kalan Süre:</span>
                    <span className="text-accent-cyan font-medium">25 dakika</span>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </div>

        {/* Back Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 text-center"
        >
          <NeonButton
            variant="secondary"
            onClick={() => window.location.href = '/demo'}
            className="px-8 py-3"
          >
            Demo Ana Sayfasına Dön
          </NeonButton>
        </motion.div>
      </main>
    </Layout>
  );
};

export default DemoTeacherDashboard;

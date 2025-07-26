import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, Brain, CheckCircle, Clock, Eye, TrendingUp, Users } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Area, AreaChart, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { useAI } from '../../stores/aiStore';
const emotionData = [
  { name: 'Nötr', value: 35, color: '#6B7280' },
  { name: 'Mutlu', value: 28, color: '#10B981' },
  { name: 'Şaşırmış', value: 15, color: '#F59E0B' },
  { name: 'Üzgün', value: 8, color: '#3B82F6' },
  { name: 'Öfkeli', value: 6, color: '#EF4444' },
  { name: 'Korku', value: 4, color: '#8B5CF6' },
  { name: 'İğrenme', value: 3, color: '#F97316' },
  { name: 'Küçümseme', value: 1, color: '#EC4899' }
];
const attentionTrendData = [
  { time: '09:00', attention: 92, engagement: 88 },
  { time: '09:15', attention: 89, engagement: 85 },
  { time: '09:30', attention: 75, engagement: 72 },
  { time: '09:45', attention: 68, engagement: 65 },
  { time: '10:00', attention: 85, engagement: 82 },
  { time: '10:15', attention: 91, engagement: 89 },
  { time: '10:30', attention: 87, engagement: 84 }
];
const studentData = [
  { id: 1, name: 'Cansu Y.', attention: 94, emotion: 'Mutlu', engagement: 92, status: 'active' },
  { id: 2, name: 'Kemal K.', attention: 89, emotion: 'Nötr', engagement: 87, status: 'active' },
  { id: 3, name: 'Esra S.', attention: 76, emotion: 'Şaşırmış', engagement: 73, status: 'warning' },
  { id: 4, name: 'Fatima D.', attention: 92, emotion: 'Mutlu', engagement: 90, status: 'active' },
  { id: 5, name: 'Hazel R.', attention: 68, emotion: 'Üzgün', engagement: 65, status: 'alert' },
  { id: 6, name: 'Kemal A.', attention: 88, emotion: 'Nötr', engagement: 85, status: 'active' }
];
export function LiveDashboardSection() {
  const { state } = useAI();
  const [selectedTab, setSelectedTab] = useState<'overview' | 'students' | 'trends'>('overview');
  const [liveMetrics, setLiveMetrics] = useState({
    totalStudents: 24,
    activeStudents: 22,
    averageAttention: 84,
    averageEngagement: 81
  });
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveMetrics(prev => ({
        ...prev,
        averageAttention: Math.max(70, Math.min(95, prev.averageAttention + (Math.random() - 0.5) * 4)),
        averageEngagement: Math.max(65, Math.min(90, prev.averageEngagement + (Math.random() - 0.5) * 3))
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);
  const tabVariants = {
    inactive: { backgroundColor: "rgba(255,255,255,0.05)", color: "#9CA3AF" },
    active: { backgroundColor: "rgba(59,130,246,0.2)", color: "#FFFFFF" }
  };
  return (
    <section className="py-20 bg-black relative overflow-hidden">
      {}
      <div className="absolute inset-0">
        <motion.div
          className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-purple-900/20 to-cyan-900/20"
          animate={{
            backgroundPosition: ["0% 0%", "100% 100%"],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
        {}
        <div className="absolute inset-0 opacity-10">
          <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
            <defs>
              <pattern id="dashboardGrid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="100" height="100" fill="url(#dashboardGrid)" />
          </svg>
        </div>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <motion.div
          className="text-center mb-12"
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
              Canlı Sınıf
            </span>
            <br />
            <span className="text-white">Analiz Paneli</span>
          </motion.h2>
          <motion.p 
            className="text-xl text-gray-300 max-w-3xl mx-auto mb-8"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Gerçek zamanlı veri akışı ile sınıfınızın dinamiklerini anlık olarak takip edin.
          </motion.p>
          {}
          <motion.div
            className="inline-flex items-center bg-green-500/20 border border-green-500/30 rounded-full px-6 py-3"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            animate={state.isAnalyzing ? { scale: [1, 1.05, 1] } : {}}
          >
            <div className="w-3 h-3 bg-green-400 rounded-full mr-3 animate-pulse" />
            <span className="text-green-300 font-semibold">
              {state.isAnalyzing ? 'Canlı Analiz Aktif' : 'Analiz Hazır'}
            </span>
          </motion.div>
        </motion.div>
        {}
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
        >
          {[
            {
              icon: <Users className="w-6 h-6" />,
              label: 'Toplam Öğrenci',
              value: liveMetrics.totalStudents.toString(),
              change: '+2',
              color: 'from-blue-500 to-cyan-500'
            },
            {
              icon: <Eye className="w-6 h-6" />,
              label: 'Aktif Öğrenci',
              value: liveMetrics.activeStudents.toString(),
              change: '-1',
              color: 'from-green-500 to-teal-500'
            },
            {
              icon: <Brain className="w-6 h-6" />,
              label: 'Ortalama Dikkat',
              value: `%${Math.round(liveMetrics.averageAttention)}`,
              change: '+3%',
              color: 'from-purple-500 to-pink-500'
            },
            {
              icon: <TrendingUp className="w-6 h-6" />,
              label: 'Ortalama Katılım',
              value: `%${Math.round(liveMetrics.averageEngagement)}`,
              change: '+2%',
              color: 'from-orange-500 to-red-500'
            }
          ].map((metric, index) => (
            <motion.div
              key={metric.label}
              className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
              viewport={{ once: true }}
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${metric.color} p-2.5 mb-4`}>
                <div className="text-white w-full h-full flex items-center justify-center">
                  {metric.icon}
                </div>
              </div>
              <h3 className="text-gray-400 text-sm mb-2">{metric.label}</h3>
              <div className="flex items-end justify-between">
                <motion.span
                  className="text-3xl font-bold text-white"
                  key={metric.value}
                  initial={{ scale: 1.2, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {metric.value}
                </motion.span>
                <span className="text-green-400 text-sm font-medium">
                  {metric.change}
                </span>
              </div>
            </motion.div>
          ))}
        </motion.div>
        {}
        <motion.div
          className="flex justify-center mb-8"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="bg-white/5 backdrop-blur-md rounded-2xl p-2 border border-white/10">
            {[
              { id: 'overview', label: 'Genel Bakış', icon: <BarChart className="w-4 h-4" /> },
              { id: 'students', label: 'Öğrenci Detayı', icon: <Users className="w-4 h-4" /> },
              { id: 'trends', label: 'Trend Analizi', icon: <TrendingUp className="w-4 h-4" /> }
            ].map((tab) => (
              <motion.button
                key={tab.id}
                className="flex items-center px-6 py-3 rounded-xl font-medium transition-all duration-300"
                variants={tabVariants}
                animate={selectedTab === tab.id ? "active" : "inactive"}
                onClick={() => setSelectedTab(tab.id as any)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {tab.icon}
                <span className="ml-2">{tab.label}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>
        {}
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
          >
            {selectedTab === 'overview' && (
              <div className="grid lg:grid-cols-2 gap-8">
                {}
                <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-bold text-white mb-6">Duygu Dağılımı</h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={emotionData}
                          cx="50%"
                          cy="50%"
                          innerRadius={80}
                          outerRadius={120}
                          dataKey="value"
                          stroke="none"
                        >
                          {emotionData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="grid grid-cols-4 gap-2 mt-4">
                    {emotionData.slice(0, 4).map((emotion) => (
                      <div key={emotion.name} className="text-center">
                        <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ backgroundColor: emotion.color }} />
                        <div className="text-xs text-gray-400">{emotion.name}</div>
                        <div className="text-sm font-semibold text-white">{emotion.value}%</div>
                      </div>
                    ))}
                  </div>
                </div>
                {}
                <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-bold text-white mb-6">Dikkat Isı Haritası</h3>
                  <div className="grid grid-cols-6 gap-2">
                    {Array.from({ length: 24 }).map((_, index) => {
                      const attention = Math.random() * 100;
                      const intensity = attention / 100;
                      return (
                        <motion.div
                          key={index}
                          className="aspect-square rounded-lg"
                          style={{
                            backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                            border: '1px solid rgba(255,255,255,0.1)'
                          }}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: index * 0.05 }}
                          whileHover={{ scale: 1.1 }}
                        />
                      );
                    })}
                  </div>
                  <div className="flex justify-between items-center mt-4 text-sm text-gray-400">
                    <span>Düşük Dikkat</span>
                    <span>Yüksek Dikkat</span>
                  </div>
                </div>
              </div>
            )}
            {selectedTab === 'students' && (
              <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
                <h3 className="text-xl font-bold text-white mb-6">Öğrenci Detay Analizi</h3>
                <div className="space-y-4">
                  {studentData.map((student, index) => (
                    <motion.div
                      key={student.id}
                      className="bg-white/5 rounded-xl p-4 border border-white/10"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                            student.status === 'active' ? 'bg-green-500/20 border border-green-500/30' :
                            student.status === 'warning' ? 'bg-yellow-500/20 border border-yellow-500/30' :
                            'bg-red-500/20 border border-red-500/30'
                          }`}>
                            {student.status === 'active' ? <CheckCircle className="w-5 h-5 text-green-400" /> :
                             student.status === 'warning' ? <Clock className="w-5 h-5 text-yellow-400" /> :
                             <AlertCircle className="w-5 h-5 text-red-400" />}
                          </div>
                          <div>
                            <h4 className="text-white font-medium">{student.name}</h4>
                            <p className="text-gray-400 text-sm">Duygu: {student.emotion}</p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-right">
                          <div>
                            <div className="text-sm text-gray-400">Dikkat</div>
                            <div className="text-lg font-semibold text-white">{student.attention}%</div>
                          </div>
                          <div>
                            <div className="text-sm text-gray-400">Katılım</div>
                            <div className="text-lg font-semibold text-white">{student.engagement}%</div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
            {selectedTab === 'trends' && (
              <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
                <h3 className="text-xl font-bold text-white mb-6">Trend Analizi - Son 2 Saat</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={attentionTrendData}>
                      <defs>
                        <linearGradient id="attentionGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="engagementGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="time" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Area
                        type="monotone"
                        dataKey="attention"
                        stroke="#3B82F6"
                        fillOpacity={1}
                        fill="url(#attentionGradient)"
                        strokeWidth={3}
                      />
                      <Area
                        type="monotone"
                        dataKey="engagement"
                        stroke="#10B981"
                        fillOpacity={1}
                        fill="url(#engagementGradient)"
                        strokeWidth={3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex justify-center space-x-8 mt-4">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 rounded mr-2" />
                    <span className="text-gray-300">Dikkat Seviyesi</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-green-500 rounded mr-2" />
                    <span className="text-gray-300">Katılım Seviyesi</span>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
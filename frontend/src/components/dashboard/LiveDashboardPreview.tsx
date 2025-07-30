import { motion } from 'framer-motion';
import { Eye, Heart, TrendingUp, Users } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, XAxis, YAxis } from 'recharts';

const LiveDashboardPreview = () => {
  const [liveData, setLiveData] = useState({
    attention: 85,
    engagement: 78,
    activeStudents: 24,
    totalStudents: 28,
    emotions: [
      { name: 'Mutlu', value: 12, color: '#10B981' },
      { name: 'Odaklanmış', value: 8, color: '#3B82F6' },
      { name: 'Nötr', value: 6, color: '#6B7280' },
      { name: 'Karışık', value: 2, color: '#F59E0B' },
    ]
  });

  const [timeSeriesData] = useState([
    { time: '09:00', attention: 85, engagement: 78 },
    { time: '09:15', attention: 82, engagement: 80 },
    { time: '09:30', attention: 88, engagement: 75 },
    { time: '09:45', attention: 90, engagement: 88 },
    { time: '10:00', attention: 87, engagement: 85 },
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveData(prev => ({
        ...prev,
        attention: Math.max(70, Math.min(95, prev.attention + (Math.random() - 0.5) * 10)),
        engagement: Math.max(65, Math.min(90, prev.engagement + (Math.random() - 0.5) * 8)),
        activeStudents: Math.max(20, Math.min(28, prev.activeStudents + Math.floor((Math.random() - 0.5) * 4))),
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="py-20 bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-3xl font-bold text-gray-900 dark:text-white mb-4"
          >
            Canlı Dashboard Önizleme
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-gray-600 dark:text-gray-300"
          >
            Gerçek zamanlı öğrenci verilerini görün ve sınıfınızı optimize edin
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-md rounded-3xl p-8 border border-gray-200/20 dark:border-gray-700/20 shadow-2xl"
        >
          {/* Live Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-2xl p-6 border border-blue-200/20 dark:border-blue-700/20"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Ortalama Dikkat</p>
                  <motion.p 
                    key={liveData.attention}
                    initial={{ scale: 1.2 }}
                    animate={{ scale: 1 }}
                    className="text-3xl font-bold text-blue-700 dark:text-blue-300"
                  >
                    %{Math.round(liveData.attention)}
                  </motion.p>
                </div>
                <Eye className="h-8 w-8 text-blue-500" />
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-2xl p-6 border border-purple-200/20 dark:border-purple-700/20"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-600 dark:text-purple-400">Ortalama Katılım</p>
                  <motion.p 
                    key={liveData.engagement}
                    initial={{ scale: 1.2 }}
                    animate={{ scale: 1 }}
                    className="text-3xl font-bold text-purple-700 dark:text-purple-300"
                  >
                    %{Math.round(liveData.engagement)}
                  </motion.p>
                </div>
                <TrendingUp className="h-8 w-8 text-purple-500" />
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-2xl p-6 border border-green-200/20 dark:border-green-700/20"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-600 dark:text-green-400">Aktif Öğrenci</p>
                  <motion.p 
                    key={liveData.activeStudents}
                    initial={{ scale: 1.2 }}
                    animate={{ scale: 1 }}
                    className="text-3xl font-bold text-green-700 dark:text-green-300"
                  >
                    {liveData.activeStudents}/{liveData.totalStudents}
                  </motion.p>
                </div>
                <Users className="h-8 w-8 text-green-500" />
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-2xl p-6 border border-orange-200/20 dark:border-orange-700/20"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-600 dark:text-orange-400">Ruh Hali</p>
                  <p className="text-2xl font-bold text-orange-700 dark:text-orange-300 flex items-center">
                    😊 Pozitif
                  </p>
                </div>
                <Heart className="h-8 w-8 text-orange-500" />
              </div>
            </motion.div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Time Series Chart */}
            <div className="bg-white/50 dark:bg-gray-900/50 rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Zaman İçinde Değişim
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={timeSeriesData}>
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Line 
                    type="monotone" 
                    dataKey="attention" 
                    stroke="#3B82F6" 
                    strokeWidth={3}
                    dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="engagement" 
                    stroke="#8B5CF6" 
                    strokeWidth={3}
                    dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Emotion Distribution */}
            <div className="bg-white/50 dark:bg-gray-900/50 rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Duygu Dağılımı
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={liveData.emotions}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {liveData.emotions.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Student Grid Preview */}
          <div className="mt-8 bg-white/50 dark:bg-gray-900/50 rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Öğrenci Önizleme
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {Array.from({ length: 12 }, (_, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: i * 0.05 }}
                  whileHover={{ scale: 1.05 }}
                  className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-center border border-gray-200 dark:border-gray-700"
                >
                  <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full mx-auto mb-2 flex items-center justify-center">
                    <span className="text-white text-xs font-medium">{i + 1}</span>
                  </div>
                  <div className="space-y-1">
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                      <div 
                        className="bg-blue-500 h-1 rounded-full transition-all duration-300"
                        style={{ width: `${Math.floor(Math.random() * 40) + 60}%` }}
                      ></div>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                      <div 
                        className="bg-purple-500 h-1 rounded-full transition-all duration-300"
                        style={{ width: `${Math.floor(Math.random() * 40) + 60}%` }}
                      ></div>
                    </div>
                  </div>
                  <span className="text-lg">
                    {['😊', '🎯', '😐', '😕'][Math.floor(Math.random() * 4)]}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Live Indicator */}
          <div className="mt-6 flex items-center justify-center">
            <div className="flex items-center space-x-2 bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300 px-4 py-2 rounded-full">
              <motion.div
                className="w-2 h-2 bg-red-500 rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
              <span className="text-sm font-medium">Canlı Veri Akışı</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default LiveDashboardPreview;

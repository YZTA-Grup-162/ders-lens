import { motion } from 'framer-motion';
import {
    Eye,
    Heart,
    TrendingUp,
    Users
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface Student {
  id: string;
  name: string;
  attention: number;
  engagement: number;
  emotion: string;
  avatar: string;
  status: 'active' | 'inactive' | 'away';
}

interface ClassMetrics {
  averageAttention: number;
  averageEngagement: number;
  activeStudents: number;
  totalStudents: number;
  dominantEmotion: string;
}

const TeacherDashboard = () => {
  const [students, setStudents] = useState<Student[]>([]);
  const [classMetrics, setClassMetrics] = useState<ClassMetrics>({
    averageAttention: 0,
    averageEngagement: 0,
    activeStudents: 0,
    totalStudents: 0,
    dominantEmotion: 'neutral'
  });

  const [timeSeriesData, setTimeSeriesData] = useState([
    { time: '09:00', attention: 85, engagement: 78 },
    { time: '09:15', attention: 82, engagement: 80 },
    { time: '09:30', attention: 78, engagement: 75 },
    { time: '09:45', attention: 88, engagement: 85 },
    { time: '10:00', attention: 90, engagement: 88 },
  ]);

  useEffect(() => {
    const generateMockStudents = (): Student[] => {
      const names = [
        'Ahmet Yılmaz', 'Elif Kaya', 'Mehmet Demir', 'Ayşe Öz', 'Burak Çelik',
        'Zeynep Akar', 'Emre Güler', 'Seda Koç', 'Oğuz Arslan', 'Deniz Çetin',
        'Fatma Şen', 'Ali Yıldız'
      ];
      
      return names.map((name, index) => ({
        id: `student-${index}`,
        name,
        attention: Math.floor(Math.random() * 40) + 60, // 60-100
        engagement: Math.floor(Math.random() * 40) + 60, // 60-100
        emotion: ['happy', 'focused', 'neutral', 'confused'][Math.floor(Math.random() * 4)],
        avatar: `https://i.pravatar.cc/150?img=${index + 1}`,
        status: Math.random() > 0.1 ? 'active' : 'inactive'
      }));
    };

    const updateData = () => {
      const mockStudents = generateMockStudents();
      setStudents(mockStudents);

      const activeStudents = mockStudents.filter(s => s.status === 'active');
      const avgAttention = activeStudents.reduce((sum, s) => sum + s.attention, 0) / activeStudents.length;
      const avgEngagement = activeStudents.reduce((sum, s) => sum + s.engagement, 0) / activeStudents.length;
      
      const emotionCounts: { [key: string]: number } = {};
      activeStudents.forEach(s => {
        emotionCounts[s.emotion] = (emotionCounts[s.emotion] || 0) + 1;
      });
      
      const dominantEmotion = Object.keys(emotionCounts).reduce((a, b) => 
        emotionCounts[a] > emotionCounts[b] ? a : b, 'neutral'
      );

      setClassMetrics({
        averageAttention: Math.round(avgAttention),
        averageEngagement: Math.round(avgEngagement),
        activeStudents: activeStudents.length,
        totalStudents: mockStudents.length,
        dominantEmotion
      });
    };

    updateData();
    const interval = setInterval(updateData, 3000);
    return () => clearInterval(interval);
  }, []);

  const getEmotionColor = (emotion: string) => {
    switch (emotion) {
      case 'happy': return 'bg-green-100 text-green-800 border-green-200';
      case 'focused': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'confused': return 'bg-orange-100 text-orange-800 border-orange-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getEmotionEmoji = (emotion: string) => {
    switch (emotion) {
      case 'happy': return '😊';
      case 'focused': return '🎯';
      case 'confused': return '😕';
      default: return '😐';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'away': return 'bg-yellow-500';
      default: return 'bg-gray-400';
    }
  };

  const emotionData = [
    { name: 'Mutlu', value: students.filter(s => s.emotion === 'happy').length, color: '#10B981' },
    { name: 'Odaklanmış', value: students.filter(s => s.emotion === 'focused').length, color: '#3B82F6' },
    { name: 'Kafası Karışık', value: students.filter(s => s.emotion === 'confused').length, color: '#F59E0B' },
    { name: 'Nötr', value: students.filter(s => s.emotion === 'neutral').length, color: '#6B7280' },
  ];

  return (
    <div className="min-h-screen pt-20 bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Öğretmen Paneli
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Sınıfınızın gerçek zamanlı analiz verilerini izleyin
          </p>
        </motion.div>

        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
          >
            <div className="flex items-center">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center">
                <Eye className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Ortalama Dikkat</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">%{classMetrics.averageAttention}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
          >
            <div className="flex items-center">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-xl flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Ortalama Katılım</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">%{classMetrics.averageEngagement}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
          >
            <div className="flex items-center">
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-xl flex items-center justify-center">
                <Users className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Aktif Öğrenci</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {classMetrics.activeStudents}/{classMetrics.totalStudents}
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
          >
            <div className="flex items-center">
              <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-xl flex items-center justify-center">
                <Heart className="h-6 w-6 text-orange-600 dark:text-orange-400" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Genel Ruh Hali</p>
                <p className="text-lg font-bold text-gray-900 dark:text-white flex items-center">
                  {getEmotionEmoji(classMetrics.dominantEmotion)}
                  <span className="ml-2 text-sm">
                    {classMetrics.dominantEmotion === 'happy' && 'Mutlu'}
                    {classMetrics.dominantEmotion === 'focused' && 'Odaklanmış'}
                    {classMetrics.dominantEmotion === 'confused' && 'Karışık'}
                    {classMetrics.dominantEmotion === 'neutral' && 'Nötr'}
                  </span>
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Student List */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
            >
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
                Öğrenci Listesi
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                {students.map((student, index) => (
                  <motion.div
                    key={student.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className="relative">
                        <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full flex items-center justify-center">
                          <span className="text-white text-sm font-medium">
                            {student.name.split(' ').map(n => n[0]).join('')}
                          </span>
                        </div>
                        <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(student.status)}`}></div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {student.name}
                        </p>
                        <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getEmotionColor(student.emotion)}`}>
                          {getEmotionEmoji(student.emotion)} 
                          <span className="ml-1">
                            {student.emotion === 'happy' && 'Mutlu'}
                            {student.emotion === 'focused' && 'Odaklanmış'}
                            {student.emotion === 'confused' && 'Karışık'}
                            {student.emotion === 'neutral' && 'Nötr'}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                          <span>Dikkat</span>
                          <span>%{student.attention}</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${student.attention}%` }}
                          ></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                          <span>Katılım</span>
                          <span>%{student.engagement}</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${student.engagement}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Charts */}
          <div className="space-y-6">
            {/* Time Series Chart */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Zaman İçinde Değişim
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="attention" stroke="#3B82F6" strokeWidth={2} />
                  <Line type="monotone" dataKey="engagement" stroke="#8B5CF6" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Emotion Distribution */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
              className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Duygu Dağılımı
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={emotionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {emotionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2">
                {emotionData.map((emotion, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <div className="flex items-center">
                      <div
                        className="w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: emotion.color }}
                      ></div>
                      <span className="text-gray-700 dark:text-gray-300">{emotion.name}</span>
                    </div>
                    <span className="font-medium text-gray-900 dark:text-white">{emotion.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeacherDashboard;

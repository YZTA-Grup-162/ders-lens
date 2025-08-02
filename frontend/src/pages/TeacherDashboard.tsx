import { motion } from 'framer-motion';
import React, { useEffect, useState } from 'react';
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const TeacherDashboard: React.FC = () => {
  const [students, setStudents] = useState([
    { id: 1, name: 'Başak Dilara Çevik', status: 'active', emotion: 'focused', attentionLevel: 95, heartRate: 72 },
    { id: 2, name: 'Süleyman Kayyum Buberka', status: 'active', emotion: 'happy', attentionLevel: 88, heartRate: 68 },
    { id: 3, name: 'Enes Yıldırım', status: 'idle', emotion: 'neutral', attentionLevel: 75, heartRate: 74 },
    { id: 4, name: 'Hümeyra Betül Şahin', status: 'active', emotion: 'focused', attentionLevel: 92, heartRate: 70 },
    { id: 5, name: 'Muhammed Enes Güler', status: 'active', emotion: 'happy', attentionLevel: 90, heartRate: 66 },
    { id: 6, name: 'Nova Sterling', status: 'active', emotion: 'focused', attentionLevel: 94, heartRate: 69 },
    { id: 7, name: 'Zara Quantum', status: 'idle', emotion: 'confused', attentionLevel: 65, heartRate: 78 },
    { id: 8, name: 'Kai Nebula', status: 'active', emotion: 'happy', attentionLevel: 87, heartRate: 71 },
    { id: 9, name: 'Luna Phoenix', status: 'active', emotion: 'focused', attentionLevel: 89, heartRate: 67 },
    { id: 10, name: 'Alex Prism', status: 'active', emotion: 'neutral', attentionLevel: 82, heartRate: 73 },
    { id: 11, name: 'Maya Vector', status: 'idle', emotion: 'confused', attentionLevel: 70, heartRate: 76 },
    { id: 12, name: 'Rio Cosmos', status: 'active', emotion: 'happy', attentionLevel: 86, heartRate: 68 }
  ]);

  const [timeSeriesData, setTimeSeriesData] = useState([
    { time: '09:00', attention: 75, focus: 80, neuralActivity: 65, stress: 30, cognitiveLoad: 45 },
    { time: '09:15', attention: 78, focus: 82, neuralActivity: 70, stress: 28, cognitiveLoad: 48 },
    { time: '09:30', attention: 82, focus: 85, neuralActivity: 75, stress: 25, cognitiveLoad: 52 },
    { time: '09:45', attention: 88, focus: 90, neuralActivity: 82, stress: 22, cognitiveLoad: 58 },
    { time: '10:00', attention: 85, focus: 88, neuralActivity: 78, stress: 24, cognitiveLoad: 55 },
    { time: '10:15', attention: 90, focus: 92, neuralActivity: 85, stress: 20, cognitiveLoad: 62 },
    { time: '10:30', attention: 87, focus: 89, neuralActivity: 80, stress: 23, cognitiveLoad: 58 },
    { time: '10:45', attention: 92, focus: 95, neuralActivity: 88, stress: 18, cognitiveLoad: 65 }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setStudents(prevStudents => 
        prevStudents.map(student => ({
          ...student,
          attentionLevel: Math.max(0, Math.min(100, student.attentionLevel + (Math.random() - 0.5) * 10)),
          heartRate: Math.max(60, Math.min(100, student.heartRate + (Math.random() - 0.5) * 4))
        }))
      );

      setTimeSeriesData(prevData => {
        const newData = [...prevData.slice(1)];
        const lastEntry = prevData[prevData.length - 1];
        const currentTime = new Date();
        const timeString = `${currentTime.getHours().toString().padStart(2, '0')}:${currentTime.getMinutes().toString().padStart(2, '0')}`;
        
        newData.push({
          time: timeString,
          attention: Math.max(0, Math.min(100, lastEntry.attention + (Math.random() - 0.5) * 8)),
          focus: Math.max(0, Math.min(100, lastEntry.focus + (Math.random() - 0.5) * 6)),
          neuralActivity: Math.max(0, Math.min(100, lastEntry.neuralActivity + (Math.random() - 0.5) * 10)),
          stress: Math.max(0, Math.min(100, lastEntry.stress + (Math.random() - 0.5) * 5)),
          cognitiveLoad: Math.max(0, Math.min(100, lastEntry.cognitiveLoad + (Math.random() - 0.5) * 7))
        });
        
        return newData;
      });
    }, 3000);

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
      case 'idle': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const averageAttention = Math.round(students.reduce((sum, student) => sum + student.attentionLevel, 0) / students.length);
  const activeStudents = students.filter(student => student.status === 'active').length;

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

        {/* Stats Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 h-24 flex items-center"
          >
            <div className="flex items-center w-full">
              <div className="bg-blue-100 dark:bg-blue-900 p-3 rounded-full">
                <svg className="w-6 h-6 text-blue-600 dark:text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Toplam Öğrenci</p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">{students.length}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 h-24 flex items-center"
          >
            <div className="flex items-center w-full">
              <div className="bg-green-100 dark:bg-green-900 p-3 rounded-full">
                <svg className="w-6 h-6 text-green-600 dark:text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Aktif Öğrenci</p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">{activeStudents}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 h-24 flex items-center"
          >
            <div className="flex items-center w-full">
              <div className="bg-purple-100 dark:bg-purple-900 p-3 rounded-full">
                <svg className="w-6 h-6 text-purple-600 dark:text-purple-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Ortalama Dikkat</p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">{averageAttention}%</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 h-24 flex items-center"
          >
            <div className="flex items-center w-full">
              <div className="bg-orange-100 dark:bg-orange-900 p-3 rounded-full">
                <svg className="w-6 h-6 text-orange-600 dark:text-orange-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Ders Süresi</p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">45 dk</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Neural Dynamics Matrix - En Üstte ve Büyük */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 shadow-2xl mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-3xl font-bold text-white mb-2 flex items-center">
                <svg className="w-8 h-8 mr-3 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z"/>
                  <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z"/>
                </svg>
                Sınıf Beyin Dalgaları Analizi
              </h3>
              <p className="text-purple-300 text-lg">📊 Öğrencilerin anlık zihinsel durumlarını görün</p>
            </div>
            <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-lg p-4 border border-purple-400/30">
              <div className="flex items-center">
                <div className="w-4 h-4 bg-green-400 rounded-full animate-pulse mr-2"></div>
                <span className="text-green-400 text-sm font-medium">CANLI</span>
              </div>
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={timeSeriesData}>
              <defs>
                <linearGradient id="attentionGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="focusGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="neuralGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#F59E0B" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis 
                dataKey="time" 
                stroke="#9CA3AF"
                fontSize={14}
                tickLine={false}
                label={{ value: '⏰ Zaman (Dakika)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
              />
              <YAxis 
                stroke="#9CA3AF"
                fontSize={14}
                tickLine={false}
                axisLine={false}
                label={{ value: '📈 Yüzde (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(17, 24, 39, 0.95)',
                  border: '1px solid rgba(139, 92, 246, 0.3)',
                  borderRadius: '12px',
                  backdropFilter: 'blur(10px)',
                  color: '#fff'
                }}
                labelStyle={{ color: '#E5E7EB', fontWeight: 'bold' }}
                formatter={(value: any, name: string) => [
                  `${typeof value === 'number' ? value.toFixed(2) : value}%`,
                  name === 'attention' ? '🧠 Dikkat Seviyesi' :
                  name === 'focus' ? '🎯 Odaklanma Derecesi' :
                  name === 'neuralActivity' ? '⚡ Beyin Aktivitesi' : name
                ]}
              />
              <Area
                type="monotone"
                dataKey="attention"
                stroke="#8B5CF6"
                strokeWidth={3}
                fillOpacity={1}
                fill="url(#attentionGradient)"
                name="attention"
              />
              <Area
                type="monotone"
                dataKey="focus"
                stroke="#06B6D4"
                strokeWidth={3}
                fillOpacity={1}
                fill="url(#focusGradient)"
                name="focus"
              />
              <Area
                type="monotone"
                dataKey="neuralActivity"
                stroke="#F59E0B"
                strokeWidth={3}
                fillOpacity={1}
                fill="url(#neuralGradient)"
                name="neuralActivity"
              />
            </AreaChart>
          </ResponsiveContainer>

          {/* Chart Legend */}
          <div className="mt-4 flex flex-wrap justify-center gap-6 text-sm">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-purple-500 rounded mr-2"></div>
              <span className="text-purple-300">🧠 Dikkat Seviyesi - Öğrencilerin ne kadar dikkatli olduğu</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-cyan-500 rounded mr-2"></div>
              <span className="text-cyan-300">🎯 Odaklanma Derecesi - Görevlere ne kadar odaklandıkları</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-amber-500 rounded mr-2"></div>
              <span className="text-amber-300">⚡ Beyin Aktivitesi - Zihinsel enerji ve aktiflik</span>
            </div>
          </div>

          {/* AI Insights Panel - Self Explaining */}
          <div className="mt-8 grid grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-gradient-to-r from-cyan-500/10 to-cyan-600/10 rounded-lg p-4 border border-cyan-400/20 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-cyan-400/20 rounded-full -mr-8 -mt-8"></div>
              <div className="text-cyan-400 text-xs font-medium flex items-center">
                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                DİKKAT SEVİYESİ
              </div>
              <div className="text-white text-2xl font-bold">
                {(timeSeriesData[timeSeriesData.length - 1]?.attention || 0).toFixed(2)}%
              </div>
              <div className="text-cyan-300 text-xs mt-1">Sınıf Ortalama Odaklanma</div>
            </div>
            
            <div className="bg-gradient-to-r from-purple-500/10 to-purple-600/10 rounded-lg p-4 border border-purple-400/20 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-purple-400/20 rounded-full -mr-8 -mt-8"></div>
              <div className="text-purple-400 text-xs font-medium flex items-center">
                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                AKIŞ DURUMU
              </div>
              <div className="text-white text-2xl font-bold">
                {(timeSeriesData[timeSeriesData.length - 1]?.focus || 0).toFixed(2)}%
              </div>
              <div className="text-purple-300 text-xs mt-1">Öğrenciler Görevde Odaklanmış</div>
            </div>
            
            <div className="bg-gradient-to-r from-pink-500/10 to-pink-600/10 rounded-lg p-4 border border-pink-400/20 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-pink-400/20 rounded-full -mr-8 -mt-8"></div>
              <div className="text-pink-400 text-xs font-medium flex items-center">
                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                </svg>
                ZİHİNSEL YÜK
              </div>
              <div className="text-white text-2xl font-bold">
                {(timeSeriesData[timeSeriesData.length - 1]?.cognitiveLoad || 0).toFixed(2)}%
              </div>
              <div className="text-pink-300 text-xs mt-1">Beyin Kapasitesi Kullanımı</div>
            </div>
            
            <div className="bg-gradient-to-r from-emerald-500/10 to-emerald-600/10 rounded-lg p-4 border border-emerald-400/20 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-emerald-400/20 rounded-full -mr-8 -mt-8"></div>
              <div className="text-emerald-400 text-xs font-medium flex items-center">
                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
                </svg>
                VERİMLİLİK
              </div>
              <div className="text-white text-2xl font-bold">
                {(((timeSeriesData[timeSeriesData.length - 1]?.attention || 0) + (timeSeriesData[timeSeriesData.length - 1]?.focus || 0)) / 2).toFixed(2)}%
              </div>
              <div className="text-emerald-300 text-xs mt-1">Genel Sınıf Performansı</div>
            </div>
          </div>
        </motion.div>

        {/* Student List and Emotion Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Student List - 2/3 genişlik */}
          <div className="lg:col-span-2 h-full">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg h-full flex flex-col"
            >
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
                Öğrenci Listesi
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 flex-1 overflow-y-auto">
                {students.map((student, index) => (
                  <motion.div
                    key={student.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-100 dark:border-gray-700 hover:shadow-md transition-shadow h-36 flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center min-w-0 flex-1">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(student.status)} mr-2 flex-shrink-0`}></div>
                        <h3 className="font-medium text-gray-900 dark:text-white text-xs truncate" title={student.name}>
                          {student.name.length > 15 ? student.name.substring(0, 15) + '...' : student.name}
                        </h3>
                      </div>
                      <span className="text-lg flex-shrink-0">{getEmotionEmoji(student.emotion)}</span>
                    </div>
                    
                    <div className="space-y-2 flex-1">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600 dark:text-gray-400 flex items-center">
                          <span className="mr-1">🧠</span>Dikkat
                        </span>
                        <span className="text-xs font-semibold text-gray-900 dark:text-white">
                          {student.attentionLevel.toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-500 relative overflow-hidden" 
                          style={{ width: `${student.attentionLevel}%` }}
                        >
                          <div className="absolute inset-0 bg-white/30 animate-pulse"></div>
                        </div>
                      </div>
                    </div>
                      
                    <div className="flex justify-between items-center text-xs mt-3">
                      <span className="text-gray-600 dark:text-gray-400 flex items-center">
                        <span className="mr-1">💓</span>Nabız: {student.heartRate.toFixed(0)}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs border ${getEmotionColor(student.emotion)} flex items-center flex-shrink-0`}>
                        <span className="mr-1">{getEmotionEmoji(student.emotion)}</span>
                        {student.emotion === 'happy' ? 'Mutlu' : 
                         student.emotion === 'focused' ? 'Odak' : 
                         student.emotion === 'confused' ? 'Karışık' : 'Nötr'}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Emotion Distribution - 1/3 genişlik */}
          <div className="lg:col-span-1 h-full">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg h-full flex flex-col"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <span className="mr-2">😊</span>
                Sınıf Ruh Hali Dağılımı
              </h3>
              <div className="text-center mb-4">
                <div className="text-3xl font-bold text-gray-900 dark:text-white">
                  {students.length}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Toplam Öğrenci</div>
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie
                    data={emotionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={90}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {emotionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: any, name: string) => [
                      `${value} öğrenci`,
                      name
                    ]}
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid rgba(0, 0, 0, 0.1)',
                      borderRadius: '8px',
                      fontSize: '14px'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-3">
                {emotionData.map((emotion, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <div className="flex items-center">
                      <div
                        className="w-4 h-4 rounded-full mr-3 shadow-sm"
                        style={{ backgroundColor: emotion.color }}
                      ></div>
                      <span className="text-gray-700 dark:text-gray-300 flex items-center">
                        <span className="mr-1">
                          {emotion.name === 'Mutlu' ? '😊' :
                           emotion.name === 'Odaklanmış' ? '🎯' :
                           emotion.name === 'Kafası Karışık' ? '😕' : '😐'}
                        </span>
                        {emotion.name}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-gray-900 dark:text-white">{emotion.value}</span>
                      <span className="text-gray-500 text-xs ml-1">
                        ({((emotion.value / students.length) * 100).toFixed(1)}%)
                      </span>
                    </div>
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

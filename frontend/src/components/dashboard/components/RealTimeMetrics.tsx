import { motion } from 'framer-motion';
import { Activity, Brain, Clock, Eye, TrendingUp, Users, Zap } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
interface DashboardMetrics {
  totalStudents: number;
  activeStudents: number;
  averageAttention: number;
  averageEngagement: number;
  dominantEmotion: string;
  modelAccuracy: number;
  processingSpeed: number;
  frameRate: number;
}
interface RealTimeMetricsProps {
  metrics: DashboardMetrics;
  isLive: boolean;
}
const timeSeriesData = [
  { time: '14:00', attention: 92, engagement: 88, focus: 85 },
  { time: '14:05', attention: 89, engagement: 85, focus: 82 },
  { time: '14:10', attention: 75, engagement: 72, focus: 78 },
  { time: '14:15', attention: 68, engagement: 65, focus: 70 },
  { time: '14:20', attention: 85, engagement: 82, focus: 84 },
  { time: '14:25', attention: 91, engagement: 89, focus: 87 },
  { time: '14:30', attention: 87, engagement: 84, focus: 85 },
  { time: '14:35', attention: 93, engagement: 90, focus: 89 }
];
const performanceData = [
  { metric: 'Dikkat Tespiti', accuracy: 94.2, confidence: 89.7 },
  { metric: 'Duygu Tanıma', accuracy: 91.8, confidence: 86.3 },
  { metric: 'Katılım Analizi', accuracy: 88.9, confidence: 84.1 },
  { metric: 'Bakış Takibi', accuracy: 92.4, confidence: 87.9 }
];
export function RealTimeMetrics({ metrics, isLive }: RealTimeMetricsProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-black/80 backdrop-blur-md border border-white/20 rounded-lg p-3">
          <p className="text-white font-medium mb-2">{`Saat: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.name}: ${entry.value}%`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };
  return (
    <div className="space-y-6">
      {}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {}
        <motion.div
          className="lg:col-span-2 bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <TrendingUp className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Gerçek Zamanlı Metrikler</h3>
                <p className="text-gray-400 text-sm">Son 40 dakika</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-gray-500'}`} />
              <span className="text-sm text-gray-400">
                {isLive ? 'Canlı' : 'Demo'}
              </span>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeSeriesData}>
                <defs>
                  <linearGradient id="attentionGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="engagementGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="focusGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip content={<CustomTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="attention" 
                  stroke="#3B82F6" 
                  fillOpacity={1} 
                  fill="url(#attentionGradient)"
                  name="Dikkat"
                />
                <Area 
                  type="monotone" 
                  dataKey="engagement" 
                  stroke="#10B981" 
                  fillOpacity={1} 
                  fill="url(#engagementGradient)"
                  name="Katılım"
                />
                <Area 
                  type="monotone" 
                  dataKey="focus" 
                  stroke="#F59E0B" 
                  fillOpacity={1} 
                  fill="url(#focusGradient)"
                  name="Odaklanma"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
        {}
        <motion.div
          className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <Activity className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Anlık Durum</h3>
              <p className="text-gray-400 text-sm">Şu anki metrikler</p>
            </div>
          </div>
          <div className="space-y-4">
            {[
              { 
                label: 'Ortalama Dikkat', 
                value: `${metrics.averageAttention}%`, 
                icon: Eye, 
                color: 'blue',
                trend: '+2.1%'
              },
              { 
                label: 'Katılım Oranı', 
                value: `${metrics.averageEngagement}%`, 
                icon: Users, 
                color: 'green',
                trend: '+1.8%'
              },
              { 
                label: 'İşleme Hızı', 
                value: `${metrics.processingSpeed}ms`, 
                icon: Zap, 
                color: 'purple',
                trend: '-1.2ms'
              },
              { 
                label: 'FPS', 
                value: `${metrics.frameRate}`, 
                icon: Clock, 
                color: 'cyan',
                trend: 'Kararlı'
              }
            ].map((metric, index) => {
              const Icon = metric.icon;
              return (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 bg-${metric.color}-500/20 rounded-lg`}>
                      <Icon className={`w-4 h-4 text-${metric.color}-400`} />
                    </div>
                    <div>
                      <p className="text-white text-sm font-medium">{metric.label}</p>
                      <p className="text-gray-400 text-xs">{metric.trend}</p>
                    </div>
                  </div>
                  <span className="text-white font-bold">{metric.value}</span>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
      {}
      <motion.div
        className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Brain className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Model Performans Analizi</h3>
            <p className="text-gray-400 text-sm">AI modellerin detaylı performans metrikleri</p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {performanceData.map((item, index) => (
            <motion.div
              key={index}
              className="bg-white/5 border border-white/10 rounded-lg p-4 hover:border-white/20 transition-all duration-300"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ scale: 1.02, y: -2 }}
            >
              <div className="text-center">
                <h4 className="text-white font-medium text-sm mb-2">{item.metric}</h4>
                <div className="space-y-2">
                  <div>
                    <p className="text-gray-400 text-xs">Doğruluk</p>
                    <p className="text-blue-400 font-bold text-lg">{item.accuracy}%</p>
                    <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                      <motion.div
                        className="bg-blue-400 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${item.accuracy}%` }}
                        transition={{ duration: 1, delay: index * 0.2 }}
                      />
                    </div>
                  </div>
                  <div>
                    <p className="text-gray-400 text-xs">Güven</p>
                    <p className="text-green-400 font-bold">{item.confidence}%</p>
                    <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                      <motion.div
                        className="bg-green-400 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${item.confidence}%` }}
                        transition={{ duration: 1, delay: index * 0.2 + 0.5 }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
      {}
      <motion.div
        className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Sistem Sağlığı</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { 
              title: 'CPU Kullanımı', 
              value: '34%', 
              status: 'good', 
              description: 'Normal seviyede' 
            },
            { 
              title: 'GPU Kullanımı', 
              value: '67%', 
              status: 'warning', 
              description: 'Yoğun işleme' 
            },
            { 
              title: 'Bellek Kullanımı', 
              value: '2.8GB', 
              status: 'good', 
              description: '8GB üzerinden' 
            }
          ].map((health, index) => (
            <div key={index} className="text-center">
              <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-3 ${
                health.status === 'good' ? 'bg-green-500/20' : 'bg-yellow-500/20'
              }`}>
                <span className={`text-2xl font-bold ${
                  health.status === 'good' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  {health.value}
                </span>
              </div>
              <h4 className="text-white font-medium mb-1">{health.title}</h4>
              <p className="text-gray-400 text-sm">{health.description}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
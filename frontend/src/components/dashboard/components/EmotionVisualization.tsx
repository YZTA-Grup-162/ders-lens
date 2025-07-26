import { motion } from 'framer-motion';
import { AlertTriangle, Angry, Ban, Brain, Frown, Meh, Smile, Zap, type LucideIcon } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
interface EmotionData {
  name: string;
  value: number;
  color: string;
  icon: LucideIcon;
  turkish: string;
}
const emotionData: EmotionData[] = [
  { name: 'neutral', value: 35, color: '#6B7280', icon: Meh, turkish: 'Nötr' },
  { name: 'happiness', value: 28, color: '#10B981', icon: Smile, turkish: 'Mutlu' },
  { name: 'surprise', value: 15, color: '#F59E0B', icon: AlertTriangle, turkish: 'Şaşırmış' },
  { name: 'sadness', value: 8, color: '#3B82F6', icon: Frown, turkish: 'Üzgün' },
  { name: 'anger', value: 6, color: '#EF4444', icon: Angry, turkish: 'Öfkeli' },
  { name: 'fear', value: 4, color: '#8B5CF6', icon: Zap, turkish: 'Korku' },
  { name: 'disgust', value: 3, color: '#F97316', icon: Ban, turkish: 'İğrenme' },
  { name: 'contempt', value: 1, color: '#EC4899', icon: Brain, turkish: 'Küçümseme' }
];
interface EmotionVisualizationProps {
  emotions: any[];
  isLive: boolean;
}
export function EmotionVisualization({ emotions, isLive }: EmotionVisualizationProps) {
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-black/80 backdrop-blur-md border border-white/20 rounded-lg p-3">
          <p className="text-white font-medium">{data.turkish}</p>
          <p className="text-blue-400">{`${data.value}% öğrenci`}</p>
        </div>
      );
    }
    return null;
  };
  return (
    <div className="space-y-6">
      {}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-purple-500/20 rounded-lg">
            <Brain className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Duygu Analizi</h2>
            <p className="text-gray-400 text-sm">FER2013+ Modeli - 8 Duygu Sınıfı</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-gray-500'}`} />
          <span className="text-sm text-gray-400">
            {isLive ? 'Gerçek Zamanlı' : 'Demo Modu'}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {}
        <motion.div
          className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <h3 className="text-lg font-semibold text-white mb-4">Duygu Dağılımı</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value, turkish }) => `${turkish}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
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
          <h3 className="text-lg font-semibold text-white mb-4">Duygu İstatistikleri</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="turkish" 
                  stroke="#9CA3AF"
                  fontSize={12}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
      {}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {emotionData.map((emotion, index) => {
          const Icon = emotion.icon;
          return (
            <motion.div
              key={emotion.name}
              className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all duration-300"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.05 }}
              whileHover={{ scale: 1.02, y: -2 }}
            >
              <div className="flex items-center space-x-3 mb-3">
                <div 
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: `${emotion.color}20` }}
                >
                  <Icon size={20} color={emotion.color} />
                </div>
                <div>
                  <p className="font-medium text-white text-sm">{emotion.turkish}</p>
                  <p className="text-xs text-gray-400">{emotion.name}</p>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-white">{emotion.value}%</span>
                <div className="flex-1 ml-3">
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: emotion.color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${emotion.value}%` }}
                      transition={{ duration: 1, delay: index * 0.1 }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
      {}
      <motion.div
        className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Duygu Zaman Çizelgesi</h3>
        <div className="h-20 bg-gray-800/50 rounded-lg flex items-center justify-center">
          <p className="text-gray-400 text-sm">
            Gerçek zamanlı duygu değişim grafiği burada görüntülenecek
          </p>
        </div>
      </motion.div>
      {}
      <motion.div
        className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Model Performansı</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Genel Doğruluk', value: '94.2%', color: 'green' },
            { label: 'Güven Skoru', value: '89.7%', color: 'blue' },
            { label: 'İşleme Hızı', value: '23ms', color: 'purple' },
            { label: 'F1 Skoru', value: '0.91', color: 'cyan' }
          ].map((metric, index) => (
            <div key={index} className="text-center">
              <p className="text-gray-400 text-sm mb-1">{metric.label}</p>
              <p className={`text-xl font-bold text-${metric.color}-400`}>{metric.value}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
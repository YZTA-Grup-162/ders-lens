import { AnimatePresence, motion } from 'framer-motion';
import {
    Activity,
    AlertTriangle,
    Brain,
    CheckCircle,
    Clock,
    Download,
    Eye,
    Filter,
    Minus,
    Search,
    TrendingDown,
    TrendingUp,
    Users
} from 'lucide-react';
import { useState } from 'react';
interface Student {
  id: number;
  name: string;
  attention: number;
  engagement: number;
  emotion: string;
  gazeDirection: string;
  timeInSession: number;
  alertsCount: number;
  status: 'active' | 'warning' | 'alert';
  lastUpdate: string;
}
interface StudentGridProps {
  students: any[];
  isLive: boolean;
}
const mockStudents: Student[] = [
  {
    id: 1,
    name: 'Cansu Y.',
    attention: 94,
    engagement: 92,
    emotion: 'Mutlu',
    gazeDirection: 'Ekran',
    timeInSession: 45,
    alertsCount: 0,
    status: 'active',
    lastUpdate: '2 saniye önce'
  },
  {
    id: 2,
    name: 'Kemal K.',
    attention: 89,
    engagement: 87,
    emotion: 'Nötr',
    gazeDirection: 'Ekran',
    timeInSession: 45,
    alertsCount: 1,
    status: 'active',
    lastUpdate: '1 saniye önce'
  },
  {
    id: 3,
    name: 'Esra S.',
    attention: 76,
    engagement: 73,
    emotion: 'Şaşırmış',
    gazeDirection: 'Pencere',
    timeInSession: 45,
    alertsCount: 3,
    status: 'warning',
    lastUpdate: '3 saniye önce'
  },
  {
    id: 4,
    name: 'Fatima D.',
    attention: 92,
    engagement: 90,
    emotion: 'Mutlu',
    gazeDirection: 'Ekran',
    timeInSession: 45,
    alertsCount: 0,
    status: 'active',
    lastUpdate: '1 saniye önce'
  },
  {
    id: 5,
    name: 'Hazel R.',
    attention: 68,
    engagement: 65,
    emotion: 'Üzgün',
    gazeDirection: 'Ekran',
    timeInSession: 45,
    alertsCount: 5,
    status: 'alert',
    lastUpdate: '4 saniye önce'
  },
  {
    id: 6,
    name: 'Kemal A.',
    attention: 88,
    engagement: 85,
    emotion: 'Nötr',
    gazeDirection: 'Ekran',
    timeInSession: 45,
    alertsCount: 1,
    status: 'active',
    lastUpdate: '2 saniye önce'
  }
];
export function StudentGrid({ students, isLive }: StudentGridProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'warning' | 'alert'>('all');
  const [sortBy, setSortBy] = useState<'name' | 'attention' | 'engagement'>('attention');
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const filteredStudents = mockStudents
    .filter(student => 
      student.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
      (filterStatus === 'all' || student.status === filterStatus)
    )
    .sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      return b[sortBy] - a[sortBy];
    });
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'green';
      case 'warning': return 'yellow';
      case 'alert': return 'red';
      default: return 'gray';
    }
  };
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'alert': return AlertTriangle;
      default: return Minus;
    }
  };
  const getTrendIcon = (value: number) => {
    if (value >= 85) return TrendingUp;
    if (value >= 70) return Minus;
    return TrendingDown;
  };
  const getEmotionColor = (emotion: string) => {
    const colors: { [key: string]: string } = {
      'Mutlu': 'green',
      'Nötr': 'gray',
      'Şaşırmış': 'yellow',
      'Üzgün': 'blue',
      'Öfkeli': 'red',
      'Korku': 'purple',
      'İğrenme': 'orange',
      'Küçümseme': 'pink'
    };
    return colors[emotion] || 'gray';
  };
  return (
    <div className="space-y-6">
      {}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-blue-500/20 rounded-lg">
            <Users className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Katılımcı Detayları</h2>
            <p className="text-gray-400 text-sm">Online görüşme katılımcı analizi ve takip</p>
          </div>
        </div>
        {}
        <motion.button
          className="flex items-center space-x-2 px-4 py-2 bg-white/10 backdrop-blur-md rounded-lg border border-white/20 text-gray-300 hover:text-white hover:bg-white/20 transition-all duration-300"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Download className="w-4 h-4" />
          <span className="text-sm">Rapor İndir</span>
        </motion.button>
      </div>
      {}
      <div className="flex flex-col sm:flex-row gap-4 items-center">
        {}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Öğrenci ara..."
            className="w-full pl-10 pr-4 py-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-400 transition-all duration-300"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        {}
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <select
            className="bg-white/10 backdrop-blur-md border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-400 transition-all duration-300"
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value as any)}
          >
            <option value="all">Tüm Durumlar</option>
            <option value="active">Aktif</option>
            <option value="warning">Uyarı</option>
            <option value="alert">Alarm</option>
          </select>
        </div>
        {}
        <select
          className="bg-white/10 backdrop-blur-md border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-400 transition-all duration-300"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
        >
          <option value="attention">Dikkate Göre</option>
          <option value="engagement">Katılıma Göre</option>
          <option value="name">İsme Göre</option>
        </select>
      </div>
      {}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnimatePresence>
          {filteredStudents.map((student, index) => {
            const StatusIcon = getStatusIcon(student.status);
            const AttentionTrend = getTrendIcon(student.attention);
            const EngagementTrend = getTrendIcon(student.engagement);
            return (
              <motion.div
                key={student.id}
                className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all duration-300 cursor-pointer"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
                whileHover={{ scale: 1.02, y: -2 }}
                onClick={() => setSelectedStudent(student)}
              >
                {}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div 
                      className={`w-10 h-10 rounded-full bg-${getStatusColor(student.status)}-500/20 flex items-center justify-center`}
                    >
                      <span className="text-white font-bold text-sm">
                        {student.name.split(' ').map(n => n[0]).join('')}
                      </span>
                    </div>
                    <div>
                      <h3 className="text-white font-medium text-sm">{student.name}</h3>
                      <p className="text-gray-400 text-xs">#{student.id}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <StatusIcon className={`w-4 h-4 text-${getStatusColor(student.status)}-400`} />
                    <div className={`w-2 h-2 rounded-full bg-${getStatusColor(student.status)}-500 ${isLive ? 'animate-pulse' : ''}`} />
                  </div>
                </div>
                {}
                <div className="space-y-3">
                  {}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-blue-400" />
                      <span className="text-gray-400 text-sm">Dikkat</span>
                      <AttentionTrend className={`w-3 h-3 text-${getTrendIcon(student.attention) === TrendingUp ? 'green' : getTrendIcon(student.attention) === TrendingDown ? 'red' : 'gray'}-400`} />
                    </div>
                    <span className="text-white font-bold">{student.attention}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      className={`h-2 rounded-full bg-${student.attention >= 85 ? 'green' : student.attention >= 70 ? 'yellow' : 'red'}-400`}
                      initial={{ width: 0 }}
                      animate={{ width: `${student.attention}%` }}
                      transition={{ duration: 1, delay: index * 0.1 }}
                    />
                  </div>
                  {}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Activity className="w-4 h-4 text-green-400" />
                      <span className="text-gray-400 text-sm">Katılım</span>
                      <EngagementTrend className={`w-3 h-3 text-${getTrendIcon(student.engagement) === TrendingUp ? 'green' : getTrendIcon(student.engagement) === TrendingDown ? 'red' : 'gray'}-400`} />
                    </div>
                    <span className="text-white font-bold">{student.engagement}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      className={`h-2 rounded-full bg-${student.engagement >= 85 ? 'green' : student.engagement >= 70 ? 'yellow' : 'red'}-400`}
                      initial={{ width: 0 }}
                      animate={{ width: `${student.engagement}%` }}
                      transition={{ duration: 1, delay: index * 0.1 + 0.2 }}
                    />
                  </div>
                  {}
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-2">
                      <Brain className={`w-4 h-4 text-${getEmotionColor(student.emotion)}-400`} />
                      <span className="text-gray-400">Duygu:</span>
                    </div>
                    <span className={`text-${getEmotionColor(student.emotion)}-400 font-medium`}>
                      {student.emotion}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-cyan-400" />
                      <span className="text-gray-400">Bakış:</span>
                    </div>
                    <span className="text-cyan-400 font-medium">{student.gazeDirection}</span>
                  </div>
                </div>
                {}
                <div className="flex items-center justify-between mt-4 pt-3 border-t border-white/10">
                  <div className="flex items-center space-x-2">
                    <Clock className="w-3 h-3 text-gray-400" />
                    <span className="text-gray-400 text-xs">{student.lastUpdate}</span>
                  </div>
                  {student.alertsCount > 0 && (
                    <div className="flex items-center space-x-1">
                      <AlertTriangle className="w-3 h-3 text-red-400" />
                      <span className="text-red-400 text-xs">{student.alertsCount} uyarı</span>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
      {}
      <motion.div
        className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Sınıf Özeti</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            {
              label: 'Toplam Öğrenci',
              value: filteredStudents.length,
              icon: Users,
              color: 'blue'
            },
            {
              label: 'Ortalama Dikkat',
              value: `${Math.round(filteredStudents.reduce((sum, s) => sum + s.attention, 0) / filteredStudents.length)}%`,
              icon: Eye,
              color: 'green'
            },
            {
              label: 'Ortalama Katılım',
              value: `${Math.round(filteredStudents.reduce((sum, s) => sum + s.engagement, 0) / filteredStudents.length)}%`,
              icon: Activity,
              color: 'purple'
            },
            {
              label: 'Aktif Uyarılar',
              value: filteredStudents.reduce((sum, s) => sum + s.alertsCount, 0),
              icon: AlertTriangle,
              color: 'red'
            }
          ].map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div key={index} className="text-center">
                <div className={`p-3 bg-${stat.color}-500/20 rounded-lg mx-auto w-fit mb-2`}>
                  <Icon className={`w-6 h-6 text-${stat.color}-400`} />
                </div>
                <div className="text-white font-bold text-xl">{stat.value}</div>
                <div className="text-gray-400 text-sm">{stat.label}</div>
              </div>
            );
          })}
        </div>
      </motion.div>
      {}
      <AnimatePresence>
        {selectedStudent && (
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedStudent(null)}
          >
            <motion.div
              className="bg-gray-900 border border-white/20 rounded-xl p-6 max-w-md w-full"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="text-center mb-6">
                <div className={`w-16 h-16 mx-auto rounded-full bg-${getStatusColor(selectedStudent.status)}-500/20 flex items-center justify-center mb-3`}>
                  <span className="text-white font-bold text-lg">
                    {selectedStudent.name.split(' ').map(n => n[0]).join('')}
                  </span>
                </div>
                <h3 className="text-white font-bold text-xl">{selectedStudent.name}</h3>
                <p className="text-gray-400">Detaylı Öğrenci Analizi</p>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">{selectedStudent.attention}%</div>
                    <div className="text-gray-400 text-sm">Dikkat Seviyesi</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">{selectedStudent.engagement}%</div>
                    <div className="text-gray-400 text-sm">Katılım Oranı</div>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Duygu:</span>
                    <span className={`text-${getEmotionColor(selectedStudent.emotion)}-400 font-medium`}>
                      {selectedStudent.emotion}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Bakış Yönü:</span>
                    <span className="text-cyan-400 font-medium">{selectedStudent.gazeDirection}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Oturum Süresi:</span>
                    <span className="text-white font-medium">{selectedStudent.timeInSession} dakika</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Uyarı Sayısı:</span>
                    <span className="text-red-400 font-medium">{selectedStudent.alertsCount}</span>
                  </div>
                </div>
              </div>
              <button
                className="w-full mt-6 px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-all duration-300"
                onClick={() => setSelectedStudent(null)}
              >
                Kapat
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
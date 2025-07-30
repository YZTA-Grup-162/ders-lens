import { motion } from 'framer-motion';
import {
    AlertTriangle,
    BarChart3,
    Bell,
    Calendar,
    Camera,
    CheckCircle,
    Clock,
    Download,
    FileText,
    Filter,
    Heart,
    Home,
    LogOut,
    RefreshCw,
    Settings,
    Target,
    TrendingDown,
    TrendingUp,
    User
} from 'lucide-react';
import React, { useEffect, useState } from 'react';
const ModernTeacherDashboard: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [selectedStudent, setSelectedStudent] = useState<number | null>(null);
  const [classMode, setClassMode] = useState<'overview' | 'individual'>('overview');
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const teacherData = {
    name: "Enes Hümeyra Kayyum"
  };
  const classData = {
    className: "Matematik 101 - Diferansiyel Denklemler",
    totalStudents: 24,
    activeStudents: 22,
    sessionDuration: 45,
    overallStats: {
      avgAttention: 87,
      avgEngagement: 89,
      avgEmotion: 4.2,
      alertStudents: 3
    },
    students: [
      {
        id: 1,
        name: "Başak Avcı",
        attention: 94,
        engagement: 88,
        emotion: "4.5",
        status: 'good',
        gaze: 'center'
      },
      ...Array.from({ length: 23 }, (_, i) => ({
        id: i + 2,
        name: `Öğrenci ${i + 2}`,
        attention: Math.floor(Math.random() * 30) + 70,
        engagement: Math.floor(Math.random() * 30) + 70,
        emotion: (Math.random() * 2 + 3).toFixed(1),
        status: Math.random() > 0.8 ? 'alert' : Math.random() > 0.6 ? 'warning' : 'good',
        gaze: ['center', 'left', 'right', 'away'][Math.floor(Math.random() * 4)]
      }))
    ]
  };
  const StatCard = ({ icon: Icon, title, value, subtitle, trend, color, large = false }: any) => (
    <motion.div
      className={`bg-white rounded-2xl p-6 shadow-lg border border-slate-100 ${large ? 'md:col-span-2' : ''}`}
      whileHover={{ y: -2 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`w-12 h-12 bg-gradient-to-r ${color} rounded-xl flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        {trend !== undefined && (
          <div className={`flex items-center ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {trend >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
            <span className="text-sm font-medium">{Math.abs(trend)}%</span>
          </div>
        )}
      </div>
      <div className="text-3xl font-bold text-slate-900 mb-1">{value}</div>
      <div className="text-slate-600 text-sm">{title}</div>
      {subtitle && <div className="text-slate-500 text-xs mt-1">{subtitle}</div>}
    </motion.div>
  );
  const StudentCard = ({ student, index }: any) => {
    const getStatusColor = (status: string) => {
      switch (status) {
        case 'alert': return 'border-red-300 bg-red-50';
        case 'warning': return 'border-yellow-300 bg-yellow-50';
        default: return 'border-green-300 bg-green-50';
      }
    };
    const getStatusIcon = (status: string) => {
      switch (status) {
        case 'alert': return <AlertTriangle className="w-4 h-4 text-red-600" />;
        case 'warning': return <Clock className="w-4 h-4 text-yellow-600" />;
        default: return <CheckCircle className="w-4 h-4 text-green-600" />;
      }
    };
    return (
      <motion.div
        className={`border-2 rounded-xl p-4 cursor-pointer transition-all duration-300 ${getStatusColor(student.status)} ${
          selectedStudent === student.id ? 'ring-2 ring-blue-500' : ''
        }`}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: index * 0.05 }}
        whileHover={{ scale: 1.02 }}
        onClick={() => setSelectedStudent(selectedStudent === student.id ? null : student.id)}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full flex items-center justify-center">
              <span className="text-white text-xs font-bold">{student.id}</span>
            </div>
            <span className="font-medium text-slate-900">{student.name}</span>
          </div>
          {getStatusIcon(student.status)}
        </div>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-600">Dikkat</span>
            <span className={`text-sm font-medium ${
              student.attention >= 80 ? 'text-green-600' : 
              student.attention >= 60 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {student.attention}%
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-1.5">
            <div 
              className={`h-1.5 rounded-full ${
                student.attention >= 80 ? 'bg-green-400' : 
                student.attention >= 60 ? 'bg-yellow-400' : 'bg-red-400'
              }`}
              style={{ width: `${student.attention}%` }}
            ></div>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-600">Katılım</span>
            <span className={`text-sm font-medium ${
              student.engagement >= 80 ? 'text-green-600' : 
              student.engagement >= 60 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {student.engagement}%
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-1.5">
            <div 
              className={`h-1.5 rounded-full ${
                student.engagement >= 80 ? 'bg-green-400' : 
                student.engagement >= 60 ? 'bg-yellow-400' : 'bg-red-400'
              }`}
              style={{ width: `${student.engagement}%` }}
            ></div>
          </div>
          <div className="flex justify-between items-center mt-2">
            <span className="text-xs text-slate-600">Duygu</span>
            <span className="text-sm font-medium text-purple-600">{student.emotion}/5</span>
          </div>
        </div>
      </motion.div>
    );
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {}
      <nav className="fixed left-0 top-0 h-full w-64 bg-white/90 backdrop-blur-lg border-r border-slate-200 z-40">
        <div className="p-6">
          <div className="flex items-center space-x-4 mb-8">
            <img 
              src="/derslens-logo.png" 
              alt="DersLens Logo" 
              className="h-10 w-auto object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              DersLens
            </span>
          </div>
          <div className="space-y-2">
            <a href="#" className="flex items-center space-x-3 px-4 py-3 bg-blue-50 text-blue-600 rounded-lg">
              <Home className="w-5 h-5" />
              <span>Sınıf Dashboard</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <BarChart3 className="w-5 h-5" />
              <span>Analizler</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <FileText className="w-5 h-5" />
              <span>Raporlar</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <Calendar className="w-5 h-5" />
              <span>Geçmiş Dersler</span>
            </a>
            <a href="#" className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
              <Settings className="w-5 h-5" />
              <span>Ayarlar</span>
            </a>
          </div>
        </div>
        <div className="absolute bottom-6 left-6 right-6">
          <button className="flex items-center space-x-3 px-4 py-3 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors w-full">
            <LogOut className="w-5 h-5" />
            <span>Çıkış Yap</span>
          </button>
        </div>
      </nav>
      {}
      <div className="ml-64">
        {}
        <header className="bg-white/80 backdrop-blur-lg border-b border-slate-200 px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">{classData.className}</h1>
              <p className="text-slate-600">
                {teacherData.name} • {classData.activeStudents}/{classData.totalStudents} öğrenci aktif • 
                {currentTime.toLocaleTimeString('tr-TR')} • 
                {classData.sessionDuration} dk ders süresi
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex space-x-2">
                <button 
                  onClick={() => setClassMode('overview')}
                  className={`px-4 py-2 rounded-lg transition-colors ${
                    classMode === 'overview' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  Genel Bakış
                </button>
                <button 
                  onClick={() => setClassMode('individual')}
                  className={`px-4 py-2 rounded-lg transition-colors ${
                    classMode === 'individual' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  Bireysel
                </button>
              </div>
              <button className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                <RefreshCw className="w-5 h-5" />
              </button>
              <button className="relative p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                <Bell className="w-5 h-5" />
                {classData.overallStats.alertStudents > 0 && (
                  <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></div>
                )}
              </button>
              <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
            </div>
          </div>
        </header>
        {}
        <main className="p-8">
          {}
          <motion.div 
            className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-3xl p-8 mb-8 text-white"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-2">Canlı Ders Oturumu</h2>
                <p className="text-blue-100">
                  {classData.activeStudents} öğrenci aktif olarak takip ediliyor
                </p>
              </div>
              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <div className="text-3xl font-bold">{classData.sessionDuration}</div>
                  <div className="text-blue-100 text-sm">Dakika</div>
                </div>
                <div className="flex items-center text-green-300">
                  <Camera className="w-5 h-5 mr-2" />
                  <span>Kamera Aktif</span>
                </div>
              </div>
            </div>
          </motion.div>
          {}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              icon={Target}
              title="Ortalama Dikkat"
              value={`${classData.overallStats.avgAttention}%`}
              subtitle="Sınıf geneli"
              trend={3}
              color="from-blue-500 to-indigo-500"
            />
            <StatCard
              icon={TrendingUp}
              title="Ortalama Katılım"
              value={`${classData.overallStats.avgEngagement}%`}
              subtitle="Aktif öğrenciler"
              trend={5}
              color="from-green-500 to-emerald-500"
            />
            <StatCard
              icon={Heart}
              title="Duygu Skoru"
              value={classData.overallStats.avgEmotion}
              subtitle="5 üzerinden"
              trend={-1}
              color="from-purple-500 to-violet-500"
            />
            <StatCard
              icon={AlertTriangle}
              title="Uyarı Durumu"
              value={classData.overallStats.alertStudents}
              subtitle="Dikkat gerektiren öğrenci"
              trend={undefined}
              color="from-red-500 to-pink-500"
            />
          </div>
          {classMode === 'overview' ? (
            <>
              {}
              <div className="grid lg:grid-cols-3 gap-8 mb-8">
                {}
                <motion.div
                  className="lg:col-span-2 bg-white rounded-2xl p-6 shadow-lg border border-slate-100"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-slate-900">Sınıf Dikkat Haritası</h3>
                    <div className="flex space-x-2">
                      <button className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg">
                        <Filter className="w-4 h-4" />
                      </button>
                      <button className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg">
                        <Download className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-6 gap-3">
                    {classData.students.map((student, index) => (
                      <div
                        key={student.id}
                        className={`aspect-square rounded-lg flex items-center justify-center text-white text-xs font-bold cursor-pointer transition-all duration-300 ${
                          student.attention >= 80 ? 'bg-green-500 hover:bg-green-600' :
                          student.attention >= 60 ? 'bg-yellow-500 hover:bg-yellow-600' :
                          'bg-red-500 hover:bg-red-600'
                        }`}
                        title={`${student.name} - ${student.attention}% dikkat`}
                        onClick={() => setSelectedStudent(student.id)}
                      >
                        {student.id}
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center justify-center space-x-6 mt-6 text-sm">
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-green-500 rounded"></div>
                      <span className="text-slate-600">Yüksek (%80+)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                      <span className="text-slate-600">Orta (%60-79)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-red-500 rounded"></div>
                      <span className="text-slate-600">Düşük (%59-)</span>
                    </div>
                  </div>
                </motion.div>
                {}
                <motion.div
                  className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <h3 className="text-xl font-bold text-slate-900 mb-6">Canlı İçgörüler</h3>
                  <div className="space-y-6">
                    {}
                    <div>
                      <h4 className="font-medium text-slate-700 mb-3">Dikkat Dağılımı</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-slate-600">Yüksek</span>
                          <span className="text-sm font-medium">
                            {classData.students.filter(s => s.attention >= 80).length} öğrenci
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-slate-600">Orta</span>
                          <span className="text-sm font-medium">
                            {classData.students.filter(s => s.attention >= 60 && s.attention < 80).length} öğrenci
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-slate-600">Düşük</span>
                          <span className="text-sm font-medium text-red-600">
                            {classData.students.filter(s => s.attention < 60).length} öğrenci
                          </span>
                        </div>
                      </div>
                    </div>
                    {}
                    <div>
                      <h4 className="font-medium text-slate-700 mb-3">Uyarılar</h4>
                      <div className="space-y-2">
                        {classData.students
                          .filter(s => s.status === 'alert')
                          .slice(0, 3)
                          .map(student => (
                            <div key={student.id} className="flex items-center space-x-2 text-sm">
                              <AlertTriangle className="w-4 h-4 text-red-500" />
                              <span className="text-slate-600">{student.name}</span>
                              <span className="text-red-600">%{student.attention}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                    {}
                    <div className="bg-blue-50 rounded-xl p-4">
                      <h4 className="font-medium text-blue-900 mb-2">💡 Öneriler</h4>
                      <p className="text-blue-700 text-sm">
                        Sınıfın %{Math.round((classData.students.filter(s => s.attention >= 80).length / classData.students.length) * 100)} 
                        'i yüksek dikkat seviyesinde. Mevcut tempo kordurun.
                      </p>
                    </div>
                  </div>
                </motion.div>
              </div>
            </>
          ) : (
            <>
              {}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {classData.students.map((student, index) => (
                  <StudentCard key={student.id} student={student} index={index} />
                ))}
              </div>
            </>
          )}
          {}
          {selectedStudent && (
            <motion.div
              className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              onClick={() => setSelectedStudent(null)}
            >
              <motion.div
                className="bg-white rounded-2xl p-8 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                onClick={(e) => e.stopPropagation()}
              >
                {(() => {
                  const student = classData.students.find(s => s.id === selectedStudent);
                  if (!student) return null;
                  return (
                    <>
                      <div className="flex items-center justify-between mb-6">
                        <h3 className="text-2xl font-bold text-slate-900">{student.name}</h3>
                        <button 
                          onClick={() => setSelectedStudent(null)}
                          className="text-slate-600 hover:bg-slate-100 p-2 rounded-lg"
                        >
                          ✕
                        </button>
                      </div>
                      <div className="grid grid-cols-2 gap-6 mb-6">
                        <div className="bg-blue-50 rounded-xl p-4">
                          <div className="text-2xl font-bold text-blue-600 mb-1">{student.attention}%</div>
                          <div className="text-slate-600 text-sm">Dikkat Seviyesi</div>
                        </div>
                        <div className="bg-green-50 rounded-xl p-4">
                          <div className="text-2xl font-bold text-green-600 mb-1">{student.engagement}%</div>
                          <div className="text-slate-600 text-sm">Katılım Oranı</div>
                        </div>
                        <div className="bg-purple-50 rounded-xl p-4">
                          <div className="text-2xl font-bold text-purple-600 mb-1">{student.emotion}/5</div>
                          <div className="text-slate-600 text-sm">Duygu Skoru</div>
                        </div>
                        <div className="bg-orange-50 rounded-xl p-4">
                          <div className="text-2xl font-bold text-orange-600 mb-1 capitalize">{student.gaze}</div>
                          <div className="text-slate-600 text-sm">Göz Yönelimi</div>
                        </div>
                      </div>
                      <div className="bg-slate-50 rounded-xl p-4">
                        <h4 className="font-medium text-slate-700 mb-3">Öneriler</h4>
                        <ul className="space-y-2 text-sm text-slate-600">
                          <li>• {student.attention < 70 ? 'Öğrencinin dikkatini çekmek için interaktif sorular sorabilirsiniz' : 'Mevcut dikkat seviyesi optimal'}</li>
                          <li>• {student.engagement < 70 ? 'Grup aktivitelerine dahil etmeyi deneyin' : 'Katılım seviyesi yeterli'}</li>
                          <li>• {student.gaze === 'away' ? 'Öğrenci dikkatini başka yöne vermiş, nazikçe uyarabilirsiniz' : 'Göz kontağı iyi'}</li>
                        </ul>
                      </div>
                    </>
                  );
                })()}
              </motion.div>
            </motion.div>
          )}
        </main>
      </div>
    </div>
  );
};
export default ModernTeacherDashboard;
import { motion } from 'framer-motion';
import { Eye, Grid, Map, Target, Users } from 'lucide-react';
import { useState } from 'react';
interface Student {
  id: number;
  name: string;
  x: number;
  y: number;
  attention: number;
  gazeX: number;
  gazeY: number;
}
interface AttentionHeatmapProps {
  students: any[];
  isLive: boolean;
}
const mockStudents: Student[] = [
  { id: 1, name: 'Cansu Y.', x: 20, y: 30, attention: 94, gazeX: 50, gazeY: 20 },
  { id: 2, name: 'Ayşe K.', x: 40, y: 30, attention: 89, gazeX: 52, gazeY: 18 },
  { id: 3, name: 'Mehmet S.', x: 60, y: 30, attention: 76, gazeX: 30, gazeY: 40 },
  { id: 4, name: 'Fatma D.', x: 80, y: 30, attention: 92, gazeX: 48, gazeY: 22 },
  { id: 5, name: 'Ali R.', x: 20, y: 50, attention: 68, gazeX: 20, gazeY: 60 },
  { id: 6, name: 'Zehra M.', x: 40, y: 50, attention: 88, gazeX: 55, gazeY: 25 },
  { id: 7, name: 'Can T.', x: 60, y: 50, attention: 91, gazeX: 45, gazeY: 15 },
  { id: 8, name: 'Elif S.', x: 80, y: 50, attention: 85, gazeX: 50, gazeY: 30 },
  { id: 9, name: 'Murat K.', x: 20, y: 70, attention: 79, gazeX: 40, gazeY: 50 },
  { id: 10, name: 'Seda A.', x: 40, y: 70, attention: 93, gazeX: 48, gazeY: 20 },
  { id: 11, name: 'Burak M.', x: 60, y: 70, attention: 87, gazeX: 52, gazeY: 18 },
  { id: 12, name: 'Deniz L.', x: 80, y: 70, attention: 82, gazeX: 35, gazeY: 45 }
];
export function AttentionHeatmap({ students, isLive }: AttentionHeatmapProps) {
  const [viewMode, setViewMode] = useState<'attention' | 'gaze' | 'combined'>('attention');
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const getAttentionColor = (attention: number) => {
    if (attention >= 85) return '#10B981'; 
    if (attention >= 70) return '#F59E0B'; 
    return '#EF4444'; 
  };
  const getAttentionOpacity = (attention: number) => {
    return Math.max(0.3, attention / 100);
  };
  return (
    <div className="space-y-6">
      {}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-cyan-500/20 rounded-lg">
            <Eye className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Online Görüşme Dikkat Analizi</h2>
            <p className="text-gray-400 text-sm">Uzaktan eğitimde katılımcı dikkat dağılımı ve ekran odaklanması</p>
          </div>
        </div>
        {}
        <div className="flex items-center space-x-2 bg-white/5 backdrop-blur-md rounded-lg p-2 border border-white/10">
          {[
            { id: 'attention', label: 'Dikkat', icon: Target },
            { id: 'gaze', label: 'Bakış', icon: Eye },
            { id: 'combined', label: 'Kombine', icon: Grid }
          ].map((mode) => {
            const Icon = mode.icon;
            return (
              <motion.button
                key={mode.id}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ${
                  viewMode === mode.id
                    ? 'bg-cyan-500/30 text-cyan-400'
                    : 'text-gray-400 hover:text-white hover:bg-white/10'
                }`}
                onClick={() => setViewMode(mode.id as any)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Icon className="w-4 h-4" />
                <span>{mode.label}</span>
              </motion.button>
            );
          })}
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {}
        <motion.div
          className="lg:col-span-2 bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Sınıf Haritası</h3>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-gray-500'}`} />
              <span className="text-sm text-gray-400">
                {isLive ? 'Gerçek Zamanlı' : 'Demo Modu'}
              </span>
            </div>
          </div>
          {}
          <div className="relative bg-gray-900/50 rounded-lg p-4" style={{ height: '400px' }}>
            {}
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-gray-700 rounded-lg px-6 py-2">
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-white">Öğretmen Ekranı</span>
              </div>
            </div>
            {}
            <div className="absolute top-16 left-1/2 transform -translate-x-1/2 w-32 h-3 bg-gray-600 rounded">
              <span className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-gray-400">
                Ekran
              </span>
            </div>
            {}
            {mockStudents.map((student, index) => (
              <motion.div
                key={student.id}
                className="absolute cursor-pointer"
                style={{
                  left: `${student.x}%`,
                  top: `${student.y}%`,
                  transform: 'translate(-50%, -50%)'
                }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5, delay: index * 0.05 }}
                onClick={() => setSelectedStudent(student)}
                whileHover={{ scale: 1.2 }}
              >
                {}
                <div
                  className="w-8 h-8 rounded-full border-2 border-white/20 flex items-center justify-center text-xs font-bold relative"
                  style={{
                    backgroundColor: getAttentionColor(student.attention),
                    opacity: getAttentionOpacity(student.attention)
                  }}
                >
                  {student.name.split(' ')[0][0]}
                  {}
                  <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full border border-black"
                       style={{ backgroundColor: getAttentionColor(student.attention) }}>
                  </div>
                </div>
                {}
                {(viewMode === 'gaze' || viewMode === 'combined') && (
                  <motion.div
                    className="absolute top-1/2 left-1/2 origin-center"
                    style={{
                      transform: `translate(-50%, -50%) rotate(${Math.atan2(
                        student.gazeY - student.y,
                        student.gazeX - student.x
                      ) * 180 / Math.PI}deg)`
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.02 }}
                  >
                    <div className="w-6 h-0.5 bg-cyan-400 opacity-70"></div>
                    <div className="absolute right-0 top-0 w-0 h-0 border-l-2 border-l-cyan-400 border-t border-b border-transparent"></div>
                  </motion.div>
                )}
                {}
                <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-black/80 text-white text-xs px-2 py-1 rounded opacity-0 hover:opacity-100 transition-opacity whitespace-nowrap">
                  {student.name} - {student.attention}%
                </div>
              </motion.div>
            ))}
            {}
            {viewMode === 'attention' && (
              <div className="absolute inset-0 pointer-events-none">
                <svg className="w-full h-full opacity-30">
                  <defs>
                    <radialGradient id="attentionGrad" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#10B981" stopOpacity="0.6"/>
                      <stop offset="50%" stopColor="#F59E0B" stopOpacity="0.3"/>
                      <stop offset="100%" stopColor="#EF4444" stopOpacity="0.1"/>
                    </radialGradient>
                  </defs>
                  {mockStudents.map((student, index) => (
                    <circle
                      key={index}
                      cx={`${student.x}%`}
                      cy={`${student.y}%`}
                      r="8"
                      fill={getAttentionColor(student.attention)}
                      opacity={getAttentionOpacity(student.attention) * 0.3}
                    />
                  ))}
                </svg>
              </div>
            )}
          </div>
          {}
          <div className="flex items-center justify-center mt-4 space-x-6">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-sm text-gray-400">Yüksek Dikkat (85%+)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span className="text-sm text-gray-400">Orta Dikkat (70-84%)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span className="text-sm text-gray-400">Düşük Dikkat (&lt;70%)</span>
            </div>
          </div>
        </motion.div>
        {}
        <motion.div
          className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <h3 className="text-lg font-semibold text-white mb-4">Detaylı Analiz</h3>
          {selectedStudent ? (
            <div className="space-y-4">
              <div className="text-center pb-4 border-b border-white/10">
                <div
                  className="w-16 h-16 mx-auto rounded-full flex items-center justify-center text-xl font-bold mb-3"
                  style={{ backgroundColor: getAttentionColor(selectedStudent.attention) }}
                >
                  {selectedStudent.name.split(' ')[0][0]}
                </div>
                <h4 className="text-white font-medium">{selectedStudent.name}</h4>
                <p className="text-gray-400 text-sm">Öğrenci #{selectedStudent.id}</p>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Dikkat Seviyesi:</span>
                  <span className="text-white font-medium">{selectedStudent.attention}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${selectedStudent.attention}%`,
                      backgroundColor: getAttentionColor(selectedStudent.attention)
                    }}
                  />
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Bakış Yönü:</span>
                  <span className="text-white font-medium">
                    {selectedStudent.gazeX > 40 && selectedStudent.gazeX < 60 && selectedStudent.gazeY < 30 
                      ? 'Ekran odaklı' 
                      : 'Dikkati dağınık'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Konum:</span>
                  <span className="text-white font-medium">
                    Sıra {Math.ceil(selectedStudent.y / 20)}, Sütun {Math.ceil(selectedStudent.x / 20)}
                  </span>
                </div>
              </div>
              <div className="pt-4 border-t border-white/10">
                <h5 className="text-white font-medium mb-2">Öneriler:</h5>
                <ul className="text-sm text-gray-400 space-y-1">
                  {selectedStudent.attention < 70 && (
                    <li>• Öğrenciye bireysel dikkat verin</li>
                  )}
                  {selectedStudent.gazeX < 40 || selectedStudent.gazeX > 60 && (
                    <li>• Bakış yönünü ekran odaklı hale getirin</li>
                  )}
                  {selectedStudent.attention >= 85 && (
                    <li>• Mükemmel dikkat seviyesi!</li>
                  )}
                </ul>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <Map className="w-12 h-12 text-gray-500 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">
                Detaylı analiz için bir öğrenciye tıklayın
              </p>
            </div>
          )}
        </motion.div>
      </div>
      {}
      <motion.div
        className="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Dikkat İstatistikleri</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { 
              label: 'Yüksek Dikkat', 
              value: mockStudents.filter(s => s.attention >= 85).length, 
              total: mockStudents.length,
              color: 'green' 
            },
            { 
              label: 'Orta Dikkat', 
              value: mockStudents.filter(s => s.attention >= 70 && s.attention < 85).length, 
              total: mockStudents.length,
              color: 'yellow' 
            },
            { 
              label: 'Düşük Dikkat', 
              value: mockStudents.filter(s => s.attention < 70).length, 
              total: mockStudents.length,
              color: 'red' 
            },
            { 
              label: 'Ekran odaklı', 
              value: mockStudents.filter(s => s.gazeX > 40 && s.gazeX < 60 && s.gazeY < 30).length, 
              total: mockStudents.length,
              color: 'blue' 
            }
          ].map((stat, index) => (
            <div key={index} className="text-center">
              <div className={`text-2xl font-bold text-${stat.color}-400 mb-1`}>
                {stat.value}/{stat.total}
              </div>
              <div className="text-gray-400 text-sm">{stat.label}</div>
              <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                <div
                  className={`h-2 rounded-full bg-${stat.color}-400 transition-all duration-1000`}
                  style={{ width: `${(stat.value / stat.total) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
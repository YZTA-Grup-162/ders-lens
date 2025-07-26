import { motion } from 'framer-motion';
import React, { useEffect, useRef, useState } from 'react';
import { webSocketService } from '../services/websocket';
interface StudentMetrics {
  id: string;
  name: string;
  attentionScore: number;
  attentionState: 'attentive' | 'distracted' | 'drowsy' | 'away';
  attentionHistory: number[];
  engagementLevel: number;
  engagementCategory: 'very_low' | 'low' | 'moderate' | 'high' | 'very_high';
  engagementIndicators: {
    headMovement: number;
    eyeContact: number;
    facialExpression: number;
    posture: number;
  };
  primaryEmotion: string;
  emotionConfidence: number;
  emotionHistory: string[];
  valence: number;
  arousal: number;
  gazeDirection: 'center' | 'left' | 'right' | 'up' | 'down' | 'away';
  gazeOnScreen: boolean;
  gazeConfidence: number;
  gazeHeatmap: Array<{x: number, y: number, intensity: number}>;
  lastUpdate: Date;
  status: 'online' | 'offline' | 'away';
  sessionDuration: number;
  alerts: Array<{
    type: 'attention' | 'distraction' | 'emotion' | 'technical';
    message: string;
    severity: 'low' | 'medium' | 'high';
    timestamp: Date;
  }>;
}
interface ClassroomMetrics {
  totalStudents: number;
  activeStudents: number;
  averageAttention: number;
  averageEngagement: number;
  attentionDistribution: Record<string, number>;
  emotionDistribution: Record<string, number>;
  alertCount: number;
  sessionStartTime: Date;
}
export const EnhancedTeacherDashboard: React.FC = () => {
  const [students, setStudents] = useState<StudentMetrics[]>([]);
  const [classroomMetrics, setClassroomMetrics] = useState<ClassroomMetrics | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'analytics'>('grid');
  const [alertFilter, setAlertFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [isConnected, setIsConnected] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const refreshInterval = useRef<NodeJS.Timeout>();
  useEffect(() => {
    const connectTeacher = async () => {
      try {
        await webSocketService.connectAsTeacher({
          teacherId: 'teacher-1',
          classId: 'class-1'
        });
        setIsConnected(true);
        webSocketService.onStudentUpdate((data) => {
          setStudents(prev => {
            const index = prev.findIndex(s => s.id === data.studentId);
            if (index >= 0) {
              const updated = [...prev];
              updated[index] = { ...updated[index], ...data };
              return updated;
            } else {
              return [...prev, data as StudentMetrics];
            }
          });
        });
        webSocketService.onClassroomMetrics((metrics) => {
          setClassroomMetrics(metrics);
        });
        webSocketService.onAlert((alert) => {
          setStudents(prev => prev.map(student => 
            student.id === alert.studentId 
              ? { ...student, alerts: [alert, ...student.alerts.slice(0, 9)] }
              : student
          ));
        });
      } catch (error) {
        console.error('Failed to connect as teacher:', error);
      }
    };
    connectTeacher();
    return () => {
      webSocketService.disconnect();
      setIsConnected(false);
    };
  }, []);
  useEffect(() => {
    if (autoRefresh && isConnected) {
      refreshInterval.current = setInterval(() => {
        webSocketService.requestClassroomUpdate();
      }, 5000);
    } else {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
      }
    }
    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
      }
    };
  }, [autoRefresh, isConnected]);
  const getAttentionColor = (state: string) => {
    switch (state) {
      case 'attentive': return 'bg-green-500';
      case 'distracted': return 'bg-yellow-500';
      case 'drowsy': return 'bg-orange-500';
      case 'away': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  };
  const getEngagementColor = (level: number) => {
    if (level >= 0.8) return 'text-green-600';
    if (level >= 0.6) return 'text-blue-600';
    if (level >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };
  const getEmotionEmoji = (emotion: string) => {
    const emojiMap: Record<string, string> = {
      'happy': '😊',
      'sad': '😢',
      'angry': '😠',
      'surprised': '😲',
      'fear': '😨',
      'disgust': '🤢',
      'neutral': '😐',
      'engaged': '🤔',
      'bored': '😴',
      'confused': '😕',
      'frustrated': '😤'
    };
    return emojiMap[emotion.toLowerCase()] || '😐';
  };
  const filteredAlerts = students.flatMap(student => 
    student.alerts.filter(alert => 
      alertFilter === 'all' || alert.severity === alertFilter
    ).map(alert => ({ ...alert, studentName: student.name, studentId: student.id }))
  ).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 10);
  const renderStudentCard = (student: StudentMetrics) => (
    <motion.div
      key={student.id}
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`bg-white rounded-lg shadow-md p-4 cursor-pointer transition-all hover:shadow-lg border-l-4 ${
        student.status === 'online' ? 'border-green-500' : 
        student.status === 'away' ? 'border-yellow-500' : 'border-gray-400'
      }`}
      onClick={() => setSelectedStudent(student.id)}
    >
      {}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${getAttentionColor(student.attentionState)}`} />
          <div>
            <h3 className="font-semibold text-gray-900">{student.name}</h3>
            <p className="text-sm text-gray-500">ID: {student.id}</p>
          </div>
        </div>
        <div className="text-2xl">
          {getEmotionEmoji(student.primaryEmotion)}
        </div>
      </div>
      {}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="text-center">
          <div className="text-sm text-gray-600">Attention</div>
          <div className={`text-lg font-bold ${getEngagementColor(student.attentionScore)}`}>
            {(student.attentionScore * 100).toFixed(0)}%
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-600">Engagement</div>
          <div className={`text-lg font-bold ${getEngagementColor(student.engagementLevel)}`}>
            {(student.engagementLevel * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      {}
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-600">Gaze:</span>
        <span className={`font-medium ${
          student.gazeOnScreen ? 'text-green-600' : 'text-red-600'
        }`}>
          {student.gazeOnScreen ? `On Screen (${student.gazeDirection})` : 'Away'}
        </span>
      </div>
      {}
      <div className="flex items-center justify-between text-sm mt-2">
        <span className="text-gray-600">Emotion:</span>
        <span className="font-medium">
          {student.primaryEmotion} ({(student.emotionConfidence * 100).toFixed(0)}%)
        </span>
      </div>
      {}
      {student.alerts.length > 0 && (
        <div className="mt-2">
          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
            student.alerts[0].severity === 'high' ? 'bg-red-100 text-red-800' :
            student.alerts[0].severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
            'bg-blue-100 text-blue-800'
          }`}>
            {student.alerts.length} Alert{student.alerts.length > 1 ? 's' : ''}
          </span>
        </div>
      )}
      {}
      <div className="mt-3">
        <div className="text-xs text-gray-500 mb-1">Attention Trend (last 10 min)</div>
        <div className="flex space-x-1 h-6">
          {student.attentionHistory.slice(-20).map((score, index) => (
            <div
              key={index}
              className={`flex-1 rounded-sm ${
                score >= 0.7 ? 'bg-green-300' :
                score >= 0.4 ? 'bg-yellow-300' :
                'bg-red-300'
              }`}
              style={{ height: `${score * 100}%` }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
  const renderStudentDetail = (student: StudentMetrics) => (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">{student.name} - Detailed View</h2>
        <button
          onClick={() => setSelectedStudent(null)}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          ← Back to Overview
        </button>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-bold mb-3">🎯 Attention Analysis</h3>
          <div className="space-y-3">
            <div className={`px-3 py-2 rounded-lg bg-${getAttentionColor(student.attentionState).replace('bg-', '').replace('-500', '-100')} text-${getAttentionColor(student.attentionState).replace('bg-', '').replace('-500', '-800')}`}>
              <div className="font-medium">State: {student.attentionState.toUpperCase()}</div>
              <div className="text-sm">Score: {(student.attentionScore * 100).toFixed(1)}%</div>
            </div>
            {}
            <div>
              <div className="text-sm text-gray-600 mb-2">Last 30 minutes</div>
              <svg width="100%" height="60" className="border rounded">
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points={student.attentionHistory.map((score, i) => 
                    `${(i / student.attentionHistory.length) * 100},${60 - (score * 50)}`
                  ).join(' ')}
                />
              </svg>
            </div>
          </div>
        </div>
        {}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-bold mb-3">📊 Engagement Metrics</h3>
          <div className="space-y-3">
            <div>
              <div className="font-medium">Overall Level: {student.engagementCategory.replace('_', ' ').toUpperCase()}</div>
              <div className="text-sm text-gray-600">{(student.engagementLevel * 100).toFixed(1)}%</div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Head Movement:</span>
                <span>{(student.engagementIndicators.headMovement * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Eye Contact:</span>
                <span>{(student.engagementIndicators.eyeContact * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Facial Expression:</span>
                <span>{(student.engagementIndicators.facialExpression * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Posture:</span>
                <span>{(student.engagementIndicators.posture * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>
        {}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-bold mb-3">😊 Emotion Analysis</h3>
          <div className="space-y-3">
            <div className="text-center">
              <div className="text-4xl mb-2">{getEmotionEmoji(student.primaryEmotion)}</div>
              <div className="font-medium">{student.primaryEmotion}</div>
              <div className="text-sm text-gray-600">
                Confidence: {(student.emotionConfidence * 100).toFixed(1)}%
              </div>
            </div>
            <div className="pt-3 border-t">
              <div className="text-sm space-y-1">
                <div>Valence: {student.valence.toFixed(2)} (negative ← → positive)</div>
                <div>Arousal: {student.arousal.toFixed(2)} (calm ← → excited)</div>
              </div>
            </div>
            {}
            <div>
              <div className="text-sm text-gray-600 mb-2">Recent Emotions</div>
              <div className="flex space-x-1">
                {student.emotionHistory.slice(-10).map((emotion, index) => (
                  <span key={index} className="text-lg">
                    {getEmotionEmoji(emotion)}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
        {}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-bold mb-3">👁️ Gaze Tracking</h3>
          <div className="space-y-3">
            <div className={`px-3 py-2 rounded-lg ${
              student.gazeOnScreen ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              <div className="font-medium">
                {student.gazeOnScreen ? 'Looking at Screen' : 'Looking Away'}
              </div>
              <div className="text-sm">Direction: {student.gazeDirection}</div>
              <div className="text-sm">Confidence: {(student.gazeConfidence * 100).toFixed(1)}%</div>
            </div>
            {}
            <div>
              <div className="text-sm text-gray-600 mb-2">Gaze Heatmap</div>
              <div className="relative bg-gray-200 rounded" style={{ aspectRatio: '16/9', height: '120px' }}>
                {student.gazeHeatmap.map((point, index) => (
                  <div
                    key={index}
                    className="absolute bg-red-500 rounded-full"
                    style={{
                      left: `${point.x * 100}%`,
                      top: `${point.y * 100}%`,
                      width: `${Math.max(4, point.intensity * 12)}px`,
                      height: `${Math.max(4, point.intensity * 12)}px`,
                      opacity: point.intensity,
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
      {}
      <div className="mt-6 bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-bold mb-3">🚨 Recent Alerts</h3>
        <div className="space-y-2">
          {student.alerts.slice(0, 5).map((alert, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border-l-4 ${
                alert.severity === 'high' ? 'bg-red-50 border-red-500 text-red-800' :
                alert.severity === 'medium' ? 'bg-yellow-50 border-yellow-500 text-yellow-800' :
                'bg-blue-50 border-blue-500 text-blue-800'
              }`}
            >
              <div className="font-medium">{alert.type.toUpperCase()}</div>
              <div className="text-sm">{alert.message}</div>
              <div className="text-xs opacity-75">
                {alert.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
  const renderClassroomOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm text-gray-600">Total Students</div>
        <div className="text-2xl font-bold text-gray-900">
          {classroomMetrics?.totalStudents || 0}
        </div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm text-gray-600">Active Students</div>
        <div className="text-2xl font-bold text-green-600">
          {classroomMetrics?.activeStudents || 0}
        </div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm text-gray-600">Avg Attention</div>
        <div className={`text-2xl font-bold ${getEngagementColor(classroomMetrics?.averageAttention || 0)}`}>
          {((classroomMetrics?.averageAttention || 0) * 100).toFixed(0)}%
        </div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm text-gray-600">Avg Engagement</div>
        <div className={`text-2xl font-bold ${getEngagementColor(classroomMetrics?.averageEngagement || 0)}`}>
          {((classroomMetrics?.averageEngagement || 0) * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
  if (selectedStudent) {
    const student = students.find(s => s.id === selectedStudent);
    if (student) {
      return renderStudentDetail(student);
    }
  }
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">👨‍🏫 Teacher Dashboard</h1>
            <p className="text-gray-600">Real-time classroom attention & engagement monitoring</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
            </div>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg font-medium ${
                autoRefresh ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-700'
              }`}
            >
              {autoRefresh ? '⏸️ Pause' : '▶️ Resume'} Auto-refresh
            </button>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as any)}
              className="px-3 py-2 border rounded-lg"
            >
              <option value="grid">Grid View</option>
              <option value="list">List View</option>
              <option value="analytics">Analytics View</option>
            </select>
          </div>
        </div>
        {}
        {renderClassroomOverview()}
        {}
        {filteredAlerts.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-4 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold">🚨 Active Alerts</h2>
              <select
                value={alertFilter}
                onChange={(e) => setAlertFilter(e.target.value as any)}
                className="px-3 py-1 border rounded"
              >
                <option value="all">All Alerts</option>
                <option value="high">High Priority</option>
                <option value="medium">Medium Priority</option>
                <option value="low">Low Priority</option>
              </select>
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {filteredAlerts.map((alert, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg border-l-4 cursor-pointer hover:bg-gray-50 ${
                    alert.severity === 'high' ? 'bg-red-50 border-red-500' :
                    alert.severity === 'medium' ? 'bg-yellow-50 border-yellow-500' :
                    'bg-blue-50 border-blue-500'
                  }`}
                  onClick={() => setSelectedStudent(alert.studentId)}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium">{alert.studentName}</div>
                      <div className="text-sm text-gray-600">{alert.message}</div>
                    </div>
                    <div className="text-xs text-gray-500">
                      {alert.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {students.map(renderStudentCard)}
        </div>
        {students.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">No students connected</div>
            <div className="text-gray-400 text-sm mt-2">
              Students will appear here when they join the session
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
export default EnhancedTeacherDashboard;
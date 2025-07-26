import { AnimatePresence, motion } from 'framer-motion';
import React, { useCallback, useEffect, useState } from 'react';
import { EnhancedLiveCamera } from '../components/EnhancedLiveCamera';
import { webSocketService } from '../services/websocket';
interface StudentMetrics {
  attention: {
    score: number;
    state: 'attentive' | 'distracted' | 'drowsy' | 'away';
    trend: number[];
  };
  engagement: {
    level: number;
    category: string;
    indicators: {
      participation: number;
      focus: number;
      interaction: number;
    };
  };
  emotion: {
    primary: string;
    confidence: number;
    valence: number;
    arousal: number;
  };
  gaze: {
    onScreen: boolean;
    direction: string;
    screenTime: number;
  };
  session: {
    duration: number;
    breaks: number;
    productivity: number;
  };
}
interface Feedback {
  type: 'suggestion' | 'achievement' | 'warning' | 'information';
  message: string;
  actionable?: string;
  timestamp: Date;
}
interface StudySession {
  id: string;
  courseId: string;
  courseName: string;
  startTime: Date;
  targetDuration: number;
  goals: string[];
}
export const EnhancedStudentPage: React.FC = () => {
  const [studentId] = useState(`student-${Date.now()}`);
  const [isActive, setIsActive] = useState(false);
  const [metrics, setMetrics] = useState<StudentMetrics | null>(null);
  const [feedback, setFeedback] = useState<Feedback[]>([]);
  const [currentSession, setCurrentSession] = useState<StudySession | null>(null);
  const [showCamera, setShowCamera] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [goals, setGoals] = useState<string[]>(['Stay focused for 25 minutes', 'Complete reading assignment']);
  const [newGoal, setNewGoal] = useState('');
  const [points, setPoints] = useState(0);
  const [streak, setStreak] = useState(0);
  const [achievements, setAchievements] = useState<string[]>([]);
  const [level, setLevel] = useState(1);
  const [lastBreakTime, setLastBreakTime] = useState<Date>(new Date());
  const [showBreakReminder, setShowBreakReminder] = useState(false);
  useEffect(() => {
    const connect = async () => {
      try {
        await webSocketService.connectAsStudent({
          studentId,
          sessionId: 'session-1'
        });
        setIsConnected(true);
        webSocketService.getSocket()?.on('student_feedback', (data) => {
          setFeedback(prev => [data, ...prev.slice(0, 9)]);
          if (data.metrics) {
            setMetrics(data.metrics);
          }
          if (data.achievement) {
            setAchievements(prev => [...prev, data.achievement]);
            setPoints(prev => prev + data.points || 0);
          }
        });
        webSocketService.getSocket()?.on('session_update', (sessionData) => {
          setCurrentSession(sessionData);
        });
      } catch (error) {
        console.error('Failed to connect:', error);
      }
    };
    connect();
    return () => {
      webSocketService.disconnect();
      setIsConnected(false);
    };
  }, [studentId]);
  useEffect(() => {
    if (!isActive) return;
    const checkBreakTime = setInterval(() => {
      const now = new Date();
      const timeSinceBreak = (now.getTime() - lastBreakTime.getTime()) / (1000 * 60); 
      if (timeSinceBreak > 25) { 
        setShowBreakReminder(true);
      }
    }, 60000); 
    return () => clearInterval(checkBreakTime);
  }, [isActive, lastBreakTime]);
  useEffect(() => {
    const newLevel = Math.floor(points / 100) + 1;
    if (newLevel > level) {
      setLevel(newLevel);
      setFeedback(prev => [{
        type: 'achievement',
        message: `🎉 Level up! You're now level ${newLevel}!`,
        timestamp: new Date()
      }, ...prev.slice(0, 9)]);
    }
  }, [points, level]);
  const startSession = useCallback(() => {
    setIsActive(true);
    setCurrentSession({
      id: `session-${Date.now()}`,
      courseId: 'course-1',
      courseName: 'Current Study Session',
      startTime: new Date(),
      targetDuration: 60, 
      goals
    });
    webSocketService.getSocket()?.emit('start_session', {
      studentId,
      session: currentSession
    });
  }, [studentId, goals, currentSession]);
  const endSession = useCallback(() => {
    setIsActive(false);
    if (currentSession) {
      const duration = (new Date().getTime() - currentSession.startTime.getTime()) / (1000 * 60);
      const sessionPoints = Math.floor(duration * 2); 
      setPoints(prev => prev + sessionPoints);
      setFeedback(prev => [{
        type: 'information',
        message: `Session completed! Duration: ${duration.toFixed(1)} minutes. +${sessionPoints} points`,
        timestamp: new Date()
      }, ...prev.slice(0, 9)]);
    }
    setCurrentSession(null);
    webSocketService.getSocket()?.emit('end_session', {
      studentId
    });
  }, [studentId, currentSession]);
  const takeBreak = useCallback(() => {
    setLastBreakTime(new Date());
    setShowBreakReminder(false);
    setStreak(prev => prev + 1);
    setFeedback(prev => [{
      type: 'suggestion',
      message: '✨ Great job taking a break! Streak: ' + (streak + 1),
      timestamp: new Date()
    }, ...prev.slice(0, 9)]);
  }, [streak]);
  const addGoal = useCallback(() => {
    if (newGoal.trim()) {
      setGoals(prev => [...prev, newGoal.trim()]);
      setNewGoal('');
    }
  }, [newGoal]);
  const removeGoal = useCallback((index: number) => {
    setGoals(prev => prev.filter((_, i) => i !== index));
  }, []);
  const getAttentionColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };
  const getEmotionEmoji = (emotion: string) => {
    const emojiMap: Record<string, string> = {
      'happy': '😊',
      'focused': '🤔',
      'neutral': '😐',
      'tired': '😴',
      'stressed': '😰',
      'confused': '😕',
      'engaged': '🧠'
    };
    return emojiMap[emotion] || '😐';
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        {}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">🎓 My Learning Dashboard</h1>
              <p className="text-gray-600">Track your attention, engagement, and learning progress</p>
            </div>
            <div className="flex items-center space-x-4">
              {}
              <div className="text-center">
                <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-lg">
                  <div className="text-sm">Level {level}</div>
                  <div className="font-bold">{points} pts</div>
                </div>
              </div>
              {}
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
              </div>
              {}
              <button
                onClick={isActive ? endSession : startSession}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  isActive 
                    ? 'bg-red-500 text-white hover:bg-red-600' 
                    : 'bg-green-500 text-white hover:bg-green-600'
                }`}
              >
                {isActive ? '⏹️ End Session' : '▶️ Start Session'}
              </button>
            </div>
          </div>
          {}
          {currentSession && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-blue-900">{currentSession.courseName}</h3>
                  <p className="text-sm text-blue-700">
                    Started: {currentSession.startTime.toLocaleTimeString()}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-sm text-blue-700">Target: {currentSession.targetDuration} min</div>
                  <div className="text-lg font-bold text-blue-900">
                    {Math.floor((new Date().getTime() - currentSession.startTime.getTime()) / (1000 * 60))} min
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {}
          <div className="lg:col-span-2 space-y-6">
            {}
            {showCamera && (
              <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                <div className="p-4 border-b flex items-center justify-between">
                  <h2 className="text-xl font-bold text-gray-900">📹 Live Analysis</h2>
                  <button
                    onClick={() => setShowCamera(false)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                </div>
                <EnhancedLiveCamera />
              </div>
            )}
            {}
            {showMetrics && metrics && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">📊 Performance Metrics</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-4 bg-gradient-to-br from-green-100 to-green-200 rounded-lg">
                    <div className="text-sm text-green-700">Attention</div>
                    <div className={`text-2xl font-bold ${getAttentionColor(metrics.attention.score)}`}>
                      {(metrics.attention.score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-green-600">{metrics.attention.state}</div>
                  </div>
                  <div className="text-center p-4 bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg">
                    <div className="text-sm text-blue-700">Engagement</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {(metrics.engagement.level * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-blue-600">{metrics.engagement.category}</div>
                  </div>
                  <div className="text-center p-4 bg-gradient-to-br from-purple-100 to-purple-200 rounded-lg">
                    <div className="text-sm text-purple-700">Emotion</div>
                    <div className="text-2xl">{getEmotionEmoji(metrics.emotion.primary)}</div>
                    <div className="text-xs text-purple-600">{metrics.emotion.primary}</div>
                  </div>
                  <div className="text-center p-4 bg-gradient-to-br from-orange-100 to-orange-200 rounded-lg">
                    <div className="text-sm text-orange-700">Screen Time</div>
                    <div className="text-2xl font-bold text-orange-600">
                      {metrics.gaze.screenTime.toFixed(0)}%
                    </div>
                    <div className="text-xs text-orange-600">
                      {metrics.gaze.onScreen ? 'On Screen' : 'Away'}
                    </div>
                  </div>
                </div>
                {}
                <div className="mb-4">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Engagement Breakdown</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Participation:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full" 
                            style={{ width: `${metrics.engagement.indicators.participation * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {(metrics.engagement.indicators.participation * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Focus:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full" 
                            style={{ width: `${metrics.engagement.indicators.focus * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {(metrics.engagement.indicators.focus * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Interaction:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-purple-500 h-2 rounded-full" 
                            style={{ width: `${metrics.engagement.indicators.interaction * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {(metrics.engagement.indicators.interaction * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                {}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Attention Trend</h3>
                  <svg width="100%" height="60" className="border rounded bg-gray-50">
                    <polyline
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2"
                      points={metrics.attention.trend.map((score, i) => 
                        `${(i / Math.max(metrics.attention.trend.length - 1, 1)) * 100},${60 - (score * 50)}`
                      ).join(' ')}
                    />
                  </svg>
                </div>
              </div>
            )}
          </div>
          {}
          <div className="space-y-6">
            {}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">🎯 Session Goals</h2>
              <div className="space-y-2 mb-4">
                {goals.map((goal, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">{goal}</span>
                    <button
                      onClick={() => removeGoal(index)}
                      className="text-red-500 hover:text-red-700 text-sm"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={newGoal}
                  onChange={(e) => setNewGoal(e.target.value)}
                  placeholder="Add a goal..."
                  className="flex-1 px-3 py-2 border rounded-lg text-sm"
                  onKeyPress={(e) => e.key === 'Enter' && addGoal()}
                />
                <button
                  onClick={addGoal}
                  className="px-3 py-2 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600"
                >
                  Add
                </button>
              </div>
            </div>
            {}
            {achievements.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">🏆 Recent Achievements</h2>
                <div className="space-y-2">
                  {achievements.slice(-5).map((achievement, index) => (
                    <div key={index} className="p-2 bg-yellow-50 border border-yellow-200 rounded text-sm">
                      {achievement}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">💬 Live Feedback</h2>
              <div className="space-y-3 max-h-60 overflow-y-auto">
                {feedback.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`p-3 rounded-lg border-l-4 ${
                      item.type === 'achievement' ? 'bg-green-50 border-green-500 text-green-800' :
                      item.type === 'suggestion' ? 'bg-blue-50 border-blue-500 text-blue-800' :
                      item.type === 'warning' ? 'bg-yellow-50 border-yellow-500 text-yellow-800' :
                      'bg-gray-50 border-gray-500 text-gray-800'
                    }`}
                  >
                    <div className="text-sm font-medium">{item.message}</div>
                    {item.actionable && (
                      <div className="text-xs mt-1 opacity-75">{item.actionable}</div>
                    )}
                    <div className="text-xs opacity-50 mt-1">
                      {item.timestamp.toLocaleTimeString()}
                    </div>
                  </motion.div>
                ))}
                {feedback.length === 0 && (
                  <div className="text-gray-500 text-sm text-center py-4">
                    No feedback yet. Start your session to begin receiving insights!
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        {}
        <AnimatePresence>
          {showBreakReminder && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="bg-white p-6 rounded-xl shadow-xl max-w-md mx-4"
              >
                <div className="text-center">
                  <div className="text-4xl mb-4">☕</div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">Time for a Break!</h3>
                  <p className="text-gray-600 mb-6">
                    You've been studying for 25 minutes. Taking regular breaks helps maintain focus and productivity.
                  </p>
                  <div className="flex space-x-3">
                    <button
                      onClick={takeBreak}
                      className="flex-1 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
                    >
                      Take Break
                    </button>
                    <button
                      onClick={() => setShowBreakReminder(false)}
                      className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                    >
                      Continue
                    </button>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
export default EnhancedStudentPage;
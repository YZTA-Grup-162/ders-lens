import { motion } from 'framer-motion';
import React, { useCallback, useEffect, useRef, useState } from 'react';
interface Metrics {
  attention: number;
  engagement: number;
  emotion: string;
  emotion_confidence: number;
  gaze_direction: string;
  gaze_x: number;
  gaze_y: number;
  posture: string;
  overall_focus: number;
}
interface DemoAnalysis {
  display_metrics: Metrics;
  raw_metrics: Record<string, number>;
  smoothed_metrics: Record<string, number>;
  face_detected: boolean;
  hands_detected: boolean;
  processing_time_ms: number;
  frame_count: number;
  session_duration: number;
  model_version: string;
  models_loaded: string[];
  error?: string;
}
const DEMO_API_URL = 'http://localhost:8000/api/demo';
export const StudentDemo: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [metrics, setMetrics] = useState<Metrics>({
    attention: 0,
    engagement: 0,
    emotion: 'neutral',
    emotion_confidence: 0,
    gaze_direction: 'center',
    gaze_x: 0.5,
    gaze_y: 0.5,
    posture: 'good',
    overall_focus: 0
  });
  const [isActive, setIsActive] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [sessionInfo, setSessionInfo] = useState({
    duration: 0,
    frameCount: 0,
    modelVersion: '',
    modelsLoaded: [] as string[]
  });
  const [error, setError] = useState<string | null>(null);
  const [attentionTrend, setAttentionTrend] = useState<number[]>([]);
  const [engagementTrend, setEngagementTrend] = useState<number[]>([]);
  const [avgProcessingTime, setAvgProcessingTime] = useState<number>(0);
  const [lastUpdateTime, setLastUpdateTime] = useState<number>(0);
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status', {
          method: 'GET',
          signal: AbortSignal.timeout(3000)
        });
        if (response.ok) {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch (error) {
        setBackendStatus('offline');
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 10000); 
    return () => clearInterval(interval);
  }, []);
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480, 
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
        setError(null);
      }
    } catch (err) {
      setError('Camera access denied. Please enable camera permissions.');
      console.error('Camera error:', err);
    }
  }, []);
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, []);
  useEffect(() => {
    if (!isActive || backendStatus !== 'online') return;
    const analyzeFrame = async () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob>((resolve) => {
          canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.85);
        });
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        const response = await fetch(DEMO_API_URL, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data: DemoAnalysis = await response.json();
        setMetrics(data.display_metrics);
        setSessionInfo({
          duration: data.session_duration,
          frameCount: data.frame_count,
          modelVersion: data.model_version,
          modelsLoaded: data.models_loaded
        });
        if (data.frame_count % 5 === 0) {
          setAttentionTrend(prev => [...prev.slice(-20), data.display_metrics.attention]);
          setEngagementTrend(prev => [...prev.slice(-20), data.display_metrics.engagement]);
        }
        setAvgProcessingTime(data.processing_time_ms);
        setLastUpdateTime(Date.now());
        setError(null);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Analysis failed';
        setError(`Analysis failed: ${errorMsg}`);
        console.error('Demo analysis error:', err);
      }
    };
    const interval = setInterval(analyzeFrame, 500);
    return () => clearInterval(interval);
  }, [isActive, backendStatus]);
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  const getAttentionLevel = (score: number) => {
    if (score >= 0.8) return { level: 'Excellent', color: 'text-green-500', bg: 'bg-green-100' };
    if (score >= 0.6) return { level: 'Good', color: 'text-blue-500', bg: 'bg-blue-100' };
    if (score >= 0.4) return { level: 'Moderate', color: 'text-yellow-500', bg: 'bg-yellow-100' };
    return { level: 'Low', color: 'text-red-500', bg: 'bg-red-100' };
  };
  const getEngagementLevel = (score: number) => {
    if (score >= 0.8) return { level: 'Highly Engaged', color: 'text-green-600' };
    if (score >= 0.6) return { level: 'Engaged', color: 'text-blue-600' };
    if (score >= 0.4) return { level: 'Partially Engaged', color: 'text-yellow-600' };
    return { level: 'Disengaged', color: 'text-red-600' };
  };
  const getEmotionEmoji = (emotion: string) => {
    const emojiMap: Record<string, string> = {
      'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Fear': '😨',
      'Surprise': '😲', 'Disgust': '🤢', 'Neutral': '😐', 'Focused': '🤔'
    };
    return emojiMap[emotion] || '😐';
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🎓 Student Engagement Demo
          </h1>
          <p className="text-gray-600 text-lg">
            AI-powered analysis using trained models • Stable visualization • No eye strain
          </p>
          {}
          <div className="flex items-center justify-center gap-4 mt-4">
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
              backendStatus === 'online' ? 'bg-green-100 text-green-700' :
              backendStatus === 'offline' ? 'bg-red-100 text-red-700' :
              'bg-yellow-100 text-yellow-700'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                backendStatus === 'online' ? 'bg-green-500' :
                backendStatus === 'offline' ? 'bg-red-500' :
                'bg-yellow-500'
              }`} />
              Backend: {backendStatus}
            </div>
            {sessionInfo.modelVersion && (
              <div className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                {sessionInfo.modelVersion}
              </div>
            )}
          </div>
        </motion.div>
        {}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                📹 Live Camera Feed
              </h2>
              <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas
                  ref={canvasRef}
                  className="hidden"
                />
                {}
                {isActive && (
                  <div className="absolute top-3 left-3">
                    <div className="flex items-center gap-2 bg-red-500 text-white px-2 py-1 rounded text-xs">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                      LIVE
                    </div>
                  </div>
                )}
                {}
                {isActive && metrics.gaze_direction !== 'center' && (
                  <div 
                    className="absolute w-3 h-3 bg-yellow-400 rounded-full border-2 border-white"
                    style={{
                      left: `${metrics.gaze_x * 100}%`,
                      top: `${metrics.gaze_y * 100}%`,
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                )}
              </div>
              {}
              <div className="flex gap-3 mt-4">
                {!isActive ? (
                  <button
                    onClick={startCamera}
                    disabled={backendStatus !== 'online'}
                    className="flex-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    Start Demo
                  </button>
                ) : (
                  <button
                    onClick={stopCamera}
                    className="flex-1 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    Stop Demo
                  </button>
                )}
              </div>
              {error && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {error}
                </div>
              )}
            </div>
          </motion.div>
          {}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 space-y-6"
          >
            {}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  🎯 Attention Level
                </h3>
                <div className="text-center">
                  <div className="text-4xl font-bold mb-2 text-blue-600">
                    {(metrics.attention * 100).toFixed(0)}%
                  </div>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                    getAttentionLevel(metrics.attention).bg
                  } ${getAttentionLevel(metrics.attention).color}`}>
                    {getAttentionLevel(metrics.attention).level}
                  </div>
                  {}
                  <div className="w-full bg-gray-200 rounded-full h-3 mt-4">
                    <motion.div 
                      className="bg-blue-500 h-3 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${metrics.attention * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
              </div>
              {}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  ⚡ Engagement Level
                </h3>
                <div className="text-center">
                  <div className="text-4xl font-bold mb-2 text-green-600">
                    {(metrics.engagement * 100).toFixed(0)}%
                  </div>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                    getEngagementLevel(metrics.engagement).color
                  } bg-gray-100`}>
                    {getEngagementLevel(metrics.engagement).level}
                  </div>
                  {}
                  <div className="w-full bg-gray-200 rounded-full h-3 mt-4">
                    <motion.div 
                      className="bg-green-500 h-3 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${metrics.engagement * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
              </div>
            </div>
            {}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  😊 Emotion Analysis
                </h3>
                <div className="text-center">
                  <div className="text-5xl mb-2">
                    {getEmotionEmoji(metrics.emotion)}
                  </div>
                  <div className="text-xl font-semibold text-gray-800 mb-1">
                    {metrics.emotion}
                  </div>
                  <div className="text-sm text-gray-600">
                    Confidence: {(metrics.emotion_confidence * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-2">
                    Using ONNX model: best_model.onnx
                  </div>
                </div>
              </div>
              {}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  👁️ Gaze Tracking
                </h3>
                <div className="text-center">
                  <div className="text-2xl font-bold mb-2 text-purple-600">
                    {metrics.gaze_direction.charAt(0).toUpperCase() + metrics.gaze_direction.slice(1)}
                  </div>
                  <div className="text-sm text-gray-600 mb-3">
                    Screen Position: ({(metrics.gaze_x * 100).toFixed(0)}%, {(metrics.gaze_y * 100).toFixed(0)}%)
                  </div>
                  {}
                  <div className="relative w-32 h-20 mx-auto bg-gray-100 rounded border-2 border-gray-300">
                    <div 
                      className="absolute w-2 h-2 bg-purple-500 rounded-full"
                      style={{
                        left: `${metrics.gaze_x * 100}%`,
                        top: `${metrics.gaze_y * 100}%`,
                        transform: 'translate(-50%, -50%)'
                      }}
                    />
                  </div>
                  <div className="text-xs text-gray-500 mt-2">
                    Geometric estimation via MediaPipe
                  </div>
                </div>
              </div>
            </div>
            {}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                📊 Session Information
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-blue-600">
                    {formatDuration(sessionInfo.duration)}
                  </div>
                  <div className="text-xs text-gray-600">Duration</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-600">
                    {sessionInfo.frameCount}
                  </div>
                  <div className="text-xs text-gray-600">Frames</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-purple-600">
                    {avgProcessingTime.toFixed(0)}ms
                  </div>
                  <div className="text-xs text-gray-600">Avg Processing</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-orange-600">
                    {(metrics.overall_focus * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-600">Overall Focus</div>
                </div>
              </div>
              {}
              {sessionInfo.modelsLoaded.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="text-sm font-medium text-gray-700 mb-2">Models Loaded:</div>
                  <div className="flex flex-wrap gap-2">
                    {sessionInfo.modelsLoaded.map((model, index) => (
                      <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
                        {model}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
        {}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-8 text-gray-500 text-sm"
        >
          Student Engagement Demo • Real AI Models • Stable Visualization
          <br />
          Update Rate: 2 Hz (No Flickering) • Models: ONNX + PyTorch + Scikit-learn
        </motion.div>
      </div>
    </div>
  );
};
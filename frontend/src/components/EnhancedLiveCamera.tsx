import { AnimatePresence, motion } from 'framer-motion';
import React, { useCallback, useEffect, useRef, useState } from 'react';
interface GazeData {
  x: number;
  y: number;
  confidence: number;
  direction: 'center' | 'left' | 'right' | 'up' | 'down';
  onScreen: boolean;
}
interface EmotionData {
  dominant: string;
  scores: Record<string, number>;
  valence: number;
  arousal: number;
  confidence: number;
}
interface AttentionData {
  score: number;
  state: 'attentive' | 'distracted' | 'drowsy' | 'away';
  confidence: number;
  focusRegions: Array<{x: number, y: number, width: number, height: number}>;
}
interface EngagementData {
  level: number;
  category: 'very_low' | 'low' | 'moderate' | 'high' | 'very_high';
  trends: number[];
  indicators: {
    headMovement: number;
    eyeContact: number;
    facialExpression: number;
    posture: number;
  };
}
interface FaceMetrics {
  detected: boolean;
  confidence: number;
  landmarks: Array<{x: number, y: number}>;
  headPose: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  eyeAspectRatio: number;
  mouthAspectRatio: number;
}
interface ComprehensiveAnalysis {
  gaze: GazeData;
  emotion: EmotionData;
  attention: AttentionData;
  engagement: EngagementData;
  face: FaceMetrics;
  timestamp: number;
  processingTime: number;
  modelVersion: string;
}
const API_URL = 'http://localhost:8000/api/analyze'
export const EnhancedLiveCamera: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [analysis, setAnalysis] = useState<ComprehensiveAnalysis | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [gazeHistory, setGazeHistory] = useState<Array<{x: number, y: number, timestamp: number}>>([]);
  const [attentionHistory, setAttentionHistory] = useState<number[]>([]);
  const [emotionHistory, setEmotionHistory] = useState<string[]>([]);
  const [engagementHistory, setEngagementHistory] = useState<number[]>([]);
  const [calibrationPoints, setCalibrationPoints] = useState<Array<{x: number, y: number}>>([]);
  const [currentCalibrationPoint, setCurrentCalibrationPoint] = useState(0);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/health', {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        if (response.ok) {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch {
        setBackendStatus('offline');
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 30000); 
    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: 1280, 
            height: 720,
            frameRate: 30 
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setConnected(true);
        }
      } catch (err) {
        setError('Cannot access camera. Please check permissions.');
        console.error('Camera error:', err);
      }
    };
    initCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);
  const startGazeCalibration = useCallback(() => {
    setCalibrating(true);
    setCurrentCalibrationPoint(0);
    setCalibrationPoints([
      {x: 0.1, y: 0.1}, {x: 0.5, y: 0.1}, {x: 0.9, y: 0.1},
      {x: 0.1, y: 0.5}, {x: 0.5, y: 0.5}, {x: 0.9, y: 0.5},
      {x: 0.1, y: 0.9}, {x: 0.5, y: 0.9}, {x: 0.9, y: 0.9}
    ]);
  }, []);
  const nextCalibrationPoint = useCallback(() => {
    if (currentCalibrationPoint < calibrationPoints.length - 1) {
      setCurrentCalibrationPoint(prev => prev + 1);
    } else {
      setCalibrating(false);
    }
  }, [currentCalibrationPoint, calibrationPoints.length]);
  useEffect(() => {
    if (!connected || !isActive || calibrating) return;
    const analyzeFrame = async () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      try {
        const blob = await new Promise<Blob>((resolve) => {
          canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.8);
        });
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        formData.append('include_gaze', 'true');
        formData.append('include_emotions', 'true');
        formData.append('include_engagement', 'true');
        formData.append('include_attention', 'true');
        const response = await fetch(API_URL, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) throw new Error('Analysis failed');
        const data = await response.json();
        setAnalysis(data);
        console.log('📊 AI Analysis Result:', {
          attention: `${(data.attention.score * 100).toFixed(1)}% (${data.attention.state})`,
          emotion: `${data.emotion.dominant} (${(data.emotion.confidence * 100).toFixed(1)}%)`,
          engagement: `${(data.engagement.level * 100).toFixed(1)}% (${data.engagement.category})`,
          gaze: `${data.gaze.direction} at (${(data.gaze.x * 100).toFixed(0)}%, ${(data.gaze.y * 100).toFixed(0)}%)`,
          face: `detected: ${data.face.detected}, confidence: ${(data.face.confidence * 100).toFixed(1)}%`,
          processing: `${data.processingTime.toFixed(1)}ms`
        });
        if (data.attention.score === 0.5 && data.gaze.x === 0.5 && data.emotion.dominant === 'neutral') {
          console.warn('⚠️ Backend may be returning default values - check model loading');
        }
        if (data.gaze && data.gaze.onScreen) {
          setGazeHistory(prev => [...prev.slice(-50), {
            x: data.gaze.x,
            y: data.gaze.y,
            timestamp: Date.now()
          }]);
        }
        setAttentionHistory(prev => [...prev.slice(-30), data.attention.score]);
        setEmotionHistory(prev => [...prev.slice(-30), data.emotion.dominant]);
        setEngagementHistory(prev => [...prev.slice(-30), data.engagement.level]);
        setError(null);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Analysis failed';
        setError(`Analysis failed: ${errorMsg}. Check backend connection (${API_URL})`);
        console.error('🔴 Analysis error:', err);
        if (errorMsg.includes('fetch') || errorMsg.includes('network')) {
          setConnected(false);
          setError('Backend connection lost. Please check if the backend is running.');
        }
      }
    };
    const interval = setInterval(analyzeFrame, 100); 
    return () => clearInterval(interval);
  }, [connected, isActive, calibrating]);
  useEffect(() => {
    if (!analysis || !overlayCanvasRef.current || !videoRef.current) return;
    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (analysis.gaze.onScreen) {
      const gazeX = analysis.gaze.x * canvas.width;
      const gazeY = analysis.gaze.y * canvas.height;
      ctx.beginPath();
      ctx.arc(gazeX, gazeY, 10, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(255, 0, 0, ${analysis.gaze.confidence})`;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (gazeHistory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.3)';
      ctx.lineWidth = 3;
      for (let i = 1; i < gazeHistory.length; i++) {
        const point = gazeHistory[i];
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        if (i === 1) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
    analysis.attention.focusRegions.forEach(region => {
      ctx.strokeStyle = analysis.attention.state === 'attentive' ? '#00ff00' : '#ff9800';
      ctx.lineWidth = 3;
      ctx.strokeRect(
        region.x * canvas.width,
        region.y * canvas.height,
        region.width * canvas.width,
        region.height * canvas.height
      );
    });
    if (analysis.face.detected && analysis.face.landmarks) {
      ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
      analysis.face.landmarks.slice(0, 5).forEach(landmark => { 
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, [analysis, gazeHistory]);
  const getAttentionColor = (state: string) => {
    switch (state) {
      case 'attentive': return 'text-green-600 bg-green-100';
      case 'distracted': return 'text-yellow-600 bg-yellow-100';
      case 'drowsy': return 'text-orange-600 bg-orange-100';
      case 'away': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };
  const getEngagementColor = (level: number) => {
    if (level >= 0.8) return 'text-green-600';
    if (level >= 0.6) return 'text-blue-600';
    if (level >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };
  return (
    <div className="max-w-6xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          🎯 Advanced Attention & Gaze Tracking
        </h2>
        <div className="flex gap-4 mb-4">
          <button
            onClick={() => setIsActive(!isActive)}
            className={`px-4 py-2 rounded-lg font-medium ${
              isActive 
                ? 'bg-red-500 text-white hover:bg-red-600' 
                : 'bg-green-500 text-white hover:bg-green-600'
            }`}
          >
            {isActive ? '⏹️ Stop Analysis' : '▶️ Start Analysis'}
          </button>
          <button
            onClick={startGazeCalibration}
            disabled={!connected || isActive}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            🎯 Calibrate Gaze
          </button>
        </div>
      </div>
      {}
      <div className="relative mb-6">
        <div className="relative inline-block">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="rounded-lg border-2 border-gray-300"
            style={{ maxWidth: '640px', width: '100%' }}
          />
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 pointer-events-none"
            style={{ maxWidth: '640px', width: '100%' }}
          />
        </div>
        {}
        <div className="absolute top-4 right-4 space-y-2">
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {connected ? '🟢 Camera Connected' : '🔴 Camera Disconnected'}
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            backendStatus === 'online' ? 'bg-green-100 text-green-800' : 
            backendStatus === 'offline' ? 'bg-red-100 text-red-800' :
            'bg-yellow-100 text-yellow-800'
          }`}>
            {backendStatus === 'online' ? '🟢 AI Backend Online' : 
             backendStatus === 'offline' ? '🔴 AI Backend Offline' :
             '🟡 Checking AI Backend...'}
          </div>
          {isActive && (
            <div className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
              🔄 Analyzing
            </div>
          )}
        </div>
      </div>
      {}
      <AnimatePresence>
        {calibrating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          >
            <div className="bg-white p-6 rounded-lg text-center">
              <h3 className="text-lg font-bold mb-4">Gaze Calibration</h3>
              <p className="mb-4">
                Look at the red dot and click when ready ({currentCalibrationPoint + 1}/9)
              </p>
              <button
                onClick={nextCalibrationPoint}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Next Point
              </button>
            </div>
            {calibrationPoints[currentCalibrationPoint] && (
              <div
                className="fixed w-4 h-4 bg-red-500 rounded-full"
                style={{
                  left: `${calibrationPoints[currentCalibrationPoint].x * 100}%`,
                  top: `${calibrationPoints[currentCalibrationPoint].y * 100}%`,
                  transform: 'translate(-50%, -50%)'
                }}
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
      {}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-300 text-red-700 rounded-lg">
          ⚠️ {error}
        </div>
      )}
      {}
      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-bold mb-3">🎯 Attention & Engagement</h3>
            <div className="space-y-3">
              <div className={`px-3 py-2 rounded-lg ${getAttentionColor(analysis.attention.state)}`}>
                <div className="font-medium">Attention: {analysis.attention.state.toUpperCase()}</div>
                <div className="text-sm">Score: {(analysis.attention.score * 100).toFixed(1)}%</div>
                <div className="text-sm">Confidence: {(analysis.attention.confidence * 100).toFixed(1)}%</div>
              </div>
              <div className={`px-3 py-2 rounded-lg bg-gray-100 ${getEngagementColor(analysis.engagement.level)}`}>
                <div className="font-medium">Engagement: {analysis.engagement.category.replace('_', ' ').toUpperCase()}</div>
                <div className="text-sm">Level: {(analysis.engagement.level * 100).toFixed(1)}%</div>
              </div>
              {}
              <div className="text-sm space-y-1">
                <div>Head Movement: {(analysis.engagement.indicators.headMovement * 100).toFixed(0)}%</div>
                <div>Eye Contact: {(analysis.engagement.indicators.eyeContact * 100).toFixed(0)}%</div>
                <div>Expression: {(analysis.engagement.indicators.facialExpression * 100).toFixed(0)}%</div>
                <div>Posture: {(analysis.engagement.indicators.posture * 100).toFixed(0)}%</div>
              </div>
            </div>
          </div>
          {}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-bold mb-3">😊 Emotion Analysis</h3>
            <div className="space-y-3">
              <div className="font-medium text-lg">
                Primary: {analysis.emotion.dominant}
                <span className="text-sm text-gray-600 ml-2">
                  ({(analysis.emotion.confidence * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="space-y-1 text-sm">
                {Object.entries(analysis.emotion.scores)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 5)
                  .map(([emotion, score]) => (
                    <div key={emotion} className="flex justify-between">
                      <span>{emotion}:</span>
                      <span>{(score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
              <div className="pt-2 border-t">
                <div className="text-sm">
                  Valence: {analysis.emotion.valence.toFixed(2)} | 
                  Arousal: {analysis.emotion.arousal.toFixed(2)}
                </div>
              </div>
            </div>
          </div>
          {}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-bold mb-3">👁️ Gaze Tracking</h3>
            <div className="space-y-3">
              <div className={`px-3 py-2 rounded-lg ${
                analysis.gaze.onScreen ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                <div className="font-medium">
                  {analysis.gaze.onScreen ? 'Looking at Screen' : 'Looking Away'}
                </div>
                {analysis.gaze.onScreen && (
                  <div className="text-sm">
                    Position: ({(analysis.gaze.x * 100).toFixed(0)}%, {(analysis.gaze.y * 100).toFixed(0)}%)
                  </div>
                )}
                <div className="text-sm">Direction: {analysis.gaze.direction}</div>
                <div className="text-sm">Confidence: {(analysis.gaze.confidence * 100).toFixed(1)}%</div>
              </div>
              {}
              <div className="text-sm text-gray-600">
                Gaze points collected: {gazeHistory.length}
              </div>
            </div>
          </div>
          {}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-bold mb-3">👤 Face & Pose Analysis</h3>
            <div className="space-y-3">
              <div className={`px-3 py-2 rounded-lg ${
                analysis.face.detected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                <div className="font-medium">
                  {analysis.face.detected ? 'Face Detected' : 'No Face Detected'}
                </div>
                {analysis.face.detected && (
                  <div className="text-sm">
                    Confidence: {(analysis.face.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
              {analysis.face.detected && (
                <div className="text-sm space-y-1">
                  <div>Head Pose:</div>
                  <div className="ml-2">
                    • Pitch: {analysis.face.headPose.pitch.toFixed(1)}°
                  </div>
                  <div className="ml-2">
                    • Yaw: {analysis.face.headPose.yaw.toFixed(1)}°
                  </div>
                  <div className="ml-2">
                    • Roll: {analysis.face.headPose.roll.toFixed(1)}°
                  </div>
                  <div>Eye Aspect Ratio: {analysis.face.eyeAspectRatio.toFixed(3)}</div>
                  <div>Mouth Aspect Ratio: {analysis.face.mouthAspectRatio.toFixed(3)}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {}
      {analysis && (
        <div className="mt-6 text-center text-sm text-gray-600">
          Processing Time: {analysis.processingTime.toFixed(1)}ms | 
          Model: {analysis.modelVersion} | 
          Updated: {new Date(analysis.timestamp).toLocaleTimeString()}
        </div>
      )}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};
export default EnhancedLiveCamera;
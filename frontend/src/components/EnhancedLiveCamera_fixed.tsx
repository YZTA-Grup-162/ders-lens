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
  focusRegions: Array<{x: number; y: number; width: number; height: number}>;
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
  landmarks: Array<{x: number; y: number}>;
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

const API_URL = '/api/analyze';

export const EnhancedLiveCamera: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [analysis, setAnalysis] = useState<ComprehensiveAnalysis | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [gazeHistory, setGazeHistory] = useState<Array<{x: number; y: number; timestamp: number}>>([]);
  const [attentionHistory, setAttentionHistory] = useState<number[]>([]);
  const [emotionHistory, setEmotionHistory] = useState<string[]>([]);
  const [engagementHistory, setEngagementHistory] = useState<number[]>([]);
  const [calibrationPoints, setCalibrationPoints] = useState<Array<{x: number; y: number}>>([]);
  const [currentCalibrationPoint, setCurrentCalibrationPoint] = useState(0);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [noFaceDetectedTime, setNoFaceDetectedTime] = useState(0);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('/api/health', {
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
        setError('Kameraya erişim sağlanamadı. Lütfen izinleri kontrol edin.');
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
        
        // Yüz algılama durumunu takip et
        if (!data.face.detected) {
          setNoFaceDetectedTime(prev => prev + 100);
        } else {
          setNoFaceDetectedTime(0);
        }

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
        setError(`Analiz başarısız: ${errorMsg}. Backend bağlantısını kontrol edin (${API_URL})`);
        console.error('🔴 Analysis error:', err);
        
        if (errorMsg.includes('fetch') || errorMsg.includes('network')) {
          setConnected(false);
          setError('❌ Backend bağlantısı kesildi. Lütfen backend\'in çalıştığını kontrol edin.');
        }
      }
    };

    const interval = setInterval(analyzeFrame, 100); 
    return () => clearInterval(interval);
  }, [connected, isActive, calibrating]);

  // Enhanced overlay rendering with better visual feedback
  useEffect(() => {
    if (!analysis || !overlayCanvasRef.current || !videoRef.current) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Gaze tracking visualization
    if (analysis.gaze.onScreen) {
      const gazeX = analysis.gaze.x * canvas.width;
      const gazeY = analysis.gaze.y * canvas.height;
      
      // Enhanced gaze point with pulsing effect
      const time = Date.now() * 0.005;
      const pulseSize = 8 + Math.sin(time) * 3;
      
      ctx.beginPath();
      ctx.arc(gazeX, gazeY, pulseSize, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(255, 0, 0, ${analysis.gaze.confidence * 0.8})`;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Add crosshair for better visibility
      ctx.beginPath();
      ctx.moveTo(gazeX - 15, gazeY);
      ctx.lineTo(gazeX + 15, gazeY);
      ctx.moveTo(gazeX, gazeY - 15);
      ctx.lineTo(gazeX, gazeY + 15);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Enhanced gaze trail
    if (gazeHistory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.4)';
      ctx.lineWidth = 4;
      for (let i = 1; i < gazeHistory.length; i++) {
        const point = gazeHistory[i];
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        const alpha = i / gazeHistory.length;
        
        if (i === 1) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Enhanced attention regions
    analysis.attention.focusRegions.forEach((region, index) => {
      const color = analysis.attention.state === 'attentive' ? '#00ff00' : '#ff9800';
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(
        region.x * canvas.width,
        region.y * canvas.height,
        region.width * canvas.width,
        region.height * canvas.height
      );
      ctx.setLineDash([]);
      
      // Add region label
      ctx.fillStyle = color;
      ctx.font = '12px Arial';
      ctx.fillText(
        `Focus ${index + 1}`,
        region.x * canvas.width + 5,
        region.y * canvas.height + 15
      );
    });

    // Enhanced face landmarks
    if (analysis.face.detected && analysis.face.landmarks) {
      ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
      analysis.face.landmarks.slice(0, 68).forEach((landmark, index) => { 
        ctx.beginPath();
        const size = index < 17 ? 1.5 : 1; // Larger dots for face outline
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, size, 0, 2 * Math.PI);
        ctx.fill();
      });
      
      // Draw face bounding box
      if (analysis.face.confidence > 0.5) {
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        const padding = 50;
        ctx.strokeRect(
          padding,
          padding,
          canvas.width - 2 * padding,
          canvas.height - 2 * padding
        );
      }
    }

    // No face detected warning overlay
    if (!analysis.face.detected && noFaceDetectedTime > 1000) {
      ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Pulsing border effect
      const pulseAlpha = 0.5 + Math.sin(Date.now() * 0.01) * 0.3;
      ctx.strokeStyle = `rgba(255, 0, 0, ${pulseAlpha})`;
      ctx.lineWidth = 8;
      ctx.strokeRect(4, 4, canvas.width - 8, canvas.height - 8);
    }

  }, [analysis, gazeHistory, noFaceDetectedTime]);

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

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: 'text-green-600 bg-green-100',
      sad: 'text-blue-600 bg-blue-100',
      angry: 'text-red-600 bg-red-100',
      fear: 'text-purple-600 bg-purple-100',
      surprise: 'text-yellow-600 bg-yellow-100',
      disgust: 'text-orange-600 bg-orange-100',
      neutral: 'text-gray-600 bg-gray-100'
    };
    return colors[emotion.toLowerCase()] || 'text-gray-600 bg-gray-100';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          🎯 Gelişmiş Dikkat & Gaze Takibi
        </h2>
        <div className="flex gap-4 mb-4">
          <button
            onClick={() => setIsActive(!isActive)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              isActive 
                ? 'bg-red-500 text-white hover:bg-red-600 shadow-lg' 
                : 'bg-green-500 text-white hover:bg-green-600 shadow-lg'
            }`}
          >
            {isActive ? '⏹️ Analizi Durdur' : '▶️ Analizi Başlat'}
          </button>
          <button
            onClick={startGazeCalibration}
            disabled={!connected || isActive}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-all shadow-lg"
          >
            🎯 Gaze Kalibrasyonu
          </button>
        </div>
      </div>

      {/* Enhanced Video Container */}
      <div className="relative mb-6">
        <div className="relative inline-block">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`rounded-lg border-4 transition-all duration-200 ${
              !analysis?.face.detected && isActive && noFaceDetectedTime > 1000
                ? 'border-red-500 shadow-lg shadow-red-500/50 animate-pulse' 
                : analysis?.face.detected && isActive
                ? 'border-green-400 shadow-lg shadow-green-400/30'
                : 'border-gray-300'
            }`}
            style={{ maxWidth: '640px', width: '100%' }}
          />
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 pointer-events-none rounded-lg"
            style={{ maxWidth: '640px', width: '100%' }}
          />
          
          {/* Enhanced No Face Warning Overlay */}
          {!analysis?.face.detected && isActive && noFaceDetectedTime > 1000 && (
            <div className="absolute inset-0 bg-red-500/20 rounded-lg flex items-center justify-center animate-pulse">
              <div className="bg-red-600 text-white px-6 py-3 rounded-lg font-bold shadow-xl border-2 border-red-400">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">⚠️</span>
                  <div>
                    <div className="text-lg">YÜZ ALGILANAMADI</div>
                    <div className="text-sm opacity-90">Lütfen kameraya bakın</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Status indicators */}
        <div className="absolute top-4 right-4 space-y-2">
          <div className={`px-3 py-1 rounded-full text-sm font-medium transition-all ${
            connected ? 'bg-green-100 text-green-800 shadow-md' : 'bg-red-100 text-red-800 shadow-md'
          }`}>
            {connected ? '🟢 Kamera Bağlı' : '🔴 Kamera Bağlantısı Yok'}
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium transition-all ${
            backendStatus === 'online' ? 'bg-green-100 text-green-800 shadow-md' : 
            backendStatus === 'offline' ? 'bg-red-100 text-red-800 shadow-md' :
            'bg-yellow-100 text-yellow-800 shadow-md'
          }`}>
            {backendStatus === 'online' ? '🟢 AI Backend Çevrimiçi' : 
             backendStatus === 'offline' ? '🔴 AI Backend Çevrimdışı' :
             '🟡 AI Backend Kontrol Ediliyor...'}
          </div>
          {isActive && (
            <div className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium shadow-md animate-pulse">
              🔄 Analiz Ediliyor
            </div>
          )}
          {analysis?.face.detected && isActive && (
            <div className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium shadow-md">
              ✅ Yüz Algılandı
            </div>
          )}
        </div>
      </div>

      {/* Calibration Modal */}
      <AnimatePresence>
        {calibrating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          >
            <div className="bg-white p-6 rounded-lg text-center shadow-xl">
              <h3 className="text-lg font-bold mb-4">Gaze Kalibrasyonu</h3>
              <p className="mb-4">
                Kırmızı noktaya bakın ve hazır olduğunuzda tıklayın ({currentCalibrationPoint + 1}/9)
              </p>
              <button
                onClick={nextCalibrationPoint}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-all"
              >
                Sonraki Nokta
              </button>
            </div>
            {calibrationPoints[currentCalibrationPoint] && (
              <div
                className="fixed w-6 h-6 bg-red-500 rounded-full animate-pulse border-2 border-white shadow-lg"
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

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-300 text-red-700 rounded-lg shadow-md">
          ⚠️ {error}
        </div>
      )}

      {/* Enhanced Analysis Results */}
      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Attention & Engagement */}
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-4">🎯 Dikkat & Katılım</h3>
            <div className="space-y-4">
              <div className={`px-4 py-3 rounded-lg transition-all ${getAttentionColor(analysis.attention.state)}`}>
                <div className="font-medium text-lg">Dikkat: {analysis.attention.state.toUpperCase()}</div>
                <div className="text-sm">Skor: {(analysis.attention.score * 100).toFixed(1)}%</div>
                <div className="text-sm">Güven: {(analysis.attention.confidence * 100).toFixed(1)}%</div>
              </div>
              <div className={`px-4 py-3 rounded-lg bg-gray-100 ${getEngagementColor(analysis.engagement.level)}`}>
                <div className="font-medium text-lg">Katılım: {analysis.engagement.category.replace('_', ' ').toUpperCase()}</div>
                <div className="text-sm">Seviye: {(analysis.engagement.level * 100).toFixed(1)}%</div>
              </div>
              
              {/* Engagement indicators */}
              <div className="text-sm space-y-2 border-t pt-3">
                <div className="flex justify-between">
                  <span>Kafa Hareketi:</span>
                  <span className="font-medium">{(analysis.engagement.indicators.headMovement * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Göz Teması:</span>
                  <span className="font-medium">{(analysis.engagement.indicators.eyeContact * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Yüz İfadesi:</span>
                  <span className="font-medium">{(analysis.engagement.indicators.facialExpression * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Duruş:</span>
                  <span className="font-medium">{(analysis.engagement.indicators.posture * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Emotion Analysis */}
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-4">😊 Duygu Analizi</h3>
            <div className="space-y-4">
              <div className={`px-4 py-3 rounded-lg transition-all ${getEmotionColor(analysis.emotion.dominant)}`}>
                <div className="font-medium text-xl">
                  Temel: {analysis.emotion.dominant}
                  <span className="text-sm text-gray-600 ml-2">
                    ({(analysis.emotion.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
              <div className="space-y-2 text-sm">
                <div className="font-medium mb-2">Duygu Skorları:</div>
                {Object.entries(analysis.emotion.scores)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 5)
                  .map(([emotion, score]) => (
                    <div key={emotion} className="flex justify-between items-center">
                      <span className="capitalize">{emotion}:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full transition-all" 
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                        <span className="font-medium">{(score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
              </div>
              <div className="pt-3 border-t">
                <div className="text-sm grid grid-cols-2 gap-2">
                  <div>Valence: <span className="font-medium">{analysis.emotion.valence.toFixed(2)}</span></div>
                  <div>Arousal: <span className="font-medium">{analysis.emotion.arousal.toFixed(2)}</span></div>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Gaze Tracking */}
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-4">👁️ Gaze Takibi</h3>
            <div className="space-y-4">
              <div className={`px-4 py-3 rounded-lg transition-all ${
                analysis.gaze.onScreen ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                <div className="font-medium text-lg">
                  {analysis.gaze.onScreen ? '👀 Ekrana Bakıyor' : '👁️ Başka Yere Bakıyor'}
                </div>
                {analysis.gaze.onScreen && (
                  <div className="text-sm mt-1">
                    Pozisyon: ({(analysis.gaze.x * 100).toFixed(0)}%, {(analysis.gaze.y * 100).toFixed(0)}%)
                  </div>
                )}
                <div className="text-sm">Yön: {analysis.gaze.direction}</div>
                <div className="text-sm">Güven: {(analysis.gaze.confidence * 100).toFixed(1)}%</div>
              </div>
              
              {/* Gaze statistics */}
              <div className="text-sm text-gray-600 space-y-1">
                <div>Toplanan gaze noktası: <span className="font-medium">{gazeHistory.length}</span></div>
                <div>Son 30 saniye dikkat ortalaması: <span className="font-medium">
                  {attentionHistory.length > 0 ? (attentionHistory.reduce((a, b) => a + b, 0) / attentionHistory.length * 100).toFixed(1) : 0}%
                </span></div>
              </div>
            </div>
          </div>

          {/* Enhanced Face Analysis */}
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-4">👤 Yüz & Poz Analizi</h3>
            <div className="space-y-4">
              <div className={`px-4 py-3 rounded-lg transition-all ${
                analysis.face.detected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                <div className="font-medium text-lg">
                  {analysis.face.detected ? '✅ Yüz Algılandı' : '❌ Yüz Algılanamadı'}
                </div>
                {analysis.face.detected && (
                  <div className="text-sm mt-1">
                    Güven: {(analysis.face.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
              {analysis.face.detected && (
                <div className="text-sm space-y-2">
                  <div className="font-medium">Kafa Pozu:</div>
                  <div className="ml-2 space-y-1">
                    <div className="flex justify-between">
                      <span>• Pitch (yukarı/aşağı):</span>
                      <span className="font-medium">{analysis.face.headPose.pitch.toFixed(1)}°</span>
                    </div>
                    <div className="flex justify-between">
                      <span>• Yaw (sağ/sol):</span>
                      <span className="font-medium">{analysis.face.headPose.yaw.toFixed(1)}°</span>
                    </div>
                    <div className="flex justify-between">
                      <span>• Roll (eğim):</span>
                      <span className="font-medium">{analysis.face.headPose.roll.toFixed(1)}°</span>
                    </div>
                  </div>
                  <div className="flex justify-between">
                    <span>Göz Açıklık Oranı:</span>
                    <span className="font-medium">{analysis.face.eyeAspectRatio.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Ağız Açıklık Oranı:</span>
                    <span className="font-medium">{analysis.face.mouthAspectRatio.toFixed(3)}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Footer Info */}
      {analysis && (
        <div className="mt-6 text-center text-sm text-gray-600 bg-gray-50 p-4 rounded-lg">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <div>İşlem Süresi: <span className="font-medium">{analysis.processingTime.toFixed(1)}ms</span></div>
            <div>Model: <span className="font-medium">{analysis.modelVersion}</span></div>
            <div>Güncellenme: <span className="font-medium">{new Date(analysis.timestamp).toLocaleTimeString()}</span></div>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default EnhancedLiveCamera;

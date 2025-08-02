import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Brain,
  Clock,
  Eye,
  EyeOff,
  Heart,
  MapPin,
  Target,
  TrendingUp,
  Zap
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import Header from '../components/layout/Header';
import { Layout } from '../components/layout/layout';
import { GlassCard } from '../components/ui/glass-card';
import { NeonButton } from '../components/ui/neon-button';
import { getEndpointUrl, ACTIVE_SERVICE, checkServiceHealth } from '../config/aiServiceConfig';

// Enhanced interfaces from EnhancedLiveCamera
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

// Neural Network Background Animation Component
const NeuralNetworkBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const nodes: Array<{x: number, y: number, vx: number, vy: number}> = [];
    const connections: Array<{from: number, to: number, strength: number}> = [];
    
    // Create nodes
    for (let i = 0; i < 50; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5
      });
    }
    
    // Create connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (Math.random() < 0.1) {
          connections.push({
            from: i,
            to: j,
            strength: Math.random()
          });
        }
      }
    }
    
    let animationId: number;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Update nodes
      nodes.forEach(node => {
        node.x += node.vx;
        node.y += node.vy;
        
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
      });
      
      // Draw connections
      connections.forEach(conn => {
        const fromNode = nodes[conn.from];
        const toNode = nodes[conn.to];
        const distance = Math.sqrt(
          Math.pow(fromNode.x - toNode.x, 2) + Math.pow(fromNode.y - toNode.y, 2)
        );
        
        if (distance < 150) {
          const opacity = (1 - distance / 150) * conn.strength * 0.3;
          ctx.strokeStyle = `rgba(59, 130, 246, ${opacity})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          ctx.stroke();
        }
      });
      
      // Draw nodes
      nodes.forEach(node => {
        ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
        ctx.beginPath();
        ctx.arc(node.x, node.y, 2, 0, Math.PI * 2);
        ctx.fill();
      });
      
      animationId = requestAnimationFrame(animate);
    };
    
    animate();
    
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none opacity-20 z-0"
      style={{ mixBlendMode: 'screen' }}
    />
  );
};

// Face Detection Warning Overlay with Neon Flashing
const FaceDetectionWarning: React.FC<{ show: boolean }> = ({ show }) => {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-40 pointer-events-none"
        >
          {/* Neon flashing border effect */}
          <div className="absolute inset-0 border-4 border-red-500 animate-pulse shadow-2xl shadow-red-500/50" 
               style={{
                 animation: 'flash 0.5s infinite alternate',
                 boxShadow: '0 0 20px rgba(239, 68, 68, 0.5), inset 0 0 20px rgba(239, 68, 68, 0.1)'
               }}
          />
          
          {/* Central warning message */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="bg-red-900/80 backdrop-blur-md rounded-xl p-8 border border-red-500/50 shadow-2xl"
            >
              <div className="text-center">
                <EyeOff className="w-16 h-16 text-red-400 mx-auto mb-4 animate-bounce" />
                <h3 className="text-2xl font-bold text-white mb-2">
                  Yüz Algılanamıyor
                </h3>
                <p className="text-red-200">
                  Lütfen kameraya bakın ve iyi ışıklandırma sağlayın
                </p>
              </div>
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Enhanced Real-time Chart Components
const AttentionGauge: React.FC<{ score: number; state: string }> = ({ score, state }) => {
  const radius = 80;
  const strokeWidth = 12;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (score * circumference);
  
  const getStateColor = (state: string) => {
    switch (state) {
      case 'attentive': return '#10B981'; // emerald-500
      case 'distracted': return '#F59E0B'; // amber-500
      case 'drowsy': return '#EF4444'; // red-500
      case 'away': return '#6B7280'; // gray-500
      default: return '#6B7280';
    }
  };
  
  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
        {/* Background circle */}
        <circle
          cx="100"
          cy="100"
          r={radius}
          stroke="rgba(75, 85, 99, 0.3)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress circle */}
        <motion.circle
          cx="100"
          cy="100"
          r={radius}
          stroke={getStateColor(state)}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={strokeDasharray}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: "easeOut" }}
          style={{
            filter: `drop-shadow(0 0 8px ${getStateColor(state)}50)`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-3xl font-bold text-white">
            {Math.round(score * 100)}%
          </div>
          <div className="text-sm text-gray-300 capitalize">
            {state.replace('_', ' ')}
          </div>
        </div>
      </div>
    </div>
  );
};

const EmotionRadar: React.FC<{ emotions: Record<string, number> }> = ({ emotions }) => {
  const emotionList = ['Mutlu', 'Odaklanmış', 'Nötr', 'Şaşkın', 'Yorgun'];
  const emotionKeys = ['happy', 'focused', 'neutral', 'confused', 'tired'];
  
  return (
    <div className="relative w-64 h-64 mx-auto">
      <svg viewBox="0 0 200 200" className="w-full h-full">
        {/* Grid lines */}
        {[1, 2, 3, 4, 5].map(ring => (
          <circle
            key={ring}
            cx="100"
            cy="100"
            r={ring * 15}
            fill="none"
            stroke="rgba(75, 85, 99, 0.3)"
            strokeWidth="1"
          />
        ))}
        
        {/* Axis lines */}
        {emotionKeys.map((_, index) => {
          const angle = (index * 2 * Math.PI) / emotionKeys.length - Math.PI / 2;
          const x = 100 + Math.cos(angle) * 75;
          const y = 100 + Math.sin(angle) * 75;
          return (
            <line
              key={index}
              x1="100"
              y1="100"
              x2={x}
              y2={y}
              stroke="rgba(75, 85, 99, 0.3)"
              strokeWidth="1"
            />
          );
        })}
        
        {/* Emotion polygon */}
        <motion.polygon
          points={emotionKeys
            .map((key, index) => {
              const value = emotions[key] || 0;
              const angle = (index * 2 * Math.PI) / emotionKeys.length - Math.PI / 2;
              const x = 100 + Math.cos(angle) * (value * 75);
              const y = 100 + Math.sin(angle) * (value * 75);
              return `${x},${y}`;
            })
            .join(' ')}
          fill="rgba(59, 130, 246, 0.2)"
          stroke="rgb(59, 130, 246)"
          strokeWidth="2"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5 }}
        />
        
        {/* Emotion points */}
        {emotionKeys.map((key, index) => {
          const value = emotions[key] || 0;
          const angle = (index * 2 * Math.PI) / emotionKeys.length - Math.PI / 2;
          const x = 100 + Math.cos(angle) * (value * 75);
          const y = 100 + Math.sin(angle) * (value * 75);
          return (
            <circle
              key={key}
              cx={x}
              cy={y}
              r="4"
              fill="rgb(59, 130, 246)"
              className="drop-shadow-lg"
            />
          );
        })}
      </svg>
      
      {/* Labels */}
      {emotionList.map((emotion, index) => {
        const angle = (index * 2 * Math.PI) / emotionList.length - Math.PI / 2;
        const x = 50 + Math.cos(angle) * 45;
        const y = 50 + Math.sin(angle) * 45;
        return (
          <div
            key={emotion}
            className="absolute text-xs text-gray-300 text-center transform -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${x}%`, top: `${y}%` }}
          >
            {emotion}
          </div>
        );
      })}
    </div>
  );
};

const GazeHeatmap: React.FC<{ gazeHistory: Array<{x: number, y: number, timestamp: number}> }> = ({ gazeHistory }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw heatmap points
    gazeHistory.forEach((point, index) => {
      const age = (Date.now() - point.timestamp) / 1000; // age in seconds
      const opacity = Math.max(0, 1 - age / 30); // fade over 30 seconds
      
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 20, 0, 2 * Math.PI);
      
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
      gradient.addColorStop(0, `rgba(255, 0, 0, ${opacity * 0.6})`);
      gradient.addColorStop(1, `rgba(255, 0, 0, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.fill();
    });
  }, [gazeHistory]);
  
  return (
    <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
      <canvas
        ref={canvasRef}
        width={320}
        height={180}
        className="w-full h-full"
      />
      <div className="absolute inset-0 border-2 border-dashed border-blue-500/30 rounded-lg">
        <div className="absolute top-2 left-2 text-xs text-blue-400">
          Ekran Bakış Haritası
        </div>
      </div>
    </div>
  );
};

// Main Enhanced Demo Component
export const EnhancedTurkishDemo: React.FC = () => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [analysis, setAnalysis] = useState<ComprehensiveAnalysis | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  // History tracking
  const [gazeHistory, setGazeHistory] = useState<Array<{x: number, y: number, timestamp: number}>>([]);
  const [attentionHistory, setAttentionHistory] = useState<number[]>([]);
  const [emotionHistory, setEmotionHistory] = useState<string[]>([]);
  const [engagementHistory, setEngagementHistory] = useState<number[]>([]);
  
  // Calibration
  const [calibrationPoints, setCalibrationPoints] = useState<Array<{x: number, y: number}>>([]);
  const [currentCalibrationPoint, setCurrentCalibrationPoint] = useState(0);
  
  // Demo metrics
  const [sessionStartTime] = useState(Date.now());
  const [sessionDuration, setSessionDuration] = useState(0);
  const [totalFramesProcessed, setTotalFramesProcessed] = useState(0);
  
  // Enhanced visual states
  const [showNeonWarning, setShowNeonWarning] = useState(false);
  const [lastFaceDetectedTime, setLastFaceDetectedTime] = useState(Date.now());
  
  // Check backend status
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const isHealthy = await checkServiceHealth();
        if (isHealthy) {
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
  
  // Initialize camera
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
        setError('Kameraya erişim sağlanamıyor. Lütfen izinleri kontrol edin.');
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
  
  // Session timer
  useEffect(() => {
    const interval = setInterval(() => {
      setSessionDuration(Math.floor((Date.now() - sessionStartTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [sessionStartTime]);
  
  // Face detection warning system
  useEffect(() => {
    if (analysis) {
      if (analysis.face.detected) {
        setLastFaceDetectedTime(Date.now());
        setShowNeonWarning(false);
      } else {
        const timeSinceLastFace = Date.now() - lastFaceDetectedTime;
        if (timeSinceLastFace > 3000) { // 3 seconds without face
          setShowNeonWarning(true);
        }
      }
    }
  }, [analysis, lastFaceDetectedTime]);
  
  // Start gaze calibration
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
  
  // AI Analysis Loop
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
        
        const response = await fetch(getEndpointUrl('analyze'), {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        
        const data = await response.json();
        setAnalysis(data);
        setTotalFramesProcessed(prev => prev + 1);
        
        // Update history
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
        const errorMsg = err instanceof Error ? err.message : 'Analiz başarısız';
        setError(`Analiz hatası: ${errorMsg}. Backend bağlantısını kontrol edin.`);
        console.error('🔴 Analysis error:', err);
      }
    };
    
    const interval = setInterval(analyzeFrame, 100);
    return () => clearInterval(interval);
  }, [connected, isActive, calibrating]);
  
  // Video overlay rendering
  useEffect(() => {
    if (!analysis || !overlayCanvasRef.current || !videoRef.current) return;
    
    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;
    if (!ctx) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw gaze point
    if (analysis.gaze.onScreen) {
      const gazeX = analysis.gaze.x * canvas.width;
      const gazeY = analysis.gaze.y * canvas.height;
      
      ctx.beginPath();
      ctx.arc(gazeX, gazeY, 12, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(59, 130, 246, ${analysis.gaze.confidence})`;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      ctx.stroke();
      
      // Gaze direction indicator
      const directionMap = {
        'left': '←', 'right': '→', 'up': '↑', 'down': '↓', 'center': '●'
      };
      ctx.fillStyle = '#fff';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(directionMap[analysis.gaze.direction] || '●', gazeX, gazeY - 20);
    }
    
    // Draw gaze history trail
    if (gazeHistory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 4;
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
    
    // Draw face landmarks
    if (analysis.face.detected && analysis.face.landmarks) {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
      analysis.face.landmarks.slice(0, 10).forEach(landmark => {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
    
    // Draw attention focus regions
    analysis.attention.focusRegions.forEach(region => {
      ctx.strokeStyle = analysis.attention.state === 'attentive' ? '#10B981' : '#F59E0B';
      ctx.lineWidth = 4;
      ctx.strokeRect(
        region.x * canvas.width,
        region.y * canvas.height,
        region.width * canvas.width,
        region.height * canvas.height
      );
    });
    
  }, [analysis, gazeHistory]);
  
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  const getStateIcon = (state: string) => {
    switch (state) {
      case 'attentive': return <Eye className="w-5 h-5 text-green-500" />;
      case 'distracted': return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'drowsy': return <Clock className="w-5 h-5 text-orange-500" />;
      case 'away': return <EyeOff className="w-5 h-5 text-red-500" />;
      default: return <Target className="w-5 h-5 text-gray-500" />;
    }
  };
  
  return (
    <Layout>
      <Header />
      <NeuralNetworkBackground />
      <FaceDetectionWarning show={showNeonWarning} />
      
      <main className="relative z-10 container mx-auto px-4 pt-24 pb-8 min-h-screen">
        {/* Hero Section */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-primary-600/20 to-accent-cyan/20 border border-primary-500/30 mb-8 backdrop-blur-sm"
          >
            <Zap className="w-5 h-5 text-accent-cyan mr-2 animate-pulse" />
            <span className="text-primary-200 font-medium">
              Gerçek Zamanlı AI Analiz Sistemi
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-4xl md:text-6xl font-bold text-white mb-6"
          >
            <span className="bg-gradient-to-r from-primary-400 via-accent-cyan to-accent-purple bg-clip-text text-transparent">
              DersLens
            </span>
            <br />
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="text-lg md:text-xl text-gray-300 mb-8 max-w-3xl mx-auto"
          >
            Yüz tanıma, bakış takibi, duygu analizi ve dikkat seviyesi ölçümü ile 
            tam entegre eğitim deneyimi
          </motion.p>

          {/* Control Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8"
          >
            <NeonButton
              size="lg"
              variant={isActive ? "secondary" : "primary"}
              onClick={() => setIsActive(!isActive)}
              disabled={!connected}
              className="text-lg px-8 py-4"
            >
              {isActive ? (
                <>
                  <EyeOff className="w-5 h-5 mr-2" />
                  Analizi Durdur
                </>
              ) : (
                <>
                  <Eye className="w-5 h-5 mr-2" />
                  Analizi Başlat
                </>
              )}
            </NeonButton>

            <NeonButton
              size="lg"
              variant="accent"
              onClick={startGazeCalibration}
              disabled={!connected || isActive}
              className="text-lg px-8 py-4"
            >
              <Target className="w-5 h-5 mr-2" />
              Bakış Kalibrasyonu
            </NeonButton>
          </motion.div>
          
          {/* Status indicators */}
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <div className={`flex items-center px-4 py-2 rounded-full ${
              connected ? 'bg-green-900/40 text-green-300 border border-green-500/30' : 'bg-red-900/40 text-red-300 border border-red-500/30'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              {connected ? 'Kamera Bağlı' : 'Kamera Bağlantısı Yok'}
            </div>
            
            <div className={`flex items-center px-4 py-2 rounded-full ${
              backendStatus === 'online' ? 'bg-green-900/40 text-green-300 border border-green-500/30' : 
              backendStatus === 'offline' ? 'bg-red-900/40 text-red-300 border border-red-500/30' :
              'bg-yellow-900/40 text-yellow-300 border border-yellow-500/30'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                backendStatus === 'online' ? 'bg-green-500 animate-pulse' : 
                backendStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
              }`} />
              {backendStatus === 'online' ? 'AI Sistemi Aktif' : 
               backendStatus === 'offline' ? 'AI Sistemi Bağlantısız' :
               'AI Sistemi Kontrol Ediliyor...'}
            </div>
            
            {isActive && (
              <div className="flex items-center px-4 py-2 rounded-full bg-blue-900/40 text-blue-300 border border-blue-500/30">
                <Activity className="w-4 h-4 mr-2 animate-pulse" />
                Analiz Aktif
              </div>
            )}
          </div>
        </motion.section>

        {/* Main Video and Analysis Section */}
        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <GlassCard neonAccent padding="lg" className="relative">
              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full rounded-lg border-2 border-gray-600/30"
                />
                <canvas
                  ref={overlayCanvasRef}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none rounded-lg"
                />
                
                {/* Video overlay info */}
                {analysis && (
                  <div className="absolute top-4 left-4 space-y-2">
                    <div className="flex items-center px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm text-white text-sm">
                      {getStateIcon(analysis.attention.state)}
                      <span className="ml-2">
                        Dikkat: {(analysis.attention.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm text-white text-sm">
                      <Heart className="w-4 h-4 text-pink-400 mr-2" />
                      <span>
                        Duygu: {analysis.emotion.dominant}
                      </span>
                    </div>
                    
                    <div className="flex items-center px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm text-white text-sm">
                      <MapPin className="w-4 h-4 text-blue-400 mr-2" />
                      <span>
                        Bakış: {analysis.gaze.direction}
                      </span>
                    </div>
                  </div>
                )}
                
                {/* Processing info */}
                {analysis && (
                  <div className="absolute bottom-4 right-4 px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm text-white text-xs">
                    İşlem: {analysis.processingTime.toFixed(1)}ms
                  </div>
                )}
              </div>
            </GlassCard>
          </div>
          
          {/* Quick Stats */}
          <div className="space-y-6">
            <GlassCard neonAccent className="text-center">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center justify-center">
                <Clock className="w-5 h-5 mr-2 text-blue-400" />
                Oturum Bilgileri
              </h3>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-cyan">
                    {formatDuration(sessionDuration)}
                  </div>
                  <div className="text-gray-300">Süre</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-purple">
                    {totalFramesProcessed}
                  </div>
                  <div className="text-gray-300">Kare</div>
                </div>
                
                <div className="text-center col-span-2">
                  <div className="text-xl font-bold text-accent-emerald">
                    {gazeHistory.length}
                  </div>
                  <div className="text-gray-300">Bakış Noktası</div>
                </div>
              </div>
            </GlassCard>
            
            {analysis && (
              <GlassCard neonAccent className="text-center">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
                  Anlık Metrikler
                </h3>
                
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Katılım:</span>
                    <span className="text-white font-bold">
                      {(analysis.engagement.level * 100).toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Güven:</span>
                    <span className="text-white font-bold">
                      {(analysis.face.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Başpozisyonu:</span>
                    <span className="text-white font-bold text-xs">
                      Y:{analysis.face.headPose.yaw.toFixed(0)}° 
                      P:{analysis.face.headPose.pitch.toFixed(0)}°
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Göz Oranı:</span>
                    <span className="text-white font-bold">
                      {analysis.face.eyeAspectRatio.toFixed(2)}
                    </span>
                  </div>
                </div>
              </GlassCard>
            )}
          </div>
        </div>

        {/* Feature Cards - Turkish Focus */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12"
        >
          {/* Dikkat Takibi */}
          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-accent-cyan rounded-full flex items-center justify-center mx-auto mb-4">
              <Eye className="w-8 h-8 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Dikkat Takibi
            </h3>
            <p className="text-gray-300 text-sm mb-4">
              Yüz yönelimi ve varlık tespiti ile dikkat seviyesini ölçer
            </p>
            {analysis && (
              <AttentionGauge score={analysis.attention.score} state={analysis.attention.state} />
            )}
          </GlassCard>

          {/* Katılım Analizi */}
          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-cyan to-accent-purple rounded-full flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Katılım Analizi
            </h3>
            <p className="text-gray-300 text-sm mb-4">
              Hareket, ekran etkileşimi ve duruş analiziyle katılımı değerlendirir
            </p>
            <div className="h-24 flex items-end gap-1">
              {engagementHistory.map((value, index) => (
                <div
                  key={index}
                  className="flex-1 bg-gradient-to-t from-accent-cyan to-accent-purple rounded-t opacity-70 transition-all duration-500"
                  style={{ height: `${value * 100}%` }}
                />
              ))}
            </div>
          </GlassCard>

          {/* Duygu Tanıma */}
          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-purple to-accent-emerald rounded-full flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Duygu Tanıma
            </h3>
            <p className="text-gray-300 text-sm mb-4">
              Yüz mikroifadelerini analiz ederek duygu durumunu tespit eder
            </p>
            {analysis && (
              <EmotionRadar emotions={analysis.emotion.scores} />
            )}
          </GlassCard>

          {/* Bakış Haritalama */}
          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-emerald to-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <Target className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Bakış Haritalama
            </h3>
            <p className="text-gray-300 text-sm mb-4">
              Öğrencilerin ekranın hangi bölümüne odaklandığını takip eder
            </p>
            <GazeHeatmap gazeHistory={gazeHistory} />
          </GlassCard>
        </motion.section>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-6"
          >
            <div className="bg-red-900/40 border border-red-500/50 text-red-200 p-4 rounded-lg backdrop-blur-sm">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 mr-2" />
                <span>{error}</span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Calibration Modal */}
        <AnimatePresence>
          {calibrating && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50"
            >
              <GlassCard className="text-center max-w-md w-full mx-4">
                <h3 className="text-2xl font-bold text-white mb-4">
                  🎯 Bakış Kalibrasyonu
                </h3>
                <p className="text-gray-300 mb-6">
                  Kırmızı noktaya bakın ve hazır olduğunuzda 'İleri' butonuna tıklayın
                </p>
                <div className="text-lg text-accent-cyan mb-6">
                  Nokta {currentCalibrationPoint + 1}/9
                </div>
                <NeonButton
                  variant="primary"
                  onClick={nextCalibrationPoint}
                  className="w-full"
                >
                  {currentCalibrationPoint < calibrationPoints.length - 1 ? 'İleri' : 'Tamamla'}
                </NeonButton>
              </GlassCard>
              
              {calibrationPoints[currentCalibrationPoint] && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="fixed w-6 h-6 bg-red-500 rounded-full shadow-2xl shadow-red-500/50"
                  style={{
                    left: `${calibrationPoints[currentCalibrationPoint].x * 100}%`,
                    top: `${calibrationPoints[currentCalibrationPoint].y * 100}%`,
                    transform: 'translate(-50%, -50%)',
                    boxShadow: '0 0 20px rgba(239, 68, 68, 0.8), 0 0 40px rgba(239, 68, 68, 0.4)'
                  }}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
      
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      <style>
        {`
        @keyframes flash {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        
        .animate-pulse {
          animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        `}
      </style>
    </Layout>
  );
};

export default EnhancedTurkishDemo;

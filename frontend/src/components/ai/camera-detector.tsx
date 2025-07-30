import { motion } from 'framer-motion';
import {
  Activity,
  AlertCircle,
  Brain,
  Camera,
  CameraOff,
  CheckCircle,
  Eye,
  Heart,
  Loader
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { GlassCard } from '../ui/glass-card';
import { NeonButton } from '../ui/neon-button';

interface EmotionData {
  dominant: string;
  confidence: number;
  emotions: Record<string, number>;
  valence: number;
  arousal: number;
}

interface AttentionData {
  score: number;
  isAttentive: boolean;
  headPose: {
    pitch: number;
    yaw: number;
    roll: number;
  };
}

interface AnalysisResult {
  emotion?: EmotionData;
  attention?: AttentionData;
  engagement?: {
    score: number;
    level: string;
  };
  gaze?: {
    isLookingAtScreen: boolean;
    confidence: number;
  };
  faceDetected?: boolean;
  confidence?: number;
}

interface CameraDetectorProps {
  onAnalysisUpdate?: (result: AnalysisResult) => void;
  className?: string;
}

export const CameraDetector: React.FC<CameraDetectorProps> = ({
  onAnalysisUpdate,
  className
}) => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isActive, setIsActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null);
  const [fps, setFps] = useState(0);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 15 }
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsActive(true);
      }
    } catch (err) {
      setError('Kamera erişimi reddedildi. Lütfen izinleri kontrol edin.');
      console.error('Camera access denied:', err);
    }
  }, []);


  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsActive(false);
    setIsAnalyzing(false);
    setCurrentResult(null);
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isAnalyzing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.videoWidth === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    setIsAnalyzing(true);

    try {
      const response = await fetch('http://localhost:5000/api/v1/analyze/frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,  
          timestamp: Date.now(),
          sessionId: 'demo-session',
          options: {
            detectEmotion: true,
            detectAttention: true,
            detectEngagement: true,
            detectGaze: true
          }
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        const aiData = result.success ? result.data : result;
        
        const analysisResult: AnalysisResult = {
          emotion: aiData.emotion ? {
            dominant: aiData.emotion.dominant || 'neutral',
            confidence: aiData.emotion.confidence || 0,
            emotions: aiData.emotion.emotions || {},
            valence: aiData.emotion.valence || 0,
            arousal: aiData.emotion.arousal || 0
          } : undefined,
          attention: aiData.attention ? {
            score: aiData.attention.score || 0,
            isAttentive: aiData.attention.isAttentive || false,
            headPose: aiData.attention.headPose || { pitch: 0, yaw: 0, roll: 0 }
          } : undefined,
          engagement: aiData.engagement ? {
            score: aiData.engagement.score || 0,
            level: aiData.engagement.level || 'low'
          } : undefined,
          gaze: aiData.gaze ? {
            isLookingAtScreen: aiData.gaze.onScreen || false,
            confidence: aiData.gaze.confidence || 0
          } : undefined,
          faceDetected: aiData.metadata?.faceDetected || false,
          confidence: aiData.metadata?.averageConfidence || 0
        };
        
        setCurrentResult(analysisResult);
        onAnalysisUpdate?.(analysisResult);
      } else {
        console.warn('Analysis request failed:', response.status);
      }
    } catch (err) {
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, onAnalysisUpdate]);

  const startAnalysis = useCallback(() => {
    if (intervalRef.current) return;

    intervalRef.current = setInterval(captureAndAnalyze, 2000);
  }, [captureAndAnalyze]);

  const stopAnalysis = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const toggleCamera = useCallback(async () => {
    if (isActive) {
      stopAnalysis();
      stopCamera();
    } else {
      await startCamera();
    }
  }, [isActive, startCamera, stopCamera, stopAnalysis]);

  useEffect(() => {
    if (isActive && videoRef.current?.readyState === 4) {
      startAnalysis();
    }
  }, [isActive, startAnalysis]);

  useEffect(() => {
    return () => {
      stopAnalysis();
      stopCamera();
    };
  }, [stopAnalysis, stopCamera]);

  useEffect(() => {
    if (!isActive) {
      setFps(0);
      return;
    }

    let frameCount = 0;
    let lastTime = Date.now();

    const calculateFps = () => {
      frameCount++;
      const now = Date.now();
      if (now - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = now;
      }
      if (isActive) {
        requestAnimationFrame(calculateFps);
      }
    };

    requestAnimationFrame(calculateFps);
  }, [isActive]);

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      'Happy': 'text-emerald-400',
      'mutlu': 'text-emerald-400',
      'Sad': 'text-blue-400',
      'üzgün': 'text-blue-400',
      'Angry': 'text-red-400',
      'kızgın': 'text-red-400',
      'Fear': 'text-purple-400',
      'korkmuş': 'text-purple-400',
      'Surprise': 'text-yellow-400',
      'şaşkın': 'text-yellow-400',
      'Disgust': 'text-orange-400',
      'iğrenmiş': 'text-orange-400',
      'Neutral': 'text-gray-400',
      'nötr': 'text-gray-400'
    };
    return colors[emotion] || 'text-gray-400';
  };

  const getAttentionColor = (score: number) => {
    if (score >= 0.8) return 'text-emerald-400';
    if (score >= 0.6) return 'text-yellow-400';
    if (score >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className={className}>
      <GlassCard>
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Camera className="w-5 h-5 text-primary-400" />
              <h3 className="text-lg font-semibold text-white">
                {t('ai.camera.title', 'AI Kamera Analizi')}
              </h3>
            </div>
            <div className="flex items-center space-x-2">
              {isActive && (
                <div className="flex items-center space-x-1 text-xs text-gray-400">
                  <Activity className="w-3 h-3" />
                  <span>{fps} FPS</span>
                </div>
              )}
              <NeonButton
                onClick={toggleCamera}
                variant={isActive ? 'secondary' : 'primary'}
                size="sm"
              >
                {isActive ? (
                  <>
                    <CameraOff className="w-4 h-4 mr-1" />
                    {t('ai.camera.stop', 'Durdur')}
                  </>
                ) : (
                  <>
                    <Camera className="w-4 h-4 mr-1" />
                    {t('ai.camera.start', 'Başlat')}
                  </>
                )}
              </NeonButton>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center p-3 bg-red-500/20 border border-red-500/30 rounded-lg"
            >
              <AlertCircle className="w-4 h-4 text-red-400 mr-2" />
              <span className="text-red-300 text-sm">{error}</span>
            </motion.div>
          )}

          {/* Video Container */}
          <div className="relative">
            <div className="relative bg-black/50 rounded-lg overflow-hidden aspect-video">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover"
                onLoadedMetadata={() => {
                  console.log('Video metadata loaded');
                }}
              />
              
              {/* Analysis Overlay */}
              {isAnalyzing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                  <div className="flex items-center space-x-2 px-3 py-2 bg-black/50 rounded-lg">
                    <Loader className="w-4 h-4 text-primary-400 animate-spin" />
                    <span className="text-white text-sm">
                      {t('ai.camera.analyzing', 'Analiz ediliyor...')}
                    </span>
                  </div>
                </div>
              )}

              {/* Status Indicator */}
              <div className="absolute top-3 right-3">
                {isActive ? (
                  <div className="flex items-center space-x-1 px-2 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                    <span className="text-emerald-300 text-xs">CANLI</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-1 px-2 py-1 bg-gray-500/20 border border-gray-500/30 rounded-full">
                    <div className="w-2 h-2 bg-gray-400 rounded-full" />
                    <span className="text-gray-300 text-xs">KAPALI</span>
                  </div>
                )}
              </div>
            </div>

            {/* Hidden canvas for frame capture */}
            <canvas
              ref={canvasRef}
              className="hidden"
            />
          </div>

          {/* Analysis Results */}
          {currentResult && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-4"
            >
              {/* Emotion Analysis */}
              {currentResult.emotion && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Heart className="w-4 h-4 text-pink-400" />
                    <span className="text-sm font-medium text-white">
                      {t('ai.emotion.title', 'Duygu Analizi')}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <div className={`text-lg font-semibold ${getEmotionColor(currentResult.emotion.dominant)}`}>
                      {currentResult.emotion.dominant}
                    </div>
                    <div className="text-xs text-gray-400">
                      %{(currentResult.emotion.confidence * 100).toFixed(1)} güven
                    </div>
                  </div>
                </div>
              )}

              {/* Attention Analysis */}
              {currentResult.attention && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Eye className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-medium text-white">
                      {t('ai.attention.title', 'Dikkat Analizi')}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <div className={`text-lg font-semibold ${getAttentionColor(currentResult.attention.score)}`}>
                      %{(currentResult.attention.score * 100).toFixed(1)}
                    </div>
                    <div className="flex items-center space-x-1">
                      {currentResult.attention.isAttentive ? (
                        <CheckCircle className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-3 h-3 text-red-400" />
                      )}
                      <span className="text-xs text-gray-400">
                        {currentResult.attention.isAttentive ? 'Dikkatli' : 'Dikkatsiz'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Engagement Analysis */}
              {currentResult.engagement && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-medium text-white">
                      {t('ai.engagement.title', 'Katılım Analizi')}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <div className="text-lg font-semibold text-purple-400">
                      %{(currentResult.engagement.score * 100).toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-400 capitalize">
                      {currentResult.engagement.level}
                    </div>
                  </div>
                </div>
              )}

              {/* Gaze Analysis */}
              {currentResult.gaze && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Eye className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-medium text-white">
                      {t('ai.gaze.title', 'Bakış Analizi')}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <div className={`text-sm font-medium ${currentResult.gaze.isLookingAtScreen ? 'text-emerald-400' : 'text-red-400'}`}>
                      {currentResult.gaze.isLookingAtScreen ? 'Ekrana bakıyor' : 'Ekrana bakmıyor'}
                    </div>
                    <div className="text-xs text-gray-400">
                      %{(currentResult.gaze.confidence * 100).toFixed(1)} güven
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </GlassCard>
    </div>
  );
};

import { motion } from 'framer-motion';
import {
    Activity,
    Brain,
    Camera,
    CameraOff,
    Eye,
    Loader2,
    Play,
    Square,
    Users,
    Zap
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
interface StudentAnalysisResult {
  emotion: {
    dominant_emotion: string;
    confidence: number;
    emotions: Record<string, number>;
  };
  attention: {
    attention_level: string;
    attention_score: number;
    focus_duration: number;
    engagement_quality: string;
  };
  engagement: {
    engagement_state: string;
    engagement_score: number;
    learning_efficiency: string;
  };
  processing_time: number;
  face_count: number;
}
interface OptimizedCameraAnalysisProps {
  onAnalysisResult?: (result: StudentAnalysisResult) => void;
  analysisInterval?: number;
  autoStart?: boolean;
}
export function OptimizedCameraAnalysis({ 
  onAnalysisResult, 
  analysisInterval = 2000,
  autoStart = false 
}: OptimizedCameraAnalysisProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<StudentAnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<StudentAnalysisResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
  const initializeCamera = useCallback(async () => {
    try {
      setCameraError(null);
      setConnectionStatus('connecting');
      const constraints = {
        video: {
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          frameRate: { ideal: 15, max: 30 },
          facingMode: 'user'
        },
        audio: false
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsInitialized(true);
          setConnectionStatus('connected');
        };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Kamera erişimi başarısız';
      setCameraError(errorMessage);
      setConnectionStatus('error');
      console.error('Camera initialization failed:', error);
    }
  }, []);
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsInitialized(false);
    setIsAnalyzing(false);
    setConnectionStatus('connecting');
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
  }, []);
  const getDefaultAnalysisData = (): StudentAnalysisResult => ({
    emotion: {
      dominant_emotion: 'neutral',
      confidence: 0,
      emotions: { neutral: 1 }
    },
    attention: {
      attention_level: 'medium',
      attention_score: 0.5,
      focus_duration: 0,
      engagement_quality: 'medium'
    },
    engagement: {
      engagement_state: 'neutral',
      engagement_score: 0.5,
      learning_efficiency: 'medium'
    },
    processing_time: 0,
    face_count: 0
  });
  const analyzeFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isInitialized) {
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }
    const maxWidth = 640;
    const maxHeight = 480;
    const aspectRatio = video.videoWidth / video.videoHeight;
    let canvasWidth = Math.min(video.videoWidth, maxWidth);
    let canvasHeight = Math.min(video.videoHeight, maxHeight);
    if (canvasWidth / canvasHeight > aspectRatio) {
      canvasWidth = canvasHeight * aspectRatio;
    } else {
      canvasHeight = canvasWidth / aspectRatio;
    }
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
    const imageData = canvas.toDataURL('image/jpeg', 0.6);
    const base64Image = imageData.split(',')[1];
    const startTime = performance.now();
    let analysisData: StudentAnalysisResult = getDefaultAnalysisData();
    try {
      const response = await fetch('http://localhost:8001/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          analysis_type: 'all'
        })
      });
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      const result = await response.json();
      const endTime = performance.now();
      const responseData = result.data || result;
      analysisData = {
        ...getDefaultAnalysisData(),
        ...responseData,
        emotion: {
          ...getDefaultAnalysisData().emotion,
          ...(responseData.emotion || {})
        },
        attention: {
          ...getDefaultAnalysisData().attention,
          ...(responseData.attention || {})
        },
        engagement: {
          ...getDefaultAnalysisData().engagement,
          ...(responseData.engagement || {})
        },
        processing_time: (endTime - startTime) / 1000,
        face_count: responseData.metadata?.face_count || 
                   (responseData.metadata?.faceDetected ? 1 : 0) || 
                   0
      };
      setCurrentResult(analysisData);
      setAnalysisHistory(prev => {
        const newHistory = [analysisData, ...prev.slice(0, 9)]; 
        return newHistory;
      });
      if (onAnalysisResult) {
        onAnalysisResult(analysisData);
      }
    } catch (error) {
      console.error('Frame analysis failed:', error);
      setCameraError(error instanceof Error ? error.message : 'Analiz hatası');
    }
  }, [isInitialized, onAnalysisResult]);
  const startAnalysis = useCallback(() => {
    if (!isInitialized) return;
    setIsAnalyzing(true);
    setCameraError(null);
    analysisIntervalRef.current = setInterval(analyzeFrame, analysisInterval);
  }, [isInitialized, analyzeFrame, analysisInterval]);
  const stopAnalysis = useCallback(() => {
    setIsAnalyzing(false);
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
  }, []);
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);
  useEffect(() => {
    if (autoStart && !isInitialized) {
      initializeCamera();
    }
  }, [autoStart, isInitialized, initializeCamera]);
  useEffect(() => {
    if (autoStart && isInitialized && !isAnalyzing) {
      startAnalysis();
    }
  }, [autoStart, isInitialized, isAnalyzing, startAnalysis]);
  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      neutral: '#6B7280',
      happiness: '#10B981',
      surprise: '#F59E0B',
      sadness: '#3B82F6',
      anger: '#EF4444',
      disgust: '#F97316',
      fear: '#8B5CF6',
      contempt: '#EC4899'
    };
    return colors[emotion] || '#6B7280';
  };
  const getEmotionLabel = (emotion: string) => {
    const labels: Record<string, string> = {
      neutral: 'Nötr',
      happiness: 'Mutlu',
      surprise: 'Şaşırmış',
      sadness: 'Üzgün',
      anger: 'Kızgın',
      disgust: 'Tiksinti',
      fear: 'Korku',
      contempt: 'Küçümseme'
    };
    return labels[emotion] || emotion;
  };
  const getAttentionColor = (level: string) => {
    const colors: Record<string, string> = {
      high: '#10B981',
      medium: '#F59E0B',
      low: '#EF4444',
      very_high: '#059669',
      very_low: '#DC2626'
    };
    return colors[level] || '#6B7280';
  };
  const getEngagementColor = (state: string) => {
    const colors: Record<string, string> = {
      engaged: '#10B981',
      neutral: '#F59E0B',
      disengaged: '#EF4444',
      highly_engaged: '#059669',
      distracted: '#DC2626'
    };
    return colors[state] || '#6B7280';
  };
  return (
    <div className="space-y-6">
      {}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center bg-gradient-to-r from-blue-600/20 to-purple-600/20 backdrop-blur-sm rounded-2xl border border-blue-500/30 p-8"
      >
        <div className="flex items-center justify-center mb-4">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity }}
            className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mr-4"
          >
            <Camera className="w-8 h-8 text-white" />
          </motion.div>
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">
              🚀 Canlı AI Model Testi
            </h2>
            <p className="text-blue-200 text-lg">
              Gerçek zamanlı öğrenci analizi - Duygu, Dikkat ve Katılım tespiti
            </p>
          </div>
        </div>
        <div className="flex items-center justify-center gap-6 text-sm text-blue-200">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4" />
            <span>FER2013+ Model</span>
          </div>
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4" />
            <span>DAISEE Dikkat</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            <span>Mendeley Katılım</span>
          </div>
        </div>
      </motion.div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gray-900/90 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6 h-full"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <Camera className="w-6 h-6 text-blue-400" />
                Öğrenci Kamerası
              </h3>
              <div className="flex items-center gap-3">
                <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
                  connectionStatus === 'connected' ? 'bg-green-500/20 text-green-400' :
                  connectionStatus === 'connecting' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-400' :
                    connectionStatus === 'connecting' ? 'bg-yellow-400' :
                    'bg-red-400'
                  }`} />
                  {connectionStatus === 'connected' ? 'Bağlı' :
                   connectionStatus === 'connecting' ? 'Bağlanıyor' : 'Hata'}
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="relative rounded-lg overflow-hidden bg-black aspect-video">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  muted
                  playsInline
                />
                <canvas ref={canvasRef} className="hidden" />
                {}
                {isAnalyzing && currentResult && (
                  <div className="absolute top-4 left-4 space-y-2">
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="bg-black/80 backdrop-blur-sm rounded-lg p-3 text-white"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                        <span className="text-sm font-medium">Canlı Analiz</span>
                      </div>
                      <div className="text-xs space-y-1">
                        <div className="flex items-center gap-2">
                          <Brain className="w-3 h-3" />
                          <span>{getEmotionLabel(currentResult.emotion?.dominant_emotion || 'neutral')}</span>
                          <span className="text-green-400">
                            %{((currentResult.emotion?.confidence || 0) * 100).toFixed(0)}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Eye className="w-3 h-3" />
                          <span>Dikkat: {currentResult.attention?.attention_level || 'yükleniyor'}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Activity className="w-3 h-3" />
                          <span>Katılım: {currentResult.engagement?.engagement_state || 'yükleniyor'}</span>
                        </div>
                      </div>
                    </motion.div>
                  </div>
                )}
                {cameraError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                    <div className="text-center text-white p-6">
                      <Camera className="w-12 h-12 text-red-400 mx-auto mb-2" />
                      <p className="text-sm mb-4">{cameraError}</p>
                      <button
                        onClick={initializeCamera}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
                      >
                        Tekrar Dene
                      </button>
                    </div>
                  </div>
                )}
              </div>
              {}
              <div className="flex items-center justify-center gap-4">
                {!isInitialized ? (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={initializeCamera}
                    className="flex items-center gap-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-medium transition-all duration-300 shadow-lg"
                  >
                    <Camera className="w-5 h-5" />
                    Kamerayı Başlat
                  </motion.button>
                ) : (
                  <div className="flex items-center gap-3">
                    {!isAnalyzing ? (
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={startAnalysis}
                        className="flex items-center gap-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-8 py-4 rounded-xl font-medium transition-all duration-300 shadow-lg"
                      >
                        <Play className="w-5 h-5" />
                        Analizi Başlat
                      </motion.button>
                    ) : (
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={stopAnalysis}
                        className="flex items-center gap-3 bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white px-8 py-4 rounded-xl font-medium transition-all duration-300 shadow-lg"
                      >
                        <Square className="w-5 h-5" />
                        Analizi Durdur
                      </motion.button>
                    )}
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={stopCamera}
                      className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 text-white px-6 py-4 rounded-xl font-medium transition-colors"
                    >
                      <CameraOff className="w-5 h-5" />
                      Durdur
                    </motion.button>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
        {}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gray-900/90 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Users className="w-5 h-5 text-green-400" />
              Öğrenci Analizi
            </h3>
            {currentResult ? (
              <div className="space-y-4">
                {}
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Brain className="w-4 h-4 text-blue-400" />
                    <h4 className="text-sm font-medium text-gray-300">Duygu Durumu</h4>
                  </div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: getEmotionColor(currentResult.emotion.dominant_emotion) }}
                      />
                      <span className="text-white font-medium text-lg">
                        {getEmotionLabel(currentResult.emotion.dominant_emotion)}
                      </span>
                    </div>
                    <span className="text-gray-400 text-sm bg-gray-700/50 px-2 py-1 rounded">
                      %{(currentResult.emotion.confidence * 100).toFixed(0)}
                    </span>
                  </div>
                </div>
                {}
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Eye className="w-4 h-4 text-green-400" />
                    <h4 className="text-sm font-medium text-gray-300">Dikkat Seviyesi</h4>
                  </div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium text-lg capitalize">
                      {currentResult.attention.attention_level}
                    </span>
                    <span className="text-gray-400 text-sm bg-gray-700/50 px-2 py-1 rounded">
                      %{(currentResult.attention.attention_score * 100).toFixed(0)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${currentResult.attention.attention_score * 100}%` }}
                      className="h-2 rounded-full"
                      style={{ backgroundColor: getAttentionColor(currentResult.attention.attention_level) }}
                    />
                  </div>
                </div>
                {}
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <h4 className="text-sm font-medium text-gray-300">Katılım Durumu</h4>
                  </div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium text-lg capitalize">
                      {currentResult.engagement.engagement_state}
                    </span>
                    <span className="text-gray-400 text-sm bg-gray-700/50 px-2 py-1 rounded">
                      %{(currentResult.engagement.engagement_score * 100).toFixed(0)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${currentResult.engagement.engagement_score * 100}%` }}
                      className="h-2 rounded-full"
                      style={{ backgroundColor: getEngagementColor(currentResult.engagement.engagement_state) }}
                    />
                  </div>
                </div>
                {}
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="w-4 h-4 text-yellow-400" />
                    <h4 className="text-sm font-medium text-gray-300">Performans</h4>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">İşlem Süresi</span>
                      <span className="text-white">{currentResult.processing_time.toFixed(2)}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Yüz Tespiti</span>
                      <span className={currentResult.face_count > 0 ? 'text-green-400' : 'text-red-400'}>
                        {currentResult.face_count > 0 ? '✓ Başarılı' : '✗ Tespit edilemedi'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-gray-400">
                {isAnalyzing ? (
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
                    <span className="text-sm">AI analiz yapıyor...</span>
                    <span className="text-xs text-gray-500">
                      Duygu, dikkat ve katılım analizi
                    </span>
                  </div>
                ) : (
                  <div className="text-center">
                    <Camera className="w-12 h-12 text-gray-500 mx-auto mb-3" />
                    <span className="text-sm">Analiz başlatılmadı</span>
                    <p className="text-xs text-gray-500 mt-1">
                      Kamerayı başlatın ve analizi başlayın
                    </p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
          {}
          {analysisHistory.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-900/90 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6"
            >
              <h3 className="text-lg font-bold text-white mb-4">Son Analizler</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {analysisHistory.slice(0, 5).map((result, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg text-sm"
                  >
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getEmotionColor(result.emotion.dominant_emotion) }}
                      />
                      <span className="text-gray-300">
                        {getEmotionLabel(result.emotion.dominant_emotion)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      <span>D:{result.attention.attention_level}</span>
                      <span>K:{result.engagement.engagement_state}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
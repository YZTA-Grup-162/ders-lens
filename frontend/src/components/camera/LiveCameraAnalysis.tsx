import { motion } from 'framer-motion';
import {
  Camera,
  CameraOff,
  CheckCircle,
  Loader2,
  Video,
  VideoOff,
  XCircle
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
interface CameraAnalysisResult {
  emotion: {
    model: string;
    emotions: Record<string, number>;
    dominant_emotion: string;
    confidence: number;
  };
  attention: {
    model: string;
    attention_level: string;
    attention_score: number;
    focus_duration: number;
    distraction_count: number;
    engagement_quality: string;
  };
  engagement: {
    model: string;
    engagement_state: string;
    engagement_score: number;
    interaction_frequency: number;
    cognitive_load: number;
    learning_efficiency: string;
  };
  processing_time: number;
  face_count: number;
}
interface LiveCameraAnalysisProps {
  onAnalysisResult?: (result: CameraAnalysisResult) => void;
  analysisInterval?: number;
}
export function LiveCameraAnalysis({ 
  onAnalysisResult, 
  analysisInterval = 1000 
}: LiveCameraAnalysisProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<CameraAnalysisResult | null>(null);
  const [fps, setFps] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const lastFrameTime = useRef(0);
  const frameCount = useRef(0);
  const initializeCamera = useCallback(async () => {
    try {
      setCameraError(null);
      const constraints = {
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 30 },
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
        };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Kamera erişimi başarısız';
      setCameraError(errorMessage);
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
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
  }, []);
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
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Image = imageData.split(',')[1];
    const startTime = performance.now();
    try {
      const response = await fetch('/api/v1/analyze/frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          timestamp: Date.now(),
          sessionId: `camera-session-${Date.now()}`,
          options: {
            detectEmotion: true,
            detectAttention: true,
            detectEngagement: true,
            detectGaze: true
          }
        })
      });
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      const result = await response.json();
      console.log('🔍 AI Analysis Result:', result); 
      const endTime = performance.now();
      const processingTime = endTime - startTime;
      setProcessingTime(processingTime);
      const analysisData = result.data || result;
      setCurrentResult(analysisData);
      if (onAnalysisResult) {
        onAnalysisResult(analysisData);
      }
      const now = performance.now();
      if (lastFrameTime.current > 0) {
        const timeDiff = now - lastFrameTime.current;
        frameCount.current++;
        if (frameCount.current >= 10) { 
          const avgFrameTime = timeDiff / frameCount.current;
          setFps(Math.round(1000 / avgFrameTime));
          frameCount.current = 0;
        }
      }
      lastFrameTime.current = now;
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
  return (
    <div className="bg-gray-900/90 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <Camera className="w-6 h-6 text-blue-400" />
          Canlı Kamera Analizi
        </h3>
        <div className="flex items-center gap-2">
          {isInitialized && (
            <div className="flex items-center gap-1 text-green-400 text-sm">
              <CheckCircle className="w-4 h-4" />
              Bağlı
            </div>
          )}
          <div className="text-gray-400 text-sm">
            {fps > 0 && `${fps} FPS`}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {}
        <div className="space-y-4">
          <div className="relative rounded-lg overflow-hidden bg-black aspect-video">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              muted
              playsInline
            />
            <canvas
              ref={canvasRef}
              className="hidden"
            />
            {}
            <div className="absolute top-4 left-4 flex items-center gap-2">
              {isAnalyzing && (
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="w-3 h-3 bg-red-500 rounded-full"
                />
              )}
              <span className="text-white text-sm bg-black/50 px-2 py-1 rounded">
                {isAnalyzing ? 'Analiz Yapılıyor' : 'Bekleniyor'}
              </span>
            </div>
            {cameraError && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                <div className="text-center text-white">
                  <XCircle className="w-12 h-12 text-red-400 mx-auto mb-2" />
                  <p className="text-sm">{cameraError}</p>
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
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
              >
                <Video className="w-5 h-5" />
                Kamerayı Başlat
              </motion.button>
            ) : (
              <div className="flex items-center gap-2">
                {!isAnalyzing ? (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={startAnalysis}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                  >
                    <Camera className="w-5 h-5" />
                    Analizi Başlat
                  </motion.button>
                ) : (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={stopAnalysis}
                    className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                  >
                    <CameraOff className="w-5 h-5" />
                    Analizi Durdur
                  </motion.button>
                )}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={stopCamera}
                  className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-3 rounded-lg font-medium transition-colors"
                >
                  <VideoOff className="w-5 h-5" />
                </motion.button>
              </div>
            )}
          </div>
        </div>
        {}
        <div className="space-y-4">
          <h4 className="text-lg font-semibold text-white mb-4">Anlık Sonuçlar</h4>
          {currentResult ? (
            <div className="space-y-4">
              {}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h5 className="text-sm font-medium text-gray-300 mb-2">Duygu Analizi</h5>
                <div className="flex items-center gap-3 mb-2">
                  <div 
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: getEmotionColor(currentResult.emotion.dominant_emotion) }}
                  />
                  <span className="text-white font-medium">
                    {getEmotionLabel(currentResult.emotion.dominant_emotion)}
                  </span>
                  <span className="text-gray-400 text-sm">
                    %{(currentResult.emotion.confidence * 100).toFixed(1)}
                  </span>
                </div>
                <div className="space-y-1">
                  {Object.entries(currentResult.emotion.emotions).map(([emotion, confidence]) => (
                    <div key={emotion} className="flex items-center justify-between text-sm">
                      <span className="text-gray-300">{getEmotionLabel(emotion)}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${confidence * 100}%` }}
                            transition={{ duration: 0.5 }}
                            className="h-full rounded-full"
                            style={{ backgroundColor: getEmotionColor(emotion) }}
                          />
                        </div>
                        <span className="text-gray-400 w-10 text-right">
                          {(confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h5 className="text-sm font-medium text-gray-300 mb-2">Dikkat Analizi</h5>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Dikkat Seviyesi</span>
                    <span className="text-white font-medium capitalize">
                      {currentResult.attention.attention_level}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Dikkat Skoru</span>
                    <span className="text-white font-medium">
                      %{(currentResult.attention.attention_score * 100).toFixed(1)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Odaklanma Süresi</span>
                    <span className="text-white font-medium">
                      {currentResult.attention.focus_duration.toFixed(1)}s
                    </span>
                  </div>
                  <div className="text-sm text-gray-400">
                    {currentResult.attention.engagement_quality}
                  </div>
                </div>
              </div>
              {}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h5 className="text-sm font-medium text-gray-300 mb-2">Katılım Analizi</h5>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Katılım Durumu</span>
                    <span className="text-white font-medium capitalize">
                      {currentResult.engagement.engagement_state}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Katılım Skoru</span>
                    <span className="text-white font-medium">
                      %{(currentResult.engagement.engagement_score * 100).toFixed(1)}
                    </span>
                  </div>
                  <div className="text-sm text-gray-400">
                    Öğrenme: {currentResult.engagement.learning_efficiency}
                  </div>
                </div>
              </div>
              {}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h5 className="text-sm font-medium text-gray-300 mb-2">Performans</h5>
                <div className="space-y-1 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">İşlem Süresi</span>
                    <span className="text-white">{currentResult.processing_time.toFixed(3)}s</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Yüz Algılandı</span>
                    <span className={currentResult.face_count > 0 ? 'text-green-400' : 'text-red-400'}>
                      {currentResult.face_count > 0 ? 'Evet' : 'Hayır'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Yüz Sayısı</span>
                    <span className="text-white">
                      {currentResult.face_count}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-400">
              {isAnalyzing ? (
                <div className="flex items-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin" />
                  <span>Analiz yapılıyor...</span>
                </div>
              ) : (
                <span>Analiz sonuçları burada görünecek</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
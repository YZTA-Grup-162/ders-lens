import { motion } from 'framer-motion';
import { Activity, Camera, Eye, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { AnalysisResult, apiService } from '../services/apiService';
interface LiveCameraProps {
  onAnalysisResult?: (result: AnalysisResult) => void;
  analysisInterval?: number;
}
const LiveCamera = ({ onAnalysisResult, analysisInterval = 1000 }: LiveCameraProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout>();
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const startCamera = async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsRecording(true);
      toast.success('Kamera başlatıldı');
      startAnalysis();
      setTimeout(() => {
        console.log('Triggering immediate analysis after camera start...');
        analyzeFrame();
      }, 1000);
    } catch (error) {
      console.error('Kamera erişim hatası:', error);
      const errorMessage = error instanceof Error ? error.message : 'Kamera erişimi reddedildi';
      setError(errorMessage);
      toast.error('Kamera erişimi reddedildi');
    }
  };
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsRecording(false);
    setIsAnalyzing(false);
    setError(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    toast.success('Kamera durduruldu');
  };
  const captureFrame = async (): Promise<Blob | null> => {
    if (!videoRef.current || !canvasRef.current) return null;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob);
      }, 'image/jpeg', 0.8);
    });
  };
  const analyzeFrame = async () => {
    console.log('analyzeFrame called - isRecording:', isRecording, 'isAnalyzing:', isAnalyzing);
    if (!isRecording || isAnalyzing) {
      console.log('Skipping analysis - conditions not met');
      return;
    }
    console.log('Starting frame analysis...');
    setIsAnalyzing(true);
    try {
      const frameBlob = await captureFrame();
      if (!frameBlob) {
        console.log('No frame captured');
        return;
      }
      console.log('Frame captured, sending for analysis...');
      const result = await apiService.analyzeFrame(frameBlob);
      console.log('Analysis result received:', result);
      onAnalysisResult?.(result);
    } catch (error) {
      console.error('Analiz hatası:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };
  const startAnalysis = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(analyzeFrame, analysisInterval);
  };
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);
  return (
    <motion.div
      className="bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-white/20 dark:border-gray-700/20"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Canlı Kamera</h2>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {isRecording ? 'Aktif' : 'Bekleniyor'}
          </span>
        </div>
      </div>
      <div className="flex space-x-2 mb-6">
        {!isRecording ? (
          <button
            onClick={startCamera}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Camera className="w-4 h-4 mr-2" />
            Başlat
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            <Square className="w-4 h-4 mr-2" />
            Durdur
          </button>
        )}
      </div>
      <div className="relative aspect-video bg-gray-900 rounded-2xl overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        {!isRecording && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 backdrop-blur-sm">
            <div className="text-center text-white">
              <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Analizi başlatmak için kamerayı açın</p>
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-red-900/50 backdrop-blur-sm">
            <div className="text-center text-white">
              <div className="w-16 h-16 mx-auto mb-4 bg-red-600 rounded-full flex items-center justify-center">
                <Eye className="w-8 h-8" />
              </div>
              <p className="font-semibold mb-2">Kamera Hatası</p>
              <p className="text-sm opacity-75">{error}</p>
            </div>
          </div>
        )}
        {isAnalyzing && (
          <div className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-1 rounded-full text-sm flex items-center">
            <Activity className="w-4 h-4 mr-1 animate-pulse" />
            Analiz ediliyor
          </div>
        )}
        {isRecording && !isAnalyzing && (
          <div className="absolute top-4 left-4 bg-green-600 text-white px-3 py-1 rounded-full text-sm flex items-center">
            <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
            Canlı
          </div>
        )}
      </div>
      <canvas ref={canvasRef} className="hidden" />
    </motion.div>
  );
};
export default LiveCamera;
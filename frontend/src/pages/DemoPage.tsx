import { motion } from 'framer-motion';
import { Camera, Pause, Play, RotateCcw } from 'lucide-react';
import { useRef, useState } from 'react';
import toast from 'react-hot-toast';

interface AnalysisData {
  attention: number;
  engagement: number;
  emotion: string;
  gaze: { x: number; y: number };
}

const DemoPage = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisData>({
    attention: 0,
    engagement: 0,
    emotion: 'neutral',
    gaze: { x: 0, y: 0 }
  });
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsRecording(true);
      toast.success('Kamera başlatıldı');
      
      startAnalysis();
    } catch (error) {
      toast.error('Kamera erişimi reddedildi');
      console.error('Camera error:', error);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsRecording(false);
    toast.success('Kamera durduruldu');
  };

  const startAnalysis = () => {
    const interval = setInterval(() => {
      setAnalysisData({
        attention: Math.floor(Math.random() * 100),
        engagement: Math.floor(Math.random() * 100),
        emotion: ['happy', 'neutral', 'focused', 'confused'][Math.floor(Math.random() * 4)],
        gaze: {
          x: Math.random() * 100,
          y: Math.random() * 100
        }
      });
    }, 2000);

    return () => clearInterval(interval);
  };

  const resetDemo = () => {
    stopCamera();
    setAnalysisData({
      attention: 0,
      engagement: 0,
      emotion: 'neutral',
      gaze: { x: 0, y: 0 }
    });
  };

  const getEmotionColor = (emotion: string) => {
    switch (emotion) {
      case 'happy': return 'text-green-500';
      case 'focused': return 'text-blue-500';
      case 'confused': return 'text-orange-500';
      default: return 'text-gray-500';
    }
  };

  const getEmotionEmoji = (emotion: string) => {
    switch (emotion) {
      case 'happy': return '😊';
      case 'focused': return '🎯';
      case 'confused': return '😕';
      default: return '😐';
    }
  };

  const getEmotionText = (emotion: string) => {
    switch (emotion) {
      case 'happy': return 'Mutlu';
      case 'focused': return 'Odaklanmış';
      case 'confused': return 'Kafası Karışık';
      default: return 'Nötr';
    }
  };

  return (
    <div className="min-h-screen pt-20 bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            DersLens Demo
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Yapay zeka destekli öğrenci analizi sistemini test edin
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Camera Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
                Kamera Feed
              </h2>
              <div className="flex space-x-2">
                {!isRecording ? (
                  <button
                    onClick={startCamera}
                    className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Başlat
                  </button>
                ) : (
                  <button
                    onClick={stopCamera}
                    className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    <Pause className="h-4 w-4 mr-2" />
                    Durdur
                  </button>
                )}
                <button
                  onClick={resetDemo}
                  className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Sıfırla
                </button>
              </div>
            </div>

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
                className="absolute inset-0 w-full h-full"
              />
              {!isRecording && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-300">Kamerayı başlatmak için butonlara tıklayın</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          {/* Analysis Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="space-y-6"
          >
            {/* Attention Meter */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Dikkat Seviyesi
              </h3>
              <div className="relative">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <motion.div
                    className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${analysisData.attention}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-2xl font-bold text-gray-900 dark:text-white mt-2 block">
                  %{analysisData.attention}
                </span>
              </div>
            </div>

            {/* Engagement Meter */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Katılım Seviyesi
              </h3>
              <div className="relative">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <motion.div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${analysisData.engagement}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-2xl font-bold text-gray-900 dark:text-white mt-2 block">
                  %{analysisData.engagement}
                </span>
              </div>
            </div>

            {/* Emotion Detection */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Duygu Durumu
              </h3>
              <div className="flex items-center space-x-4">
                <span className="text-4xl">
                  {getEmotionEmoji(analysisData.emotion)}
                </span>
                <div>
                  <span className={`text-2xl font-bold ${getEmotionColor(analysisData.emotion)}`}>
                    {getEmotionText(analysisData.emotion)}
                  </span>
                </div>
              </div>
            </div>

            {/* Gaze Tracking */}
            <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-200/20 dark:border-gray-700/20 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Bakış Haritası
              </h3>
              <div className="relative bg-gray-100 dark:bg-gray-800 rounded-lg h-32">
                <motion.div
                  className="absolute w-3 h-3 bg-red-500 rounded-full"
                  animate={{
                    left: `${analysisData.gaze.x}%`,
                    top: `${analysisData.gaze.y}%`
                  }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Kırmızı nokta: Mevcut bakış odağı
              </p>
            </div>
          </motion.div>
        </div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-8 bg-blue-50 dark:bg-blue-900/20 rounded-2xl p-6 border border-blue-200/20 dark:border-blue-700/20"
        >
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-2">
            Demo Talimatları
          </h3>
          <ul className="text-blue-800 dark:text-blue-200 space-y-1">
            <li>• Kamerayı başlatmak için "Başlat" butonuna tıklayın</li>
            <li>• Sistem otomatik olarak yüzünüzü tespit edecek ve analiz edecek</li>
            <li>• Veriler 2 saniyede bir güncellenir</li>
            <li>• Farklı yüz ifadeleri deneyerek duygu tanımayı test edin</li>
            <li>• Bakış yönünüzü değiştirerek bakış takibi özelliğini görün</li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
};

export default DemoPage;
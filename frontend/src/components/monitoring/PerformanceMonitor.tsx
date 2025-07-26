import { motion } from 'framer-motion';
import { Activity, Cpu, Database, Network, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
interface PerformanceMetrics {
  apiLatency: number;
  aiProcessingTime: number;
  frameRate: number;
  memoryUsage: number;
  networkStatus: 'good' | 'moderate' | 'poor';
  errorRate: number;
  uptime: number;
}
interface PerformanceMonitorProps {
  className?: string;
}
export function PerformanceMonitor({ className = '' }: PerformanceMonitorProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    apiLatency: 0,
    aiProcessingTime: 0,
    frameRate: 0,
    memoryUsage: 0,
    networkStatus: 'good',
    errorRate: 0,
    uptime: 0
  });
  const [isMonitoring, setIsMonitoring] = useState(false);
  useEffect(() => {
    let interval: NodeJS.Timeout;
    const updateMetrics = async () => {
      try {
        const startTime = performance.now();
        const response = await fetch('/api/health', { 
          method: 'GET',
          cache: 'no-cache'
        });
        const endTime = performance.now();
        const latency = endTime - startTime;
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const memInfo = (performance as any).memory;
        setMetrics(prev => ({
          apiLatency: latency,
          aiProcessingTime: prev.aiProcessingTime, 
          frameRate: prev.frameRate, 
          memoryUsage: memInfo ? (memInfo.usedJSHeapSize / memInfo.totalJSHeapSize) * 100 : 0,
          networkStatus: latency < 100 ? 'good' : latency < 300 ? 'moderate' : 'poor',
          errorRate: prev.errorRate,
          uptime: performance.now() / 1000
        }));
      } catch (error) {
        console.error('Performance monitoring error:', error);
        setMetrics(prev => ({
          ...prev,
          errorRate: prev.errorRate + 1,
          networkStatus: 'poor'
        }));
      }
    };
    if (isMonitoring) {
      updateMetrics();
      interval = setInterval(updateMetrics, 5000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isMonitoring]);
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-400';
      case 'moderate': return 'text-yellow-400';
      case 'poor': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };
  const getLatencyStatus = (latency: number) => {
    if (latency < 100) return 'excellent';
    if (latency < 300) return 'good';
    if (latency < 500) return 'moderate';
    return 'poor';
  };
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-700/30 p-6 ${className}`}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-400" />
          Sistem Performansı
        </h3>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsMonitoring(!isMonitoring)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            isMonitoring 
              ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
              : 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
          }`}
        >
          {isMonitoring ? 'İzleniyor' : 'Başlat'}
        </motion.button>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gray-800/30 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-300">API Gecikmesi</span>
          </div>
          <div className="flex items-end gap-1">
            <span className={`text-lg font-bold ${getStatusColor(getLatencyStatus(metrics.apiLatency))}`}>
              {metrics.apiLatency.toFixed(0)}
            </span>
            <span className="text-xs text-gray-500">ms</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1 mt-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min((metrics.apiLatency / 500) * 100, 100)}%` }}
              className={`h-1 rounded-full ${
                metrics.apiLatency < 100 ? 'bg-green-500' :
                metrics.apiLatency < 300 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
            />
          </div>
        </motion.div>
        {}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-gray-800/30 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-300">AI İşlem</span>
          </div>
          <div className="flex items-end gap-1">
            <span className="text-lg font-bold text-blue-400">
              {metrics.aiProcessingTime.toFixed(1)}
            </span>
            <span className="text-xs text-gray-500">s</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1 mt-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min((metrics.aiProcessingTime / 3) * 100, 100)}%` }}
              className="h-1 rounded-full bg-blue-500"
            />
          </div>
        </motion.div>
        {}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-800/30 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-300">FPS</span>
          </div>
          <div className="flex items-end gap-1">
            <span className="text-lg font-bold text-green-400">
              {metrics.frameRate.toFixed(0)}
            </span>
            <span className="text-xs text-gray-500">fps</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1 mt-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(metrics.frameRate / 30) * 100}%` }}
              className="h-1 rounded-full bg-green-500"
            />
          </div>
        </motion.div>
        {}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-800/30 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-300">Bellek</span>
          </div>
          <div className="flex items-end gap-1">
            <span className="text-lg font-bold text-purple-400">
              {metrics.memoryUsage.toFixed(1)}
            </span>
            <span className="text-xs text-gray-500">%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1 mt-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${metrics.memoryUsage}%` }}
              className={`h-1 rounded-full ${
                metrics.memoryUsage < 70 ? 'bg-green-500' :
                metrics.memoryUsage < 85 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
            />
          </div>
        </motion.div>
        {}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-gray-800/30 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Network className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-gray-300">Ağ Durumu</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-lg font-bold capitalize ${getStatusColor(metrics.networkStatus)}`}>
              {metrics.networkStatus === 'good' ? 'İyi' :
               metrics.networkStatus === 'moderate' ? 'Orta' : 'Zayıf'}
            </span>
            <div className={`w-2 h-2 rounded-full ${
              metrics.networkStatus === 'good' ? 'bg-green-500' :
              metrics.networkStatus === 'moderate' ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
          </div>
        </motion.div>
        {}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-gray-800/30 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span className="text-sm text-gray-300">Çalışma Süresi</span>
          </div>
          <div className="flex items-end gap-1">
            <span className="text-lg font-bold text-emerald-400">
              {Math.floor(metrics.uptime / 60)}
            </span>
            <span className="text-xs text-gray-500">dk</span>
          </div>
        </motion.div>
      </div>
      {}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="mt-6 p-4 bg-gray-800/20 rounded-lg border border-gray-700/30"
      >
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-300">Sistem Durumu:</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              metrics.networkStatus === 'good' && metrics.apiLatency < 200 ? 'bg-green-500' :
              metrics.networkStatus === 'moderate' || metrics.apiLatency < 400 ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
            <span className={`font-medium ${
              metrics.networkStatus === 'good' && metrics.apiLatency < 200 ? 'text-green-400' :
              metrics.networkStatus === 'moderate' || metrics.apiLatency < 400 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {metrics.networkStatus === 'good' && metrics.apiLatency < 200 ? 'Mükemmel' :
               metrics.networkStatus === 'moderate' || metrics.apiLatency < 400 ? 'İyi' : 'Dikkat Gerekli'}
            </span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
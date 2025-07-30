import clsx from 'clsx';
import ReactECharts from 'echarts-for-react';
import { motion } from 'framer-motion';
import React, { useMemo } from 'react';
import { useUserPreferences } from '../../stores/user-store';
import type { EmotionMetrics, EmotionType } from '../../types';

interface SentimentIndicatorProps {
  emotions: EmotionMetrics;
  className?: string;
}

/**
 * SentimentIndicator - Emotion analysis visualization with pie chart and trend
 */
export const SentimentIndicator: React.FC<SentimentIndicatorProps> = ({
  emotions,
  className
}) => {
  const preferences = useUserPreferences();
  const isDarkMode = preferences?.theme === 'dark';

  const emotionColors = {
    happy: '#22c55e',
    neutral: '#64748b',
    focused: '#3b82f6',
    confused: '#f59e0b',
    frustrated: '#ef4444',
    surprised: '#a855f7',
    sad: '#6366f1'
  };

  const emotionLabels = {
    happy: 'Mutlu',
    neutral: 'Nötr',
    focused: 'Odaklanmış',
    confused: 'Kafası Karışık',
    frustrated: 'Sinirli',
    surprised: 'Şaşırmış',
    sad: 'Üzgün'
  };

  const pieOption = useMemo(() => {
    const data = Object.entries(emotions.emotions).map(([emotion, percentage]) => ({
      name: emotionLabels[emotion as EmotionType],
      value: percentage,
      itemStyle: {
        color: emotionColors[emotion as EmotionType]
      }
    }));

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c}% ({d}%)',
        backgroundColor: isDarkMode ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)',
        borderColor: isDarkMode ? '#334155' : '#e2e8f0',
        textStyle: {
          color: isDarkMode ? '#f1f5f9' : '#1e293b'
        }
      },
      legend: {
        type: 'scroll',
        orient: 'horizontal',
        bottom: 0,
        textStyle: {
          color: isDarkMode ? '#94a3b8' : '#64748b',
          fontSize: 12
        }
      },
      series: [
        {
          name: 'Duygu Dağılımı',
          type: 'pie',
          radius: ['30%', '70%'],
          center: ['50%', '45%'],
          avoidLabelOverlap: false,
          label: {
            show: false,
            position: 'center'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 16,
              fontWeight: 'bold',
              color: isDarkMode ? '#f1f5f9' : '#1e293b'
            },
            scale: true,
            scaleSize: 5
          },
          labelLine: {
            show: false
          },
          data: data
        }
      ],
      animationDuration: preferences?.reducedMotion ? 0 : 1000,
      animationEasing: 'cubicOut'
    };
  }, [emotions, isDarkMode, preferences?.reducedMotion]);

  const getTrendIcon = () => {
    const iconClasses = "w-5 h-5";
    
    switch (emotions.trend) {
      case 'improving':
        return (
          <svg className={clsx(iconClasses, "text-emerald-400")} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 17l9.2-9.2M17 17V7H7" />
          </svg>
        );
      case 'declining':
        return (
          <svg className={clsx(iconClasses, "text-red-400")} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 7l-9.2 9.2M7 7v10h10" />
          </svg>
        );
      case 'stable':
        return (
          <svg className={clsx(iconClasses, "text-gray-400")} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 12H6" />
          </svg>
        );
      default:
        return null;
    }
  };

  const getTrendText = () => {
    switch (emotions.trend) {
      case 'improving':
        return 'İyileşiyor';
      case 'declining':
        return 'Kötüleşiyor';
      case 'stable':
        return 'Kararlı';
      default:
        return 'Bilinmiyor';
    }
  };

  const dominantEmotionColor = emotionColors[emotions.dominant];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={clsx(
        'glass-effect-dark',
        'rounded-lg',
        'p-4',
        'border border-accent-purple/20',
        'backdrop-blur-sm',
        'reduced-motion',
        className
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">
            Duygu Analizi
          </h3>
          <p className="text-sm text-gray-400">
            Sınıfın genel ruh hali
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Güvenilirlik</div>
          <div className="text-lg font-bold text-white">
            %{Math.round(emotions.confidence)}
          </div>
        </div>
      </div>

      <div className="h-48 mb-4">
        <ReactECharts
          option={pieOption}
          style={{ height: '100%', width: '100%' }}
          opts={{
            renderer: 'canvas'
          }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-gray-400 mb-1">Baskın Duygu</div>
          <div className="flex items-center gap-2">
            <div 
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: dominantEmotionColor }}
            />
            <span className="text-lg font-bold text-white">
              {emotionLabels[emotions.dominant]}
            </span>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-sm text-gray-400 mb-1">Eğilim</div>
          <div className="flex items-center gap-2">
            {getTrendIcon()}
            <span className="text-lg font-bold text-white">
              {getTrendText()}
            </span>
          </div>
        </div>
      </div>

      {/* Emotion bars */}
      <div className="mt-4 space-y-2">
        {Object.entries(emotions.emotions)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 3)
          .map(([emotion, percentage]) => (
            <div key={emotion} className="flex items-center gap-3">
              <div className="w-16 text-sm text-gray-400">
                {emotionLabels[emotion as EmotionType]}
              </div>
              <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: emotionColors[emotion as EmotionType] }}
                />
              </div>
              <div className="w-10 text-sm text-gray-300 text-right">
                {Math.round(percentage)}%
              </div>
            </div>
          ))}
      </div>
    </motion.div>
  );
};

SentimentIndicator.displayName = 'SentimentIndicator';

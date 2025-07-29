import clsx from 'clsx';
import ReactECharts from 'echarts-for-react';
import { motion } from 'framer-motion';
import React, { useMemo } from 'react';
import { useUserPreferences } from '../../stores/user-store';
import type { ParticipationMetrics } from '../../types';

interface ParticipationMeterProps {
  metrics: ParticipationMetrics;
  className?: string;
}

/**
 * ParticipationMeter - Live participation metrics display with gauge charts
 */
export const ParticipationMeter: React.FC<ParticipationMeterProps> = ({
  metrics,
  className
}) => {
  const preferences = useUserPreferences();
  const isDarkMode = preferences?.theme === 'dark';

  const gaugeOption = useMemo(() => {
    return {
      backgroundColor: 'transparent',
      tooltip: {
        formatter: '{a} <br/>{b} : {c}%'
      },
      series: [
        {
          name: 'Kamera Açık',
          type: 'gauge',
          center: ['25%', '50%'],
          radius: '80%',
          min: 0,
          max: 100,
          progress: {
            show: true,
            width: 8
          },
          axisLine: {
            lineStyle: {
              width: 8,
              color: [
                [0.3, '#ef4444'],
                [0.7, '#f59e0b'],
                [1, '#22c55e']
              ]
            }
          },
          axisTick: {
            show: false
          },
          splitLine: {
            show: false
          },
          axisLabel: {
            show: false
          },
          anchor: {
            show: true,
            showAbove: true,
            size: 18,
            itemStyle: {
              borderWidth: 2,
              borderColor: '#3b82f6',
              color: '#ffffff'
            }
          },
          title: {
            show: true,
            offsetCenter: [0, '80%'],
            textStyle: {
              color: isDarkMode ? '#f1f5f9' : '#1e293b',
              fontSize: 12
            }
          },
          detail: {
            valueAnimation: true,
            formatter: '{value}%',
            textStyle: {
              color: isDarkMode ? '#f1f5f9' : '#1e293b',
              fontSize: 16,
              fontWeight: 'bold'
            },
            offsetCenter: [0, '40%']
          },
          data: [
            {
              value: metrics.cameraOn,
              name: 'Kamera'
            }
          ]
        },
        {
          name: 'Mikrofon Aktif',
          type: 'gauge',
          center: ['75%', '50%'],
          radius: '80%',
          min: 0,
          max: 100,
          progress: {
            show: true,
            width: 8
          },
          axisLine: {
            lineStyle: {
              width: 8,
              color: [
                [0.3, '#ef4444'],
                [0.7, '#f59e0b'],
                [1, '#22c55e']
              ]
            }
          },
          axisTick: {
            show: false
          },
          splitLine: {
            show: false
          },
          axisLabel: {
            show: false
          },
          anchor: {
            show: true,
            showAbove: true,
            size: 18,
            itemStyle: {
              borderWidth: 2,
              borderColor: '#06b6d4',
              color: '#ffffff'
            }
          },
          title: {
            show: true,
            offsetCenter: [0, '80%'],
            textStyle: {
              color: isDarkMode ? '#f1f5f9' : '#1e293b',
              fontSize: 12
            }
          },
          detail: {
            valueAnimation: true,
            formatter: '{value}%',
            textStyle: {
              color: isDarkMode ? '#f1f5f9' : '#1e293b',
              fontSize: 16,
              fontWeight: 'bold'
            },
            offsetCenter: [0, '40%']
          },
          data: [
            {
              value: metrics.micActive,
              name: 'Mikrofon'
            }
          ]
        }
      ],
      animationDuration: preferences?.reducedMotion ? 0 : 1000,
      animationEasing: 'cubicOut'
    };
  }, [metrics, isDarkMode, preferences?.reducedMotion]);

  const getEngagementLevel = () => {
    const avgEngagement = (metrics.cameraOn + metrics.micActive) / 2;
    if (avgEngagement >= 80) return { label: 'Çok Yüksek', color: 'text-emerald-400' };
    if (avgEngagement >= 60) return { label: 'Yüksek', color: 'text-green-400' };
    if (avgEngagement >= 40) return { label: 'Orta', color: 'text-yellow-400' };
    if (avgEngagement >= 20) return { label: 'Düşük', color: 'text-orange-400' };
    return { label: 'Çok Düşük', color: 'text-red-400' };
  };

  const engagementLevel = getEngagementLevel();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={clsx(
        'glass-effect-dark',
        'rounded-lg',
        'p-4',
        'border border-accent-cyan/20',
        'backdrop-blur-sm',
        'reduced-motion',
        className
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">
            Katılım Metrikleri
          </h3>
          <p className="text-sm text-gray-400">
            Anlık katılım durumu
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Genel Seviye</div>
          <div className={clsx('text-lg font-bold', engagementLevel.color)}>
            {engagementLevel.label}
          </div>
        </div>
      </div>

      <div className="h-48 mb-4">
        <ReactECharts
          option={gaugeOption}
          style={{ height: '100%', width: '100%' }}
          opts={{
            renderer: 'canvas'
          }}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-accent-cyan">
            {metrics.handRaises}
          </div>
          <div className="text-sm text-gray-400">El Kaldırma</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-accent-purple">
            {metrics.chatActivity}
          </div>
          <div className="text-sm text-gray-400">Sohbet/dk</div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-600/30">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Aktif Öğrenci:</span>
          <span className="text-white font-medium">
            {metrics.totalStudents} / {metrics.totalStudents}
          </span>
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span className="text-gray-400">Konuşma Süresi:</span>
          <span className="text-white font-medium">
            {Math.floor(metrics.speakingTime / 60)}:{(metrics.speakingTime % 60).toString().padStart(2, '0')}
          </span>
        </div>
      </div>
    </motion.div>
  );
};

ParticipationMeter.displayName = 'ParticipationMeter';

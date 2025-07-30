import clsx from 'clsx';
import ReactECharts from 'echarts-for-react';
import { motion } from 'framer-motion';
import React, { useMemo } from 'react';
import { useUserPreferences } from '../../stores/user-store';
import type { AttentionDataPoint } from '../../types';

interface AttentionTimelineProps {
  data: AttentionDataPoint[];
  timeRange?: number; // minutes
  isLive?: boolean;
  onTimeSelect?: (timestamp: number) => void;
  className?: string;
}

/**
 * AttentionTimeline - Real-time attention tracking visualization using ECharts
 */
export const AttentionTimeline: React.FC<AttentionTimelineProps> = ({
  data,
  timeRange = 30,
  isLive = true,
  onTimeSelect,
  className
}) => {
  const preferences = useUserPreferences();
  const isDarkMode = preferences?.theme === 'dark';

  const chartOption = useMemo(() => {
    const theme = isDarkMode ? 'dark' : 'light';
    
    // Filter data for the specified time range
    const now = Date.now();
    const timeRangeMs = timeRange * 60 * 1000;
    const filteredData = data.filter(point => 
      now - point.timestamp <= timeRangeMs
    );

    // Transform data for ECharts
    const chartData = filteredData.map(point => [
      new Date(point.timestamp),
      point.value
    ]);

    // Create attention level zones
    const zones = [
      { yAxis: 0, color: 'rgba(239, 68, 68, 0.2)' }, // Very Low
      { yAxis: 20, color: 'rgba(245, 158, 11, 0.2)' }, // Low
      { yAxis: 40, color: 'rgba(100, 116, 139, 0.2)' }, // Medium
      { yAxis: 60, color: 'rgba(34, 197, 94, 0.2)' }, // High
      { yAxis: 80, color: 'rgba(59, 130, 246, 0.2)' }, // Very High
      { yAxis: 100 }
    ];

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDarkMode ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)',
        borderColor: isDarkMode ? '#334155' : '#e2e8f0',
        textStyle: {
          color: isDarkMode ? '#f1f5f9' : '#1e293b'
        },
        formatter: (params: any) => {
          const point = params[0];
          const time = new Date(point.data[0]).toLocaleTimeString();
          const level = point.data[1];
          return `
            <div class="font-medium">${time}</div>
            <div>Dikkat: ${level}%</div>
          `;
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'time',
        boundaryGap: false,
        axisLine: {
          lineStyle: {
            color: isDarkMode ? '#475569' : '#cbd5e1'
          }
        },
        axisTick: {
          lineStyle: {
            color: isDarkMode ? '#475569' : '#cbd5e1'
          }
        },
        axisLabel: {
          color: isDarkMode ? '#94a3b8' : '#64748b',
          formatter: (value: number) => {
            return new Date(value).toLocaleTimeString('tr-TR', {
              hour: '2-digit',
              minute: '2-digit'
            });
          }
        }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLine: {
          lineStyle: {
            color: isDarkMode ? '#475569' : '#cbd5e1'
          }
        },
        axisTick: {
          lineStyle: {
            color: isDarkMode ? '#475569' : '#cbd5e1'
          }
        },
        axisLabel: {
          color: isDarkMode ? '#94a3b8' : '#64748b',
          formatter: '{value}%'
        },
        splitLine: {
          lineStyle: {
            color: isDarkMode ? '#334155' : '#e2e8f0',
            opacity: 0.5
          }
        }
      },
      series: [
        {
          name: 'Dikkat Seviyesi',
          type: 'line',
          data: chartData,
          smooth: true,
          lineStyle: {
            width: 3,
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 1, y2: 0,
              colorStops: [
                { offset: 0, color: '#3b82f6' },
                { offset: 1, color: '#06b6d4' }
              ]
            }
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
              ]
            }
          },
          emphasis: {
            scale: true,
            scaleSize: 10
          },
          symbol: 'circle',
          symbolSize: 4,
          itemStyle: {
            color: '#3b82f6',
            borderColor: '#ffffff',
            borderWidth: 2
          }
        }
      ],
      animationDuration: preferences?.reducedMotion ? 0 : 1000,
      animationEasing: 'cubicOut'
    };
  }, [data, timeRange, isDarkMode, preferences?.reducedMotion]);

  const handleChartClick = (params: any) => {
    if (onTimeSelect && params.data) {
      onTimeSelect(params.data[0]);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={clsx(
        'glass-effect-dark',
        'rounded-lg',
        'p-4',
        'border border-primary-500/20',
        'backdrop-blur-sm',
        'reduced-motion',
        className
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">
            Dikkat Zaman Çizelgesi
          </h3>
          <p className="text-sm text-gray-400">
            Son {timeRange} dakika
            {isLive && (
              <span className="ml-2 inline-flex items-center">
                <span className="animate-pulse w-2 h-2 bg-emerald-500 rounded-full mr-1" />
                Canlı
              </span>
            )}
          </p>
        </div>
      </div>
      
      <div className="h-64">
        <ReactECharts
          option={chartOption}
          style={{ height: '100%', width: '100%' }}
          onEvents={{
            click: handleChartClick
          }}
          opts={{
            renderer: 'canvas',
            useDirtyRect: false
          }}
        />
      </div>
    </motion.div>
  );
};

AttentionTimeline.displayName = 'AttentionTimeline';

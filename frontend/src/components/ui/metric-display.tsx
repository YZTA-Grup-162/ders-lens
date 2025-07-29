import clsx from 'clsx';
import { motion } from 'framer-motion';
import React from 'react';
import type { EmotionType, MetricDisplayProps } from '../../types';

/**
 * MetricDisplay - Real-time data visualization component with trend indicators
 */
export const MetricDisplay: React.FC<MetricDisplayProps> = ({
  title,
  value,
  unit = '',
  trend,
  format = 'number',
  color = 'primary',
  className,
  ...props
}) => {
  const formatValue = (val: number | string): string => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'percentage':
        return `${Math.round(val)}%`;
      case 'time':
        const hours = Math.floor(val / 3600);
        const minutes = Math.floor((val % 3600) / 60);
        const seconds = Math.floor(val % 60);
        if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
      default:
        return val.toLocaleString();
    }
  };

  const getColorClasses = (colorKey: EmotionType | 'primary' | 'accent') => {
    const colorMap = {
      primary: 'text-primary-300 border-primary-500/30',
      accent: 'text-accent-cyan border-accent-cyan/30',
      happy: 'text-emotion-happy border-emotion-happy/30',
      neutral: 'text-emotion-neutral border-emotion-neutral/30',
      focused: 'text-emotion-focused border-emotion-focused/30',
      confused: 'text-emotion-confused border-emotion-confused/30',
      frustrated: 'text-emotion-frustrated border-emotion-frustrated/30',
      surprised: 'text-emotion-surprised border-emotion-surprised/30',
      sad: 'text-emotion-sad border-emotion-sad/30',
    };
    return colorMap[colorKey] || colorMap.primary;
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    
    const iconClasses = "w-4 h-4 ml-1";
    
    switch (trend) {
      case 'up':
        return (
          <svg className={clsx(iconClasses, "text-emerald-400")} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 17l9.2-9.2M17 17V7H7" />
          </svg>
        );
      case 'down':
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={clsx(
        'glass-effect-dark',
        'rounded-lg',
        'p-4',
        'border',
        getColorClasses(color),
        'backdrop-blur-sm',
        'reduced-motion',
        className
      )}
      {...props}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-300 mb-1">
            {title}
          </h3>
          <div className="flex items-center">
            <motion.span
              key={value}
              initial={{ scale: 1.1 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.2 }}
              className={clsx(
                'text-2xl font-bold',
                getColorClasses(color).split(' ')[0] // Extract text color class
              )}
            >
              {formatValue(value)}
              {unit && <span className="text-lg ml-1 text-gray-400">{unit}</span>}
            </motion.span>
            {getTrendIcon()}
          </div>
        </div>
        
        {/* Optional pulse animation for real-time data */}
        <div className="relative">
          <div className={clsx(
            'w-2 h-2 rounded-full',
            'animate-pulse',
            color === 'primary' && 'bg-primary-500',
            color === 'accent' && 'bg-accent-cyan',
            color !== 'primary' && color !== 'accent' && `bg-emotion-${color}`,
          )} />
          <div className={clsx(
            'absolute inset-0 w-2 h-2 rounded-full',
            'animate-ping',
            color === 'primary' && 'bg-primary-500',
            color === 'accent' && 'bg-accent-cyan',
            color !== 'primary' && color !== 'accent' && `bg-emotion-${color}`,
            'opacity-75'
          )} />
        </div>
      </div>
    </motion.div>
  );
};

MetricDisplay.displayName = 'MetricDisplay';

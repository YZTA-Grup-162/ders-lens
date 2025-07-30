import clsx from 'clsx';
import { motion } from 'framer-motion';
import React from 'react';
import type { GlassCardProps } from '../../types';


export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className,
  neonAccent = false,
  padding = 'md',
  blur = 'md',
  ...props
}) => {
  const paddingClasses = {
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  };

  const blurClasses = {
    sm: 'backdrop-blur-sm',
    md: 'backdrop-blur-md',
    lg: 'backdrop-blur-lg',
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        duration: 0.3,
        ease: 'easeOut',
      }}
      className={clsx(
        'glass-effect-dark',
        blurClasses[blur],
        'rounded-lg',
        paddingClasses[padding],
        
        neonAccent && 'border-primary-500/30 shadow-glow-sm',
        
        'reduced-motion',
        
        className
      )}
      {...props}
    >
      {children}
    </motion.div>
  );
};

GlassCard.displayName = 'GlassCard';

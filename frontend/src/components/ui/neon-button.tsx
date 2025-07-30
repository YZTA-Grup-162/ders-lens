import clsx from 'clsx';
import { motion } from 'framer-motion';
import React from 'react';
import type { NeonButtonProps } from '../../types';


export const NeonButton: React.FC<NeonButtonProps> = ({
  children,
  className,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  ...props
}) => {
  const baseClasses = [
    'relative',
    'font-medium',
    'rounded-lg',
    'transition-all',
    'duration-300',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-offset-2',
    'focus:ring-offset-transparent',
    'disabled:opacity-50',
    'disabled:cursor-not-allowed',
    'reduced-motion'
  ];

  const variantClasses = {
    primary: [
      'bg-primary-600/20',
      'border',
      'border-primary-500/30',
      'text-primary-100',
      'hover:bg-primary-500/30',
      'hover:border-primary-400/50',
      'hover:shadow-glow-md',
      'focus:ring-primary-500',
      'backdrop-blur-sm'
    ],
    secondary: [
      'bg-dark-700/20',
      'border',
      'border-dark-600/30',
      'text-dark-100',
      'hover:bg-dark-600/30',
      'hover:border-dark-500/50',
      'focus:ring-dark-500',
      'backdrop-blur-sm'
    ],
    accent: [
      'bg-accent-cyan/20',
      'border',
      'border-accent-cyan/30',
      'text-white',
      'hover:bg-accent-cyan/30',
      'hover:border-accent-cyan/50',
      'hover:shadow-neon-cyan',
      'focus:ring-accent-cyan',
      'backdrop-blur-sm'
    ]
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  const buttonClasses = clsx(
    baseClasses,
    variantClasses[variant],
    sizeClasses[size],
    className
  );

  const handleClick = () => {
    if (!disabled && !loading && onClick) {
      onClick();
    }
  };

  return (
    <motion.button
      type="button"
      className={buttonClasses}
      onClick={handleClick}
      disabled={disabled || loading}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{
        type: "spring",
        stiffness: 400,
        damping: 17
      }}
      {...props}
    >
      <span className="relative flex items-center justify-center gap-2">
        {loading && (
          <svg
            className="animate-spin -ml-1 mr-2 h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </span>
    </motion.button>
  );
};

NeonButton.displayName = 'NeonButton';

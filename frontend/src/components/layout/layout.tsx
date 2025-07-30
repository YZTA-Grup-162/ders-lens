import clsx from 'clsx';
import { motion } from 'framer-motion';
import React from 'react';
import { useTranslation } from '../../lib/i18n';
import { useUserStore } from '../../stores/user-store';

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
}


export const Layout: React.FC<LayoutProps> = ({ children, className }) => {
  const { t } = useTranslation();
  const { user } = useUserStore();

  return (
    <div className={clsx(
      'min-h-screen',
      'bg-gradient-to-br from-dark-950 via-dark-900 to-dark-950',
      'text-white',
      'overflow-x-hidden',
      className
    )}>
      {/* Animated Background */}
      <div className="animated-background" />
      
      {/* Background animation */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-30">
          <motion.div
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.2, 0.3],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.div
            className="absolute top-3/4 right-1/4 w-64 h-64 bg-accent-cyan/10 rounded-full blur-3xl"
            animate={{
              scale: [1.2, 1, 1.2],
              opacity: [0.2, 0.4, 0.2],
            }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.div
            className="absolute top-1/2 right-1/3 w-32 h-32 bg-accent-purple/10 rounded-full blur-2xl"
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.4, 0.2, 0.4],
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

Layout.displayName = 'Layout';

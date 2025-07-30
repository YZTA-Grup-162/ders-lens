import { motion } from 'framer-motion';

const AnimatedBackground = () => {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      {/* Gradient Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900" />
      
      {/* Animated Orbs */}
      <motion.div
        className="absolute top-1/4 left-1/4 w-72 h-72 bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-full filter blur-3xl"
        animate={{
          x: [0, 100, 0],
          y: [0, -50, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      <motion.div
        className="absolute top-3/4 right-1/4 w-96 h-96 bg-gradient-to-r from-purple-400/20 to-pink-400/20 rounded-full filter blur-3xl"
        animate={{
          x: [0, -80, 0],
          y: [0, 60, 0],
          scale: [1, 0.9, 1],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      <motion.div
        className="absolute top-1/2 left-1/2 w-64 h-64 bg-gradient-to-r from-indigo-400/20 to-blue-400/20 rounded-full filter blur-3xl"
        animate={{
          x: [0, -120, 0],
          y: [0, 80, 0],
          rotate: [0, 180, 360],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      {/* Neural Network Pattern */}
      <svg className="absolute inset-0 w-full h-full opacity-10 dark:opacity-5" viewBox="0 0 1000 1000">
        <defs>
          <pattern id="neuralPattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
            <circle cx="50" cy="50" r="2" fill="currentColor" className="text-indigo-600" />
            <line x1="50" y1="50" x2="100" y2="25" stroke="currentColor" strokeWidth="0.5" className="text-indigo-400" />
            <line x1="50" y1="50" x2="100" y2="75" stroke="currentColor" strokeWidth="0.5" className="text-indigo-400" />
            <line x1="50" y1="50" x2="0" y2="25" stroke="currentColor" strokeWidth="0.5" className="text-indigo-400" />
            <line x1="50" y1="50" x2="0" y2="75" stroke="currentColor" strokeWidth="0.5" className="text-indigo-400" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#neuralPattern)" />
      </svg>
    </div>
  );
};

export default AnimatedBackground;

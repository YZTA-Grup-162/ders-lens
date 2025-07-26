import { motion } from 'framer-motion';
import { Moon, Sun } from 'lucide-react';
import React from 'react';
import { useTheme } from '../contexts/ThemeContext';
const ThemeToggle: React.FC = () => {
  const { isDarkMode, toggleTheme } = useTheme();
  return (
    <motion.button
      onClick={toggleTheme}
      className={`relative w-12 h-6 rounded-full p-1 transition-colors duration-300 ${
        isDarkMode ? 'bg-blue-600' : 'bg-slate-300'
      }`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <motion.div
        className={`w-4 h-4 rounded-full bg-white flex items-center justify-center shadow-sm ${
          isDarkMode ? 'translate-x-6' : 'translate-x-0'
        } transition-transform duration-300`}
        animate={{ x: isDarkMode ? 24 : 0 }}
      >
        {isDarkMode ? (
          <Moon className="w-3 h-3 text-blue-600" />
        ) : (
          <Sun className="w-3 h-3 text-yellow-500" />
        )}
      </motion.div>
    </motion.button>
  );
};
export default ThemeToggle;
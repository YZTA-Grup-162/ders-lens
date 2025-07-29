import clsx from 'clsx';
import { motion } from 'framer-motion';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useUserStore } from '../../stores/user-store';
import { GlassCard } from '../ui/glass-card';
import { NeonButton } from '../ui/neon-button';

interface HeaderProps {
  className?: string;
}

/**
 * Header - Main application header with navigation and language toggle
 */
export const Header: React.FC<HeaderProps> = ({ className }) => {
  const { t, i18n } = useTranslation();
  const { user, updatePreferences } = useUserStore();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const toggleLanguage = () => {
    const newLang = i18n.language === 'tr' ? 'en' : 'tr';
    i18n.changeLanguage(newLang);
    
    if (user) {
      updatePreferences({ language: newLang as 'tr' | 'en' });
    }
  };

  const toggleTheme = () => {
    const newTheme = user?.preferences.theme === 'dark' ? 'light' : 'dark';
    updatePreferences({ theme: newTheme });
    
    // Apply theme to document
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={clsx(
        'sticky top-0 z-50',
        'backdrop-blur-lg',
        'border-b border-white/10',
        className
      )}
    >
      <GlassCard className="rounded-none border-0 border-b border-white/10" padding="sm">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <motion.div
              className="flex items-center space-x-3"
              whileHover={{ scale: 1.02 }}
            >
              <img 
                src="/derslens-logo.png" 
                alt="DersLens Logo" 
                className="w-8 h-8 object-contain"
              />
              <div>
                <h1 className="text-xl font-bold text-white">
                  DersLens
                </h1>
                <p className="text-xs text-gray-400 hidden sm:block">
                  {t('common.tagline', 'Smart Classroom Analytics')}
                </p>
              </div>
            </motion.div>

            {/* Navigation - Desktop */}
            <nav className="hidden md:flex items-center space-x-6">
              <a
                href="#dashboard"
                className="text-gray-300 hover:text-white transition-colors"
              >
                {t('nav.dashboard')}
              </a>
              <a
                href="#live-class"
                className="text-gray-300 hover:text-white transition-colors"
              >
                {t('nav.liveClass')}
              </a>
              <a
                href="#analytics"
                className="text-gray-300 hover:text-white transition-colors"
              >
                {t('nav.analytics')}
              </a>
              <a
                href="#settings"
                className="text-gray-300 hover:text-white transition-colors"
              >
                {t('nav.settings')}
              </a>
            </nav>

            {/* Actions */}
            <div className="flex items-center space-x-3">
              {/* Language Toggle */}
              <motion.button
                onClick={toggleLanguage}
                className="hidden sm:flex items-center px-2 py-1 rounded text-sm text-gray-300 hover:text-white transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                </svg>
                {i18n.language.toUpperCase()}
              </motion.button>

              {/* Theme Toggle */}
              <motion.button
                onClick={toggleTheme}
                className="hidden sm:flex items-center p-2 rounded text-gray-300 hover:text-white transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {user?.preferences.theme === 'dark' ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                )}
              </motion.button>

              {/* User Menu */}
              {user ? (
                <div className="flex items-center space-x-2">
                  <div className="hidden sm:block text-right">
                    <div className="text-sm font-medium text-white">
                      {user.name}
                    </div>
                    <div className="text-xs text-gray-400 capitalize">
                      {user.role}
                    </div>
                  </div>
                  <div className="w-8 h-8 bg-gradient-to-r from-accent-purple to-accent-emerald rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-medium">
                      {user.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                </div>
              ) : (
                <NeonButton size="sm" variant="primary">
                  {t('auth.login')}
                </NeonButton>
              )}

              {/* Mobile Menu Button */}
              <motion.button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded text-gray-300 hover:text-white transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {mobileMenuOpen ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              </motion.button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden py-4 border-t border-white/10"
            >
              <nav className="flex flex-col space-y-3">
                <a
                  href="#dashboard"
                  className="text-gray-300 hover:text-white transition-colors py-2"
                >
                  {t('nav.dashboard')}
                </a>
                <a
                  href="#live-class"
                  className="text-gray-300 hover:text-white transition-colors py-2"
                >
                  {t('nav.liveClass')}
                </a>
                <a
                  href="#analytics"
                  className="text-gray-300 hover:text-white transition-colors py-2"
                >
                  {t('nav.analytics')}
                </a>
                <a
                  href="#settings"
                  className="text-gray-300 hover:text-white transition-colors py-2"
                >
                  {t('nav.settings')}
                </a>
                
                <div className="flex items-center justify-between pt-3 border-t border-white/10">
                  <button
                    onClick={toggleLanguage}
                    className="flex items-center px-3 py-2 rounded text-sm text-gray-300 hover:text-white transition-colors"
                  >
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                    </svg>
                    {i18n.language === 'tr' ? 'Türkçe' : 'English'}
                  </button>
                  
                  <button
                    onClick={toggleTheme}
                    className="flex items-center p-2 rounded text-gray-300 hover:text-white transition-colors"
                  >
                    {user?.preferences.theme === 'dark' ? (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                      </svg>
                    )}
                  </button>
                </div>
              </nav>
            </motion.div>
          )}
        </div>
      </GlassCard>
    </motion.header>
  );
};

Header.displayName = 'Header';

import { motion } from 'framer-motion';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Header } from '../components/layout/header';
import { Layout } from '../components/layout/layout';
import { GlassCard } from '../components/ui/glass-card';
import { NeonButton } from '../components/ui/neon-button';
import { useDemoStore } from '../stores/demo-store';

/**
 * DemoLandingPage - Entry point for demo experience
 */
export const DemoLandingPage: React.FC = () => {
  const { t } = useTranslation();
  const { setScenario } = useDemoStore();

  const handleJoinDemo = () => {
    // Navigate to demo student experience
    window.location.href = '/demo/student';
  };

  const handleViewTeacher = () => {
    // Navigate to demo teacher dashboard
    window.location.href = '/demo/teacher';
  };

  return (
    <Layout>
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="inline-flex items-center px-4 py-2 rounded-full bg-primary-600/20 border border-primary-500/30 mb-8"
          >
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse" />
            <span className="text-primary-200 text-sm font-medium">
              {t('demo.title')}
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-5xl md:text-7xl font-bold text-white mb-6"
          >
            <span className="bg-gradient-to-r from-primary-400 via-accent-cyan to-accent-purple bg-clip-text text-transparent">
              DersLens
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto"
          >
            {t('demo.subtitle')}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <NeonButton
              size="lg"
              variant="primary"
              onClick={handleJoinDemo}
              className="text-lg px-8 py-4"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              {t('demo.joinButton')}
            </NeonButton>

            <NeonButton
              size="lg"
              variant="secondary"
              onClick={handleViewTeacher}
              className="text-lg px-8 py-4"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 00-2-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              {t('demo.viewTeacher')}
            </NeonButton>
          </motion.div>
        </motion.section>

        {/* Features Preview */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="grid md:grid-cols-3 gap-8 mb-16"
        >
          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-accent-cyan rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              {t('teacher.attention.title')}
            </h3>
            <p className="text-gray-300">
              Gerçek zamanlı dikkat seviyesi takibi ve analizi
            </p>
          </GlassCard>

          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-cyan to-accent-purple rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              {t('teacher.participation.title')}
            </h3>
            <p className="text-gray-300">
              Öğrenci katılımı ve etkileşim metrikleri
            </p>
          </GlassCard>

          <GlassCard neonAccent className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-purple to-accent-emerald rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.586a1 1 0 01.707.293l2.414 2.414a1 1 0 00.707.293H15M9 10V9a3 3 0 013-3v0a3 3 0 013 3v1M9 10v5a2 2 0 002 2h2a2 2 0 002-2v-5m-6 0h6" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              {t('teacher.suggestions.title')}
            </h3>
            <p className="text-gray-300">
              AI destekli öğretim önerileri ve iyileştirmeler
            </p>
          </GlassCard>
        </motion.section>

        {/* Demo Information */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <GlassCard className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  {t('demo.student.welcome')}
                </h3>
                <div className="space-y-3 text-gray-300">
                  <p>{t('demo.student.instructions')}</p>
                  <p>{t('demo.student.duration')}</p>
                  <p>{t('demo.student.features')}</p>
                  <p className="text-sm text-accent-cyan">
                    {t('demo.student.privacy')}
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  {t('demo.teacher.welcome')}
                </h3>
                <div className="space-y-3 text-gray-300">
                  <p>{t('demo.teacher.overview')}</p>
                  <p>{t('demo.teacher.features')}</p>
                  <p>{t('demo.teacher.controls')}</p>
                </div>
              </div>
            </div>

            <div className="mt-8 pt-8 border-t border-gray-600/30">
              <h4 className="text-lg font-bold text-white mb-4">
                {t('demo.scenarios.title', 'Demo Scenarios')}
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                {[
                  { key: 'standard', icon: '📚' },
                  { key: 'interactive', icon: '💬' },
                  { key: 'challenging', icon: '🧮' },
                  { key: 'groupWork', icon: '👥' },
                  { key: 'presentation', icon: '📊' }
                ].map(({ key, icon }) => (
                  <div
                    key={key}
                    className="text-center p-3 rounded-lg bg-dark-800/50 border border-gray-600/30"
                  >
                    <div className="text-2xl mb-1">{icon}</div>
                    <div className="text-sm text-gray-300">
                      {t(`demo.scenarios.${key}`)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </GlassCard>
        </motion.section>
      </main>
    </Layout>
  );
};

export default DemoLandingPage;

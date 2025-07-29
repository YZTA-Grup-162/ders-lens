import { AnimatePresence, motion } from 'framer-motion';
import React, { Suspense, useEffect } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import './App.css';
import { Header } from './components/layout/header';
import { Layout } from './components/layout/layout';
import { GlassCard } from './components/ui/glass-card';
import { NeonButton } from './components/ui/neon-button';
import './lib/i18n'; // Initialize i18n
import DemoLandingPage from './pages/demo-landing';
import DemoStudentDashboardEnhanced from './pages/demo-student-dashboard-enhanced';
import DemoTeacherDashboard from './pages/demo-teacher-dashboard';
import { useUserStore } from './stores/user-store';

// Loading component
const LoadingScreen: React.FC = () => (
  <Layout>
    <div className="min-h-screen flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="text-center"
      >
        <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-accent-cyan rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
          <span className="text-white font-bold text-xl">DL</span>
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">DersLens</h2>
        <p className="text-gray-400">Yükleniyor...</p>
      </motion.div>
    </div>
  </Layout>
);

// Home page component
const HomePage: React.FC = () => {
  const { setUser, user } = useUserStore();

  // Initialize demo user
  useEffect(() => {
    const demoUser = {
      id: 'demo-user',
      name: 'Demo Kullanıcısı',
      email: 'demo@derslens.com',
      role: 'demo' as const,
      preferences: {
        language: 'tr' as const,
        theme: 'dark' as const,
        reducedMotion: false,
        fontSize: 'medium' as const,
        notifications: {
          sound: true,
          visual: true,
          attention: true,
          participation: true,
          sentiment: true
        }
      }
    };
    setUser(demoUser);
  }, [setUser]);

  // Apply theme to document when user changes
  useEffect(() => {
    if (user?.preferences.theme) {
      if (user.preferences.theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    }
  }, [user?.preferences.theme]);

  return (
    <Layout>
      <Header />
      <main className="container mx-auto px-4 py-8">
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <motion.h1
            className="text-6xl md:text-8xl font-bold text-white mb-6"
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <span className="bg-gradient-to-r from-primary-400 via-accent-cyan to-accent-purple bg-clip-text text-transparent">
              DersLens
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="text-2xl text-gray-300 mb-8 max-w-3xl mx-auto"
          >
            Canlı Çevrimiçi Sınıf Analitik Platformu
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <NeonButton
              size="lg"
              variant="primary"
              onClick={() => window.location.href = '/demo'}
              className="text-lg px-8 py-4"
            >
              Demo'yu Deneyin
            </NeonButton>
            
            <NeonButton
              size="lg"
              variant="secondary"
              className="text-lg px-8 py-4"
            >
              Daha Fazla Bilgi
            </NeonButton>
          </motion.div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="grid md:grid-cols-3 gap-8"
        >
          <GlassCard neonAccent className="text-center p-6">
            <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-accent-cyan rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 00-2-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Gerçek Zamanlı Analitik
            </h3>
            <p className="text-gray-300">
              Öğrenci dikkat ve katılımını anlık olarak izleyin
            </p>
          </GlassCard>

          <GlassCard neonAccent className="text-center p-6">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-cyan to-accent-purple rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.586a1 1 0 01.707.293l2.414 2.414a1 1 0 00.707.293H15M9 10V9a3 3 0 013-3v0a3 3 0 013 3v1M9 10v5a2 2 0 002 2h2a2 2 0 002-2v-5m-6 0h6" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              AI Destekli Öneriler
            </h3>
            <p className="text-gray-300">
              Akıllı öğretim önerileri ve iyileştirme tavsiyeleri
            </p>
          </GlassCard>

          <GlassCard neonAccent className="text-center p-6">
            <div className="w-16 h-16 bg-gradient-to-r from-accent-purple to-accent-emerald rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
              Rol Tabanlı Görünümler
            </h3>
            <p className="text-gray-300">
              Öğretmen, öğrenci ve yönetici için özel paneller
            </p>
          </GlassCard>
        </motion.section>
      </main>
    </Layout>
  );
};

// Main App component
function App() {
  const { user } = useUserStore();

  // Apply theme to document
  useEffect(() => {
    const theme = user?.preferences.theme || 'dark';
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [user?.preferences.theme]);

  return (
    <Router>
      <Suspense fallback={<LoadingScreen />}>
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/demo" element={<DemoLandingPage />} />
            <Route path="/demo/student" element={<DemoStudentDashboardEnhanced />} />
            <Route path="/demo/teacher" element={<DemoTeacherDashboard />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </Router>
  );
}

export default App;

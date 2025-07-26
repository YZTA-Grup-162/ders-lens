import { AnimatePresence, motion } from 'framer-motion';
import React, { useState } from 'react';
import { Button, Card } from '../ui';
interface NavItem {
  id: string;
  label: string;
  icon: string;
  active?: boolean;
}
interface NavigationProps {
  items: NavItem[];
  onItemClick: (id: string) => void;
  activeItem: string;
}
export const Navigation: React.FC<NavigationProps> = ({ 
  items, 
  onItemClick, 
  activeItem 
}) => {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {}
          <div className="flex items-center">
            <motion.div
              className="flex-shrink-0 flex items-center"
              whileHover={{ scale: 1.05 }}
            >
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                DersLens
              </span>
              <span className="ml-2 text-lg">🎓</span>
            </motion.div>
          </div>
          {}
          <div className="flex space-x-8">
            {items.map((item) => (
              <motion.button
                key={item.id}
                onClick={() => onItemClick(item.id)}
                className={`inline-flex items-center px-1 pt-1 text-sm font-medium border-b-2 transition-colors duration-200 ${
                  activeItem === item.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                whileHover={{ y: -1 }}
                whileTap={{ y: 0 }}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </motion.button>
            ))}
          </div>
          {}
          <div className="flex items-center space-x-4">
            <Button variant="outline" size="sm">
              Settings
            </Button>
            <Button variant="primary" size="sm">
              Export Data
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};
interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}
export const Sidebar: React.FC<SidebarProps> = ({ 
  isOpen, 
  onToggle, 
  children 
}) => {
  return (
    <>
      {}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onToggle}
          />
        )}
      </AnimatePresence>
      {}
      <motion.div
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        initial={false}
        animate={{ x: isOpen ? 0 : -256 }}
      >
        <div className="flex flex-col h-full">
          {}
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Controls</h2>
            <button
              onClick={onToggle}
              className="lg:hidden p-1 rounded-md text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>
          {}
          <div className="flex-1 overflow-y-auto p-4">
            {children}
          </div>
        </div>
      </motion.div>
    </>
  );
};
interface HeaderProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  stats?: Array<{
    label: string;
    value: string | number;
    change?: number;
    icon?: string;
  }>;
}
export const Header: React.FC<HeaderProps> = ({ 
  title, 
  subtitle, 
  actions, 
  stats 
}) => {
  return (
    <div className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="md:flex md:items-center md:justify-between">
          <div className="flex-1 min-w-0">
            <motion.h1
              className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {title}
            </motion.h1>
            {subtitle && (
              <motion.p
                className="mt-1 text-sm text-gray-500"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: 0.1 }}
              >
                {subtitle}
              </motion.p>
            )}
          </div>
          {actions && (
            <motion.div
              className="mt-4 flex md:mt-0 md:ml-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
            >
              {actions}
            </motion.div>
          )}
        </div>
        {}
        {stats && stats.length > 0 && (
          <motion.div
            className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                className="bg-gray-50 overflow-hidden rounded-lg px-4 py-5 sm:p-6"
                whileHover={{ y: -2, boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
              >
                <div className="flex items-center">
                  {stat.icon && (
                    <div className="flex-shrink-0 text-2xl mr-3">
                      {stat.icon}
                    </div>
                  )}
                  <div className="w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        {stat.label}
                      </dt>
                      <dd className="flex items-baseline">
                        <div className="text-2xl font-semibold text-gray-900">
                          {stat.value}
                        </div>
                        {stat.change !== undefined && (
                          <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                            stat.change >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {stat.change >= 0 ? '↗' : '↘'}
                            {Math.abs(stat.change)}%
                          </div>
                        )}
                      </dd>
                    </dl>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  );
};
interface MainLayoutProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  header?: React.ReactNode;
  navigation?: React.ReactNode;
}
export const MainLayout: React.FC<MainLayoutProps> = ({ 
  children, 
  sidebar, 
  header, 
  navigation 
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <div className="min-h-screen bg-gray-50">
      {}
      {navigation}
      <div className="flex">
        {}
        {sidebar && (
          <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)}>
            {sidebar}
          </Sidebar>
        )}
        {}
        <div className="flex-1 flex flex-col">
          {}
          {header}
          {}
          <main className="flex-1">
            <motion.div
              className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              {children}
            </motion.div>
          </main>
        </div>
      </div>
      {}
      {sidebar && (
        <motion.button
          className="lg:hidden fixed bottom-4 left-4 z-50 bg-blue-600 text-white p-3 rounded-full shadow-lg"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          ☰
        </motion.button>
      )}
    </div>
  );
};
interface GridLayoutProps {
  children: React.ReactNode;
  columns?: 1 | 2 | 3 | 4;
  gap?: 'sm' | 'md' | 'lg';
  className?: string;
}
export const GridLayout: React.FC<GridLayoutProps> = ({ 
  children, 
  columns = 2, 
  gap = 'md',
  className = ''
}) => {
  const columnClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 lg:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4'
  };
  const gapClasses = {
    sm: 'gap-4',
    md: 'gap-6',
    lg: 'gap-8'
  };
  return (
    <div className={`grid ${columnClasses[columns]} ${gapClasses[gap]} ${className}`}>
      {children}
    </div>
  );
};
interface DashboardStatsProps {
  totalSessions: number;
  avgEngagement: number;
  totalStudents: number;
  activeNow: number;
}
export const DashboardStats: React.FC<DashboardStatsProps> = ({
  totalSessions,
  avgEngagement,
  totalStudents,
  activeNow
}) => {
  const stats = [
    {
      label: 'Total Sessions',
      value: totalSessions.toLocaleString(),
      icon: '📊',
      change: 12
    },
    {
      label: 'Avg. Engagement',
      value: `${Math.round(avgEngagement * 100)}%`,
      icon: '🎯',
      change: 5
    },
    {
      label: 'Total Students',
      value: totalStudents.toLocaleString(),
      icon: '👥',
      change: 8
    },
    {
      label: 'Active Now',
      value: activeNow.toLocaleString(),
      icon: '🟢',
      change: -2
    }
  ];
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
        >
          <Card className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0 text-3xl mr-4">
                {stat.icon}
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">
                  {stat.label}
                </p>
                <div className="flex items-baseline">
                  <p className="text-2xl font-semibold text-gray-900">
                    {stat.value}
                  </p>
                  <p className={`ml-2 text-sm font-semibold ${
                    stat.change >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {stat.change >= 0 ? '+' : ''}{stat.change}%
                  </p>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      ))}
    </div>
  );
};
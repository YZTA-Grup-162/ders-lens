import { motion } from 'framer-motion';
import { Github, Mail, Shield } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <motion.footer 
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="bg-gray-900 dark:bg-gray-950 text-white py-12"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Logo & About */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <img
                src="/ders-lens-logo.png"
                alt="DersLens"
                className="h-8 w-auto"
              />
              <span className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                DersLens
              </span>
            </div>
            <p className="text-gray-400 mb-4">
              Yapay zeka destekli sınıf analizi ile eğitimin geleceğini şekillendiriyoruz.
              Öğrenci dikkatini, katılımını ve duygularını gerçek zamanlı takip edin.
            </p>
            <div className="flex space-x-4">
              <a
                href="/github"
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="GitHub"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="/iletisim"
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="İletişim"
              >
                <Mail className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold mb-4">Hızlı Erişim</h3>
            <ul className="space-y-2 text-gray-400">
              <li>
                <a href="/demo" className="hover:text-white transition-colors">
                  Demo
                </a>
              </li>
              <li>
                <a href="/dashboard" className="hover:text-white transition-colors">
                  Öğretmen Paneli
                </a>
              </li>
              <li>
                <a href="/features" className="hover:text-white transition-colors">
                  Özellikler
                </a>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="font-semibold mb-4">Yasal</h3>
            <ul className="space-y-2 text-gray-400">
              <li>
                <a href="/gizlilik" className="hover:text-white transition-colors flex items-center">
                  <Shield className="h-4 w-4 mr-1" />
                  Gizlilik
                </a>
              </li>
              <li>
                <span className="text-green-400 text-sm">GDPR Uyumlu</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400">
          <p>&copy; {currentYear} DersLens. Tüm hakları saklıdır.</p>
        </div>
      </div>
    </motion.footer>
  );
};

export default Footer;

import { motion } from 'framer-motion';
import { Brain, Code, ExternalLink, Github, Heart, Mail } from 'lucide-react';
export function Footer() {
  const currentYear = new Date().getFullYear();
  const footerLinks = {
    product: {
      title: 'Ürün',
      links: [
        { name: 'Özellikler', href: '#features' },
        { name: 'Canlı Demo', href: '#demo' },
        { name: 'AI Modelleri', href: '#ai' },
        { name: 'Fiyatlandırma', href: '#pricing' }
      ]
    },
    resources: {
      title: 'Kaynaklar',
      links: [
        { name: 'Dokümantasyon', href: '/docs' },
        { name: 'API Referansı', href: '/api' },
        { name: 'Eğitim Videoları', href: '/tutorials' },
        { name: 'Topluluk', href: '/community' }
      ]
    },
    support: {
      title: 'Destek',
      links: [
        { name: 'Yardım Merkezi', href: '/help' },
        { name: 'İletişim', href: '/contact' },
        { name: 'Bug Raporu', href: '/bugs' },
        { name: 'Özellik Talebi', href: '/features' }
      ]
    },
    company: {
      title: 'Şirket',
      links: [
        { name: 'Hakkımızda', href: '/about' },
        { name: 'Blog', href: '/blog' },
        { name: 'Kariyer', href: '/careers' },
        { name: 'Basın Kiti', href: '/press' }
      ]
    }
  };
  const socialLinks = [
    { 
      name: 'GitHub', 
      href: 'https://github.com/badicev/ders-lens-pri',
      icon: <Github className="w-5 h-5" />,
      color: 'hover:text-gray-300'
    },
    { 
      name: 'Email', 
      href: 'mailto:info@ders-lens.com', 
      icon: <Mail className="w-5 h-5" />,
      color: 'hover:text-blue-400'
    },
    { 
      name: 'Dokümantasyon', 
      href: '/docs', 
      icon: <ExternalLink className="w-5 h-5" />,
      color: 'hover:text-cyan-400'
    }
  ];
  return (
    <footer className="bg-black border-t border-white/10 relative overflow-hidden">
      {}
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="footerGrid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#footerGrid)" />
        </svg>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <div className="py-16">
          <div className="grid lg:grid-cols-6 gap-8">
            {}
            <div className="lg:col-span-2">
              <motion.div
                className="flex items-center mb-6"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                viewport={{ once: true }}
              >
                <motion.div
                  className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center mr-3"
                  whileHover={{ rotate: 5, scale: 1.1 }}
                >
                  <Brain className="w-6 h-6 text-white" />
                </motion.div>
                <div>
                  <div className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    Ders Lens
                  </div>
                  <div className="text-xs text-gray-500">AI Powered Education</div>
                </div>
              </motion.div>
              <motion.p
                className="text-gray-400 text-sm leading-relaxed mb-6"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                viewport={{ once: true }}
              >
                Yapay zeka destekli öğrenci dikkat, katılım ve duygu analizi platformu. 
                Eğitim ortamlarında öğrenme deneyimini optimize eden yenilikçi teknoloji.
              </motion.p>
              {}
              <motion.div
                className="flex space-x-4"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                viewport={{ once: true }}
              >
                {socialLinks.map((social) => (
                  <motion.a
                    key={social.name}
                    href={social.href}
                    className={`w-10 h-10 bg-white/5 border border-white/10 rounded-xl flex items-center justify-center text-gray-400 transition-all duration-300 ${social.color}`}
                    whileHover={{ y: -2, backgroundColor: "rgba(255,255,255,0.1)" }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {social.icon}
                  </motion.a>
                ))}
              </motion.div>
            </div>
            {}
            {Object.entries(footerLinks).map(([key, section], sectionIndex) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 + sectionIndex * 0.1 }}
                viewport={{ once: true }}
              >
                <h3 className="text-white font-semibold mb-4">{section.title}</h3>
                <ul className="space-y-3">
                  {section.links.map((link, linkIndex) => (
                    <motion.li
                      key={link.name}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.4, delay: 0.4 + sectionIndex * 0.1 + linkIndex * 0.05 }}
                      viewport={{ once: true }}
                    >
                      <a
                        href={link.href}
                        className="text-gray-400 text-sm hover:text-white transition-colors duration-200"
                      >
                        {link.name}
                      </a>
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
        {}
        <motion.div
          className="py-8 border-t border-white/10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="text-center mb-6">
            <h4 className="text-gray-300 text-sm font-medium mb-4">Teknoloji Yığını</h4>
            <div className="flex flex-wrap justify-center gap-3">
              {[
                'React 18', 'TypeScript', 'Tailwind CSS', 'Framer Motion',
                'Node.js', 'Python', 'ONNX', 'WebRTC', 'Docker', 'WebSocket'
              ].map((tech, index) => (
                <motion.span
                  key={tech}
                  className="bg-white/5 border border-white/10 rounded-full px-3 py-1 text-xs text-gray-400"
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.9 + index * 0.05 }}
                  viewport={{ once: true }}
                  whileHover={{ backgroundColor: "rgba(255,255,255,0.1)", scale: 1.05 }}
                >
                  {tech}
                </motion.span>
              ))}
            </div>
          </div>
        </motion.div>
        {}
        <motion.div
          className="py-6 border-t border-white/10 flex flex-col md:flex-row justify-between items-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1 }}
          viewport={{ once: true }}
        >
          <div className="text-gray-500 text-sm mb-4 md:mb-0">
            © {currentYear} Ders Lens. Tüm hakları saklıdır.
          </div>
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <a href="/privacy" className="hover:text-gray-300 transition-colors">
              Gizlilik Politikası
            </a>
            <span>•</span>
            <a href="/terms" className="hover:text-gray-300 transition-colors">
              Kullanım Şartları
            </a>
            <span>•</span>
            <div className="flex items-center">
              <span>Yapıldı</span>
              <motion.div
                className="mx-1"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                <Heart className="w-4 h-4 text-red-500" />
              </motion.div>
              <span>ile</span>
              <Code className="w-4 h-4 ml-1 text-blue-400" />
            </div>
          </div>
        </motion.div>
        {}
        <motion.div
          className="py-4 border-t border-white/10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.1 }}
          viewport={{ once: true }}
        >
          <p className="text-xs text-gray-600">
            FER2013+, DAISEE ve MPIIGaze veri setleri kullanılarak eğitilmiş AI modelleri • 
            Açık kaynak kodlu eğitim teknolojisi projesi
          </p>
        </motion.div>
      </div>
    </footer>
  );
}
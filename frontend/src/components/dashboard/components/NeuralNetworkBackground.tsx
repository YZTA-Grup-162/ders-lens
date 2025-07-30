import { motion } from 'framer-motion';
export function NeuralNetworkBackground() {
  const nodes = Array.from({ length: 50 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 3 + 1,
    delay: Math.random() * 2
  }));
  const connections = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const distance = Math.sqrt(
        Math.pow(nodes[i].x - nodes[j].x, 2) + Math.pow(nodes[i].y - nodes[j].y, 2)
      );
      if (distance < 25) {
        connections.push({
          from: nodes[i],
          to: nodes[j],
          opacity: Math.max(0.1, 1 - distance / 25)
        });
      }
    }
  }
  return (
    <div className="fixed inset-0 z-0 overflow-hidden">
      {}
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-black via-blue-950/50 to-purple-950/50"
        animate={{
          background: [
            "linear-gradient(45deg, #000000 0%, #1e3a8a 50%, #581c87 100%)",
            "linear-gradient(45deg, #000000 0%, #1e40af 50%, #7c3aed 100%)",
            "linear-gradient(45deg, #000000 0%, #2563eb 50%, #8b5cf6 100%)",
            "linear-gradient(45deg, #000000 0%, #1e40af 50%, #581c87 100%)"
          ]
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          repeatType: "reverse"
        }}
      />
      {}
      <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <radialGradient id="nodeGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#60A5FA" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.2"/>
          </radialGradient>
          <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#60A5FA" stopOpacity="0.6"/>
            <stop offset="100%" stopColor="#06B6D4" stopOpacity="0.2"/>
          </linearGradient>
        </defs>
        {}
        {connections.map((connection, i) => (
          <motion.line
            key={`connection-${i}`}
            x1={connection.from.x}
            y1={connection.from.y}
            x2={connection.to.x}
            y2={connection.to.y}
            stroke="url(#connectionGradient)"
            strokeWidth="0.2"
            opacity={connection.opacity}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{
              duration: 3,
              delay: i * 0.05,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
        ))}
        {}
        {nodes.map((node) => (
          <motion.circle
            key={`node-${node.id}`}
            cx={node.x}
            cy={node.y}
            r={node.size}
            fill="url(#nodeGradient)"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ 
              scale: [0.8, 1.2, 0.8],
              opacity: [0.3, 0.8, 0.3]
            }}
            transition={{
              duration: 4,
              delay: node.delay,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
        ))}
        {}
        {Array.from({ length: 10 }).map((_, i) => (
          <motion.circle
            key={`pulse-${i}`}
            cx={Math.random() * 100}
            cy={Math.random() * 100}
            r="0.5"
            fill="#00D4FF"
            initial={{ scale: 0, opacity: 1 }}
            animate={{ 
              scale: [0, 3, 0],
              opacity: [1, 0.3, 0]
            }}
            transition={{
              duration: 3,
              delay: i * 0.3,
              repeat: Infinity,
              repeatType: "loop"
            }}
          />
        ))}
      </svg>
      {}
      <div className="absolute inset-0">
        {Array.from({ length: 20 }).map((_, i) => (
          <motion.div
            key={`particle-${i}`}
            className="absolute w-1 h-1 bg-blue-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -30, 0],
              x: [0, Math.random() * 20 - 10, 0],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: 5 + Math.random() * 3,
              delay: Math.random() * 2,
              repeat: Infinity
            }}
          />
        ))}
      </div>
      {}
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="neuralGrid" width="5" height="5" patternUnits="userSpaceOnUse">
              <path d="M 5 0 L 0 0 0 5" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#neuralGrid)" />
        </svg>
      </div>
    </div>
  );
}
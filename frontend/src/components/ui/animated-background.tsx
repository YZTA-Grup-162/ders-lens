import React, { useEffect, useRef } from 'react';

interface AnimatedBackgroundProps {
  variant?: 'neural' | 'particles' | 'waves' | 'grid';
  intensity?: 'low' | 'medium' | 'high';
  color?: 'blue' | 'cyan' | 'purple' | 'green' | 'rainbow';
  className?: string;
}

export const AnimatedBackground: React.FC<AnimatedBackgroundProps> = ({
  variant = 'neural',
  intensity = 'medium',
  color = 'blue',
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const nodesRef = useRef<Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
    opacity: number;
    pulsePhase: number;
  }>>([]);
  const connectionsRef = useRef<Array<{
    from: number;
    to: number;
    strength: number;
    animated: boolean;
  }>>([]);

  const getColorPalette = (colorScheme: string) => {
    switch (colorScheme) {
      case 'cyan':
        return {
          primary: [6, 182, 212], // cyan-500
          secondary: [103, 232, 249], // cyan-300
          accent: [8, 145, 178] // cyan-600
        };
      case 'purple':
        return {
          primary: [147, 51, 234], // purple-600
          secondary: [196, 181, 253], // purple-300
          accent: [124, 58, 237] // purple-600
        };
      case 'green':
        return {
          primary: [16, 185, 129], // emerald-500
          secondary: [110, 231, 183], // emerald-300
          accent: [5, 150, 105] // emerald-600
        };
      case 'rainbow':
        return {
          primary: [59, 130, 246], // blue-500
          secondary: [16, 185, 129], // emerald-500
          accent: [147, 51, 234] // purple-600
        };
      default: // blue
        return {
          primary: [59, 130, 246], // blue-500
          secondary: [147, 197, 253], // blue-300
          accent: [29, 78, 216] // blue-700
        };
    }
  };

  const getIntensitySettings = (level: string) => {
    switch (level) {
      case 'low':
        return { nodeCount: 30, connectionDensity: 0.05, animationSpeed: 0.3 };
      case 'high':
        return { nodeCount: 80, connectionDensity: 0.15, animationSpeed: 0.8 };
      default: // medium
        return { nodeCount: 50, connectionDensity: 0.1, animationSpeed: 0.5 };
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const colors = getColorPalette(color);
    const settings = getIntensitySettings(intensity);

    // Initialize nodes
    const initializeNodes = () => {
      nodesRef.current = [];
      for (let i = 0; i < settings.nodeCount; i++) {
        nodesRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * settings.animationSpeed,
          vy: (Math.random() - 0.5) * settings.animationSpeed,
          size: Math.random() * 3 + 1,
          opacity: Math.random() * 0.5 + 0.2,
          pulsePhase: Math.random() * Math.PI * 2
        });
      }
    };

    // Initialize connections
    const initializeConnections = () => {
      connectionsRef.current = [];
      for (let i = 0; i < nodesRef.current.length; i++) {
        for (let j = i + 1; j < nodesRef.current.length; j++) {
          if (Math.random() < settings.connectionDensity) {
            connectionsRef.current.push({
              from: i,
              to: j,
              strength: Math.random(),
              animated: Math.random() < 0.3
            });
          }
        }
      }
    };

    const drawNeuralNetwork = (time: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Update and draw connections
      connectionsRef.current.forEach((conn, index) => {
        const fromNode = nodesRef.current[conn.from];
        const toNode = nodesRef.current[conn.to];
        
        if (!fromNode || !toNode) return;

        const distance = Math.sqrt(
          Math.pow(fromNode.x - toNode.x, 2) + Math.pow(fromNode.y - toNode.y, 2)
        );

        if (distance < 150) {
          let opacity = (1 - distance / 150) * conn.strength * 0.4;
          
          // Animated connections pulse
          if (conn.animated) {
            opacity *= (Math.sin(time * 0.003 + index * 0.5) + 1) / 2;
          }

          const [r, g, b] = color === 'rainbow' 
            ? index % 3 === 0 ? colors.primary 
              : index % 3 === 1 ? colors.secondary 
              : colors.accent
            : colors.primary;

          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
          ctx.lineWidth = conn.animated ? 2 : 1;
          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          ctx.stroke();

          // Energy pulse effect on animated connections
          if (conn.animated && opacity > 0.2) {
            const pulsePos = (time * 0.002 + index) % 1;
            const pulseX = fromNode.x + (toNode.x - fromNode.x) * pulsePos;
            const pulseY = fromNode.y + (toNode.y - fromNode.y) * pulsePos;
            
            ctx.beginPath();
            ctx.arc(pulseX, pulseY, 3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
            ctx.fill();
          }
        }
      });

      // Update and draw nodes
      nodesRef.current.forEach((node, index) => {
        // Update position
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off walls
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Keep nodes in bounds
        node.x = Math.max(0, Math.min(canvas.width, node.x));
        node.y = Math.max(0, Math.min(canvas.height, node.y));

        // Update pulse
        node.pulsePhase += 0.02;
        const pulseScale = 1 + Math.sin(node.pulsePhase) * 0.3;

        // Draw node
        const [r, g, b] = color === 'rainbow' 
          ? index % 3 === 0 ? colors.primary 
            : index % 3 === 1 ? colors.secondary 
            : colors.accent
          : colors.secondary;

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size * pulseScale, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${node.opacity})`;
        ctx.fill();

        // Add glow effect for larger nodes
        if (node.size > 2) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.size * pulseScale * 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${node.opacity * 0.2})`;
          ctx.fill();
        }
      });
    };

    const drawParticles = (time: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      nodesRef.current.forEach((particle, index) => {
        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;

        // Wrap around screen
        if (particle.x < -10) particle.x = canvas.width + 10;
        if (particle.x > canvas.width + 10) particle.x = -10;
        if (particle.y < -10) particle.y = canvas.height + 10;
        if (particle.y > canvas.height + 10) particle.y = -10;

        // Update opacity with pulse
        const pulseOpacity = particle.opacity * (Math.sin(time * 0.001 + index) + 1) / 2;

        const [r, g, b] = colors.primary;
        
        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulseOpacity})`;
        ctx.fill();

        // Draw trail
        ctx.beginPath();
        ctx.arc(
          particle.x - particle.vx * 10, 
          particle.y - particle.vy * 10, 
          particle.size * 0.5, 0, Math.PI * 2
        );
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulseOpacity * 0.3})`;
        ctx.fill();
      });
    };

    const drawWaves = (time: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const [r, g, b] = colors.primary;
      const [r2, g2, b2] = colors.secondary;
      
      // Draw multiple wave layers
      for (let layer = 0; layer < 3; layer++) {
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        
        for (let x = 0; x <= canvas.width; x += 5) {
          const y = canvas.height / 2 + 
            Math.sin((x * 0.01) + (time * 0.001) + (layer * 0.5)) * (30 + layer * 20) +
            Math.sin((x * 0.02) + (time * 0.002) + (layer * 0.3)) * (15 + layer * 10);
          
          ctx.lineTo(x, y);
        }
        
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        ctx.closePath();
        
        const opacity = 0.1 + (layer * 0.05);
        const currentColor = layer % 2 === 0 ? [r, g, b] : [r2, g2, b2];
        ctx.fillStyle = `rgba(${currentColor[0]}, ${currentColor[1]}, ${currentColor[2]}, ${opacity})`;
        ctx.fill();
      }
    };

    const drawGrid = (time: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const [r, g, b] = colors.primary;
      const gridSize = 50;
      const pulseIntensity = (Math.sin(time * 0.002) + 1) / 2;
      
      // Draw grid lines
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.1 + pulseIntensity * 0.2})`;
      ctx.lineWidth = 1;
      
      // Vertical lines
      for (let x = 0; x <= canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      
      // Horizontal lines
      for (let y = 0; y <= canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }
      
      // Draw intersection points
      for (let x = 0; x <= canvas.width; x += gridSize) {
        for (let y = 0; y <= canvas.height; y += gridSize) {
          const distance = Math.sqrt(
            Math.pow(x - canvas.width / 2, 2) + Math.pow(y - canvas.height / 2, 2)
          );
          const normalizedDistance = distance / (Math.sqrt(canvas.width ** 2 + canvas.height ** 2) / 2);
          const opacity = (1 - normalizedDistance) * pulseIntensity * 0.5;
          
          if (opacity > 0.1) {
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
            ctx.fill();
          }
        }
      }
    };

    // Initialize based on variant
    if (variant === 'neural') {
      initializeNodes();
      initializeConnections();
    } else if (variant === 'particles') {
      initializeNodes();
      // Modify nodes for particle behavior
      nodesRef.current.forEach(node => {
        node.vx *= 2; // Faster movement
        node.vy *= 2;
        node.size *= 0.5; // Smaller particles
      });
    }

    // Animation loop
    const animate = (currentTime: number) => {
      switch (variant) {
        case 'neural':
          drawNeuralNetwork(currentTime);
          break;
        case 'particles':
          drawParticles(currentTime);
          break;
        case 'waves':
          drawWaves(currentTime);
          break;
        case 'grid':
          drawGrid(currentTime);
          break;
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [variant, intensity, color]);

  return (
    <canvas
      ref={canvasRef}
      className={`fixed inset-0 pointer-events-none z-0 ${className}`}
      style={{ 
        mixBlendMode: 'screen',
        opacity: 0.6
      }}
    />
  );
};

export default AnimatedBackground;

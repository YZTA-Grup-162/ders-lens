import { useEffect, useRef } from 'react';
interface Particle {
  x: number;
  y: number;
  dx: number;
  dy: number;
  size: number;
  opacity: number;
}
interface FuturisticBackgroundProps {
  className?: string;
}
const FuturisticBackground = ({ className = '' }: FuturisticBackgroundProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const particlesRef = useRef<Particle[]>([]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      initParticles();
    };
    const initParticles = () => {
      particlesRef.current = [];
      const particleCount = Math.floor((canvas.width * canvas.height) / 15000);
      for (let i = 0; i < particleCount; i++) {
        particlesRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          dx: (Math.random() - 0.5) * 0.5,
          dy: (Math.random() - 0.5) * 0.5,
          size: Math.random() * 2 + 1,
          opacity: Math.random() * 0.5 + 0.2,
        });
      }
    };
    const drawParticles = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.1)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < particlesRef.current.length; i++) {
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const particle1 = particlesRef.current[i];
          const particle2 = particlesRef.current[j];
          const dx = particle1.x - particle2.x;
          const dy = particle1.y - particle2.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < 100) {
            const opacity = (100 - distance) / 100 * 0.3;
            ctx.globalAlpha = opacity;
            ctx.beginPath();
            ctx.moveTo(particle1.x, particle1.y);
            ctx.lineTo(particle2.x, particle2.y);
            ctx.stroke();
          }
        }
      }
      particlesRef.current.forEach((particle) => {
        ctx.globalAlpha = particle.opacity;
        ctx.fillStyle = 'rgb(59, 130, 246)';
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
        particle.x += particle.dx;
        particle.y += particle.dy;
        if (particle.x < 0 || particle.x > canvas.width) particle.dx *= -1;
        if (particle.y < 0 || particle.y > canvas.height) particle.dy *= -1;
        particle.x = Math.max(0, Math.min(canvas.width, particle.x));
        particle.y = Math.max(0, Math.min(canvas.height, particle.y));
      });
      ctx.globalAlpha = 1;
    };
    const animate = () => {
      drawParticles();
      animationRef.current = requestAnimationFrame(animate);
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    animate();
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ opacity: 0.4 }}
    />
  );
};
export default FuturisticBackground;
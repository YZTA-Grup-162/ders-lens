import React, { useEffect, useRef } from 'react';
interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
}
const ParticleBackground: React.FC<{ isDarkMode: boolean }> = ({ isDarkMode }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    const createParticles = () => {
      const particles: Particle[] = [];
      const particleCount = 50;
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          size: Math.random() * 2 + 1,
          opacity: Math.random() * 0.5 + 0.1,
        });
      }
      particlesRef.current = particles;
    };
    const drawParticles = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particlesRef.current.forEach((particle) => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = isDarkMode 
          ? `rgba(59, 130, 246, ${particle.opacity})` 
          : `rgba(99, 102, 241, ${particle.opacity})`;
        ctx.fill();
        particlesRef.current.forEach((otherParticle) => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < 100) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            const opacity = (100 - distance) / 100 * 0.1;
            ctx.strokeStyle = isDarkMode 
              ? `rgba(59, 130, 246, ${opacity})` 
              : `rgba(99, 102, 241, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });
    };
    const animate = () => {
      drawParticles();
      animationRef.current = requestAnimationFrame(animate);
    };
    resizeCanvas();
    createParticles();
    animate();
    window.addEventListener('resize', () => {
      resizeCanvas();
      createParticles();
    });
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [isDarkMode]);
  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none opacity-30"
      style={{ zIndex: 1 }}
    />
  );
};
export default ParticleBackground;
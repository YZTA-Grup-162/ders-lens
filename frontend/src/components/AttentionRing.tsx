import React from 'react';
interface AttentionRingProps {
  attentionScore: number;
  engagementScore: number;
  size?: number;
  className?: string;
}
export const AttentionRing: React.FC<AttentionRingProps> = ({
  attentionScore,
  engagementScore,
  size = 120,
  className = ''
}) => {
  const radius = size / 2 - 10;
  const circumference = 2 * Math.PI * radius;
  const attentionOffset = circumference - (attentionScore * circumference);
  const engagementOffset = circumference - (engagementScore * circumference);
  const getAttentionColor = (score: number) => {
    if (score >= 0.7) return '#10b981'; 
    if (score >= 0.4) return '#f59e0b'; 
    return '#ef4444'; 
  };
  const getEngagementColor = (score: number) => {
    if (score >= 0.7) return '#3b82f6'; 
    if (score >= 0.4) return '#8b5cf6'; 
    return '#6b7280'; 
  };
  return (
    <div className={`relative ${className}`}>
      <svg width={size} height={size} className="transform -rotate-90">
        {}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="#e5e7eb"
          strokeWidth="8"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius - 15}
          fill="transparent"
          stroke="#e5e7eb"
          strokeWidth="6"
        />
        {}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke={getAttentionColor(attentionScore)}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={attentionOffset}
          strokeLinecap="round"
          className="transition-all duration-300 ease-out"
        />
        {}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius - 15}
          fill="transparent"
          stroke={getEngagementColor(engagementScore)}
          strokeWidth="6"
          strokeDasharray={circumference}
          strokeDashoffset={engagementOffset}
          strokeLinecap="round"
          className="transition-all duration-300 ease-out"
        />
      </svg>
      {}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800">
            {Math.round(attentionScore * 100)}%
          </div>
          <div className="text-sm text-gray-600">
            Attention
          </div>
        </div>
      </div>
      {}
      <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
        <div className="flex space-x-4 text-xs">
          <div className="flex items-center">
            <div 
              className="w-3 h-3 rounded-full mr-1" 
              style={{ backgroundColor: getAttentionColor(attentionScore) }}
            />
            <span>Attention</span>
          </div>
          <div className="flex items-center">
            <div 
              className="w-3 h-3 rounded-full mr-1" 
              style={{ backgroundColor: getEngagementColor(engagementScore) }}
            />
            <span>Engagement</span>
          </div>
        </div>
      </div>
    </div>
  );
};
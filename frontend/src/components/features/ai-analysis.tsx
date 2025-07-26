import { AnimatePresence, motion } from 'framer-motion';
import React, { useEffect, useState } from 'react';
import { Badge, Button, Card, Progress } from '../ui';
interface EmotionData {
  emotion: string;
  confidence: number;
  timestamp: number;
}
interface AttentionData {
  attentionScore: number;
  headPose: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  gazeDirection: {
    x: number;
    y: number;
  };
  timestamp: number;
}
interface EngagementData {
  engagementLevel: number;
  factors: {
    eyeContact: number;
    facialExpression: number;
    headMovement: number;
    bodyLanguage: number;
  };
  timestamp: number;
}
interface EmotionDetectionProps {
  emotionData: EmotionData[];
  isActive: boolean;
}
export const EmotionDetection: React.FC<EmotionDetectionProps> = ({ 
  emotionData, 
  isActive 
}) => {
  const [currentEmotion, setCurrentEmotion] = useState<EmotionData | null>(null);
  useEffect(() => {
    if (emotionData.length > 0) {
      setCurrentEmotion(emotionData[emotionData.length - 1]);
    }
  }, [emotionData]);
  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: 'text-green-600 bg-green-100',
      sad: 'text-blue-600 bg-blue-100',
      angry: 'text-red-600 bg-red-100',
      surprised: 'text-yellow-600 bg-yellow-100',
      disgusted: 'text-purple-600 bg-purple-100',
      fearful: 'text-gray-600 bg-gray-100',
      neutral: 'text-slate-600 bg-slate-100'
    };
    return colors[emotion.toLowerCase()] || 'text-gray-600 bg-gray-100';
  };
  const getEmotionEmoji = (emotion: string) => {
    const emojis: Record<string, string> = {
      happy: '😊',
      sad: '😢',
      angry: '😠',
      surprised: '😲',
      disgusted: '🤢',
      fearful: '😨',
      neutral: '😐'
    };
    return emojis[emotion.toLowerCase()] || '😐';
  };
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Emotion Detection</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-300'}`} />
          <span className="text-sm text-gray-500">
            {isActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
      <AnimatePresence mode="wait">
        {currentEmotion ? (
          <motion.div
            key={currentEmotion.timestamp}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.3 }}
            className="text-center space-y-4"
          >
            <div className="text-6xl mb-2">
              {getEmotionEmoji(currentEmotion.emotion)}
            </div>
            <Badge 
              variant="default" 
              size="lg"
              className={getEmotionColor(currentEmotion.emotion)}
            >
              {currentEmotion.emotion.charAt(0).toUpperCase() + currentEmotion.emotion.slice(1)}
            </Badge>
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-600">
                <span>Confidence</span>
                <span>{Math.round(currentEmotion.confidence * 100)}%</span>
              </div>
              <Progress 
                value={currentEmotion.confidence * 100}
                color="blue"
              />
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8 text-gray-500"
          >
            <div className="text-4xl mb-2">🔍</div>
            <p>Waiting for emotion data...</p>
          </motion.div>
        )}
      </AnimatePresence>
      {}
      {emotionData.length > 1 && (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Recent Emotions</h4>
          <div className="flex flex-wrap gap-2">
            {emotionData.slice(-5).map((emotion, index) => (
              <motion.div
                key={emotion.timestamp}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center space-x-1 text-sm"
              >
                <span>{getEmotionEmoji(emotion.emotion)}</span>
                <span className="text-gray-600">
                  {Math.round(emotion.confidence * 100)}%
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
};
interface AttentionTrackingProps {
  attentionData: AttentionData[];
  isActive: boolean;
}
export const AttentionTracking: React.FC<AttentionTrackingProps> = ({ 
  attentionData, 
  isActive 
}) => {
  const [currentAttention, setCurrentAttention] = useState<AttentionData | null>(null);
  useEffect(() => {
    if (attentionData.length > 0) {
      setCurrentAttention(attentionData[attentionData.length - 1]);
    }
  }, [attentionData]);
  const getAttentionLevel = (score: number) => {
    if (score >= 0.8) return { level: 'High', color: 'green', icon: '🎯' };
    if (score >= 0.6) return { level: 'Medium', color: 'yellow', icon: '👀' };
    return { level: 'Low', color: 'red', icon: '💭' };
  };
  const formatAngle = (angle: number) => {
    return `${Math.round(angle)}°`;
  };
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Attention Tracking</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-300'}`} />
          <span className="text-sm text-gray-500">
            {isActive ? 'Tracking' : 'Inactive'}
          </span>
        </div>
      </div>
      <AnimatePresence mode="wait">
        {currentAttention ? (
          <motion.div
            key={currentAttention.timestamp}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {}
            <div className="text-center">
              <div className="text-4xl mb-2">
                {getAttentionLevel(currentAttention.attentionScore).icon}
              </div>
              <Badge 
                variant={getAttentionLevel(currentAttention.attentionScore).color as any}
                size="lg"
              >
                {getAttentionLevel(currentAttention.attentionScore).level} Attention
              </Badge>
              <div className="mt-4">
                <Progress 
                  value={currentAttention.attentionScore * 100}
                  color={getAttentionLevel(currentAttention.attentionScore).color as any}
                />
                <p className="text-sm text-gray-600 mt-1">
                  {Math.round(currentAttention.attentionScore * 100)}% Attention Level
                </p>
              </div>
            </div>
            {}
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-sm text-gray-500">Pitch</div>
                <div className="font-medium">
                  {formatAngle(currentAttention.headPose.pitch)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">Yaw</div>
                <div className="font-medium">
                  {formatAngle(currentAttention.headPose.yaw)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">Roll</div>
                <div className="font-medium">
                  {formatAngle(currentAttention.headPose.roll)}
                </div>
              </div>
            </div>
            {}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Gaze Direction</h4>
              <div className="relative w-32 h-32 mx-auto bg-white rounded-lg border-2 border-gray-200">
                <motion.div
                  className="absolute w-3 h-3 bg-blue-500 rounded-full"
                  style={{
                    left: `${50 + (currentAttention.gazeDirection.x * 25)}%`,
                    top: `${50 + (currentAttention.gazeDirection.y * 25)}%`,
                    transform: 'translate(-50%, -50%)'
                  }}
                  animate={{
                    left: `${50 + (currentAttention.gazeDirection.x * 25)}%`,
                    top: `${50 + (currentAttention.gazeDirection.y * 25)}%`
                  }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-xs">
                  Screen
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8 text-gray-500"
          >
            <div className="text-4xl mb-2">👁️</div>
            <p>Initializing attention tracking...</p>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
};
interface EngagementAnalysisProps {
  engagementData: EngagementData[];
  isActive: boolean;
}
export const EngagementAnalysis: React.FC<EngagementAnalysisProps> = ({ 
  engagementData, 
  isActive 
}) => {
  const [currentEngagement, setCurrentEngagement] = useState<EngagementData | null>(null);
  useEffect(() => {
    if (engagementData.length > 0) {
      setCurrentEngagement(engagementData[engagementData.length - 1]);
    }
  }, [engagementData]);
  const getEngagementLevel = (level: number) => {
    if (level >= 0.8) return { label: 'Highly Engaged', color: 'green', icon: '🚀' };
    if (level >= 0.6) return { label: 'Moderately Engaged', color: 'yellow', icon: '📚' };
    if (level >= 0.4) return { label: 'Somewhat Engaged', color: 'yellow', icon: '🤔' };
    return { label: 'Low Engagement', color: 'red', icon: '😴' };
  };
  const factorLabels = {
    eyeContact: 'Eye Contact',
    facialExpression: 'Facial Expression',
    headMovement: 'Head Movement',
    bodyLanguage: 'Body Language'
  };
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Engagement Analysis</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-300'}`} />
          <span className="text-sm text-gray-500">
            {isActive ? 'Analyzing' : 'Inactive'}
          </span>
        </div>
      </div>
      <AnimatePresence mode="wait">
        {currentEngagement ? (
          <motion.div
            key={currentEngagement.timestamp}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {}
            <div className="text-center">
              <div className="text-4xl mb-2">
                {getEngagementLevel(currentEngagement.engagementLevel).icon}
              </div>
              <Badge 
                variant={getEngagementLevel(currentEngagement.engagementLevel).color as any}
                size="lg"
              >
                {getEngagementLevel(currentEngagement.engagementLevel).label}
              </Badge>
              <div className="mt-4">
                <Progress 
                  value={currentEngagement.engagementLevel * 100}
                  color={getEngagementLevel(currentEngagement.engagementLevel).color as any}
                />
                <p className="text-sm text-gray-600 mt-1">
                  {Math.round(currentEngagement.engagementLevel * 100)}% Overall Engagement
                </p>
              </div>
            </div>
            {}
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-700">Engagement Factors</h4>
              {Object.entries(currentEngagement.factors).map(([key, value], index) => (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="space-y-2"
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">
                      {factorLabels[key as keyof typeof factorLabels]}
                    </span>
                    <span className="font-medium">
                      {Math.round(value * 100)}%
                    </span>
                  </div>
                  <Progress 
                    value={value * 100}
                    color={value >= 0.7 ? 'green' : value >= 0.5 ? 'yellow' : 'red'}
                  />
                </motion.div>
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8 text-gray-500"
          >
            <div className="text-4xl mb-2">📊</div>
            <p>Analyzing engagement patterns...</p>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
};
interface VideoFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isStreamActive: boolean;
  onStartStream: () => void;
  onStopStream: () => void;
  overlayEnabled: boolean;
  onToggleOverlay: () => void;
}
export const VideoFeed: React.FC<VideoFeedProps> = ({
  videoRef,
  isStreamActive,
  onStartStream,
  onStopStream,
  overlayEnabled,
  onToggleOverlay
}) => {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Live Video Feed</h3>
        <div className="flex items-center space-x-2">
          <Button
            variant={overlayEnabled ? 'primary' : 'outline'}
            size="sm"
            onClick={onToggleOverlay}
          >
            AI Overlay
          </Button>
          <Button
            variant={isStreamActive ? 'secondary' : 'primary'}
            size="sm"
            onClick={isStreamActive ? onStopStream : onStartStream}
          >
            {isStreamActive ? 'Stop' : 'Start'} Stream
          </Button>
        </div>
      </div>
      <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
        />
        {!isStreamActive && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50">
            <div className="text-center text-white">
              <div className="text-4xl mb-2">📹</div>
              <p>Click "Start Stream" to begin analysis</p>
            </div>
          </div>
        )}
        {}
        <div className="absolute top-4 right-4">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
            isStreamActive ? 'bg-red-500' : 'bg-gray-500'
          } bg-opacity-80 text-white text-sm`}>
            <div className={`w-2 h-2 rounded-full ${
              isStreamActive ? 'bg-white animate-pulse' : 'bg-gray-300'
            }`} />
            <span>{isStreamActive ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>
      </div>
    </Card>
  );
};
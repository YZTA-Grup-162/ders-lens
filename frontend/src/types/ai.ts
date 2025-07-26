export interface EmotionPrediction {
  neutral: number;
  happiness: number;
  surprise: number;
  sadness: number;
  anger: number;
  disgust: number;
  fear: number;
  contempt: number;
  timestamp: number;
  confidence: number;
}
export interface AttentionData {
  score: number;
  level: 'düşük' | 'orta' | 'yüksek';
  headPose: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  timestamp: number;
}
export interface EngagementData {
  overall: number;
  factors: {
    eyeContact: number;
    facialMovement: number;
    headPosition: number;
    screenInteraction: number;
  };
  status: 'pasif' | 'aktif' | 'çok aktif';
  timestamp: number;
}
export interface GazeData {
  x: number;
  y: number;
  fixationDuration: number;
  screenQuadrant: 'üst-sol' | 'üst-sağ' | 'alt-sol' | 'alt-sağ' | 'merkez';
  timestamp: number;
}
export interface StudentAnalysis {
  id: string;
  name?: string;
  emotion: EmotionPrediction;
  attention: AttentionData;
  engagement: EngagementData;
  gaze: GazeData;
  faceDetected: boolean;
  boundingBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}
export interface ClassroomMetrics {
  totalStudents: number;
  activeStudents: number;
  averageAttention: number;
  averageEngagement: number;
  dominantEmotion: keyof EmotionPrediction;
  alertsCount: number;
  sessionDuration: number;
}
export interface ModelPerformance {
  emotionAccuracy: number;
  attentionAccuracy: number;
  engagementAccuracy: number;
  processingSpeed: number; 
  frameRate: number; 
  modelVersion: string;
}
export interface APIResponse<T> {
  success: boolean;
  data: T;
  error?: string;
  timestamp: number;
}
/**
 * Core TypeScript type definitions for DersLens
 */

// User and Role Types
export type UserRole = 'teacher' | 'student' | 'admin' | 'demo';

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  avatar?: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  language: 'tr' | 'en';
  theme: 'dark' | 'light';
  reducedMotion: boolean;
  fontSize: 'small' | 'medium' | 'large';
  notifications: NotificationSettings;
}

export interface NotificationSettings {
  sound: boolean;
  visual: boolean;
  attention: boolean;
  participation: boolean;
  sentiment: boolean;
}

// Analytics Types
export type EmotionType = 
  | 'happy' 
  | 'neutral' 
  | 'focused' 
  | 'confused' 
  | 'frustrated' 
  | 'surprised' 
  | 'sad';

export type AttentionLevel = 'very-low' | 'low' | 'medium' | 'high' | 'very-high';

export interface AttentionDataPoint {
  timestamp: number;
  value: number; // 0-100
  level: AttentionLevel;
  studentId?: string;
}

export interface ParticipationMetrics {
  cameraOn: number; // percentage
  micActive: number; // percentage
  chatActivity: number; // messages per minute
  handRaises: number; // count
  speakingTime: number; // seconds
  totalStudents: number;
}

export interface EmotionMetrics {
  emotions: Record<EmotionType, number>;
  dominant: EmotionType;
  trend: 'improving' | 'declining' | 'stable';
  confidence: number; // 0-100
}

export interface StudentAnalytics {
  id: string;
  name: string;
  attention: AttentionDataPoint[];
  participation: ParticipationMetrics;
  emotions: EmotionMetrics;
  alerts: Alert[];
  badges: Badge[];
}

export interface ClassAnalytics {
  id: string;
  name: string;
  startTime: number;
  duration: number;
  studentCount: number;
  overallAttention: AttentionDataPoint[];
  participationSummary: ParticipationMetrics;
  emotionSummary: EmotionMetrics;
  students: StudentAnalytics[];
  alerts: Alert[];
  aiSuggestions: AISuggestion[];
}

// Alert and Suggestion Types
export type AlertType = 'attention' | 'participation' | 'emotion' | 'technical';
export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface Alert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  description: string;
  timestamp: number;
  studentId?: string;
  acknowledged: boolean;
  actions?: AlertAction[];
}

export interface AlertAction {
  id: string;
  label: string;
  action: () => void;
  variant: 'primary' | 'secondary' | 'danger';
}

export interface AISuggestion {
  id: string;
  category: 'engagement' | 'pedagogy' | 'technical';
  title: string;
  description: string;
  confidence: number; // 0-100
  impact: 'low' | 'medium' | 'high';
  timeToImplement: number; // minutes
  actions: SuggestionAction[];
}

export interface SuggestionAction {
  id: string;
  label: string;
  description: string;
  action: () => void;
}

// Badge System
export type BadgeType = 
  | 'active-participant' 
  | 'focused' 
  | 'helpful' 
  | 'engaged' 
  | 'consistent';

export interface Badge {
  id: string;
  type: BadgeType;
  title: string;
  description: string;
  icon: string;
  earnedAt: number;
  progress?: number; // 0-100 for badges in progress
}

// Demo Engine Types
export interface DemoScenario {
  id: string;
  name: string;
  description: string;
  duration: number; // seconds
  events: DemoEvent[];
  metadata: DemoMetadata;
}

export interface DemoEvent {
  timestamp: number; // seconds from start
  type: 'attention' | 'participation' | 'emotion' | 'student-action' | 'alert';
  data: any;
  description: string;
}

export interface DemoMetadata {
  studentCount: number;
  className: string;
  subject: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
}

export interface DemoState {
  isPlaying: boolean;
  currentTime: number;
  playbackSpeed: number;
  scenario: DemoScenario | null;
  mode: 'live' | 'replay';
  hotkeysEnabled: boolean;
}

// Chart and Visualization Types
export interface ChartDataPoint {
  x: number | string;
  y: number;
  category?: string;
  metadata?: Record<string, any>;
}

export interface ChartOptions {
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  showLegend?: boolean;
  showGrid?: boolean;
  theme: 'dark' | 'light';
  colors?: string[];
  animations?: boolean;
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: number;
}

export interface LiveSessionData {
  classId: string;
  analytics: ClassAnalytics;
  connectedStudents: number;
  lastUpdate: number;
}

// Component Prop Types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
  'data-testid'?: string;
}

export interface GlassCardProps extends BaseComponentProps {
  neonAccent?: boolean;
  padding?: 'sm' | 'md' | 'lg';
  blur?: 'sm' | 'md' | 'lg';
}

export interface NeonButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'accent';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
}

export interface MetricDisplayProps extends BaseComponentProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  format?: 'number' | 'percentage' | 'time';
  color?: EmotionType | 'primary' | 'accent';
}

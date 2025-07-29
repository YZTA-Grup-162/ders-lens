import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
    Alert,
    AttentionDataPoint,
    ClassAnalytics,
    DemoScenario,
    DemoState
} from '../types';

interface DemoStore extends DemoState {
  // Demo data
  scenario: DemoScenario | null;
  currentAnalytics: ClassAnalytics | null;
  intervalId?: NodeJS.Timeout | null;
  
  // Actions
  setScenario: (scenario: DemoScenario) => void;
  play: () => void;
  pause: () => void;
  reset: () => void;
  setCurrentTime: (time: number) => void;
  setPlaybackSpeed: (speed: number) => void;
  toggleMode: () => void;
  toggleHotkeys: () => void;
  jumpToEvent: (eventIndex: number) => void;
  cleanupInterval: () => void;
  
  // Demo event triggers (hotkeys)
  triggerAttentionDip: () => void;
  triggerParticipationBoost: () => void;
  triggerEmotionShift: (emotion: string) => void;
  triggerAlert: (type: string) => void;
  
  // Analytics updates
  updateAnalytics: (analytics: Partial<ClassAnalytics>) => void;
}

// Demo scenario data
const createDemoScenario = (): DemoScenario => ({
  id: 'demo-standard',
  name: 'Standard Classroom Session',
  description: 'A typical 15-minute classroom session with natural engagement patterns',
  duration: 900, // 15 minutes in seconds
  events: [
    {
      timestamp: 0,
      type: 'student-action',
      data: { action: 'join', count: 25 },
      description: 'Students join the class'
    },
    {
      timestamp: 120,
      type: 'attention',
      data: { level: 85, trend: 'stable' },
      description: 'High initial attention'
    },
    {
      timestamp: 180,
      type: 'participation',
      data: { cameraOn: 92, micActive: 15 },
      description: 'Good camera participation'
    },
    {
      timestamp: 300,
      type: 'attention',
      data: { level: 72, trend: 'declining' },
      description: 'Natural attention decline'
    },
    {
      timestamp: 420,
      type: 'emotion',
      data: { dominant: 'confused', distribution: { confused: 40, neutral: 35, focused: 25 } },
      description: 'Students show confusion'
    },
    {
      timestamp: 480,
      type: 'alert',
      data: { type: 'attention', severity: 'medium', message: 'Class attention below threshold' },
      description: 'Low attention alert'
    },
    {
      timestamp: 540,
      type: 'participation',
      data: { handRaises: 8, chatActivity: 12 },
      description: 'Interactive Q&A session'
    },
    {
      timestamp: 600,
      type: 'attention',
      data: { level: 88, trend: 'improving' },
      description: 'Attention recovers with interaction'
    },
    {
      timestamp: 720,
      type: 'emotion',
      data: { dominant: 'focused', distribution: { focused: 60, happy: 25, neutral: 15 } },
      description: 'Students become more engaged'
    },
    {
      timestamp: 840,
      type: 'attention',
      data: { level: 75, trend: 'stable' },
      description: 'Sustained attention near end'
    }
  ],
  metadata: {
    studentCount: 25,
    className: 'Advanced Mathematics',
    subject: 'Calculus',
    difficulty: 'intermediate',
    tags: ['mathematics', 'calculus', 'problem-solving']
  }
});

const initialAnalytics: ClassAnalytics = {
  id: 'demo-class-1',
  name: 'Advanced Mathematics - Calculus',
  startTime: Date.now(),
  duration: 0,
  studentCount: 25,
  overallAttention: [],
  participationSummary: {
    cameraOn: 88,
    micActive: 12,
    chatActivity: 5,
    handRaises: 2,
    speakingTime: 120,
    totalStudents: 25
  },
  emotionSummary: {
    emotions: {
      happy: 20,
      neutral: 45,
      focused: 25,
      confused: 8,
      frustrated: 2,
      surprised: 0,
      sad: 0
    },
    dominant: 'neutral',
    trend: 'stable',
    confidence: 78
  },
  students: [],
  alerts: [],
  aiSuggestions: []
};

export const useDemoStore = create<DemoStore>()(
  devtools(
    (set, get) => ({
      // State
      isPlaying: false,
      currentTime: 0,
      playbackSpeed: 1,
      scenario: createDemoScenario(),
      mode: 'live',
      hotkeysEnabled: true,
      currentAnalytics: initialAnalytics,

      // Actions
      setScenario: (scenario: DemoScenario) =>
        set({ scenario }, false, 'setScenario'),

      play: () => {
        set({ isPlaying: true }, false, 'play');
        
        // Cleanup any previous interval
        const prevIntervalId = get().intervalId;
        if (prevIntervalId) {
          clearInterval(prevIntervalId);
        }

        // Start demo playback
        const interval = setInterval(() => {
          const currentState = get();
          if (!currentState.isPlaying || !currentState.scenario) {
            clearInterval(interval);
            set({ intervalId: null }, false, 'cleanupInterval');
            return;
          }

          const newTime = currentState.currentTime + (1 * currentState.playbackSpeed);
          if (newTime >= currentState.scenario.duration) {
            set({ isPlaying: false, currentTime: currentState.scenario.duration }, false, 'demo-complete');
            clearInterval(interval);
            set({ intervalId: null }, false, 'cleanupInterval');
            return;
          }

          set({ currentTime: newTime }, false, 'demo-tick');
          
          // Process events at current time
          currentState.scenario.events.forEach(event => {
            if (Math.abs(event.timestamp - newTime) < 1) {
              // Trigger event
              // Basic event processing for demo: log event to console
              console.log('Demo event triggered:', event);
            }
          });
        }, 1000 / get().playbackSpeed);

        set({ intervalId: interval }, false, 'setIntervalId');
      },

      pause: () => {
        const intervalId = get().intervalId;
        if (intervalId) {
          clearInterval(intervalId);
          set({ intervalId: null }, false, 'cleanupInterval');
        }
        set({ isPlaying: false }, false, 'pause');
      },

      reset: () => {
        const intervalId = get().intervalId;
        if (intervalId) {
          clearInterval(intervalId);
          set({ intervalId: null }, false, 'cleanupInterval');
        }
        set({ 
          isPlaying: false, 
          currentTime: 0,
          currentAnalytics: initialAnalytics 
        }, false, 'reset');
      },

      setCurrentTime: (time: number) => {
        const state = get();
        const clampedTime = Math.max(0, Math.min(time, state.scenario?.duration || 0));
        set({ currentTime: clampedTime }, false, 'setCurrentTime');
      },

      setPlaybackSpeed: (speed: number) =>
        set({ playbackSpeed: speed }, false, 'setPlaybackSpeed'),

      toggleMode: () =>
        set((state) => ({ 
          mode: state.mode === 'live' ? 'replay' : 'live' 
        }), false, 'toggleMode'),

      toggleHotkeys: () =>
        set((state) => ({ 
          hotkeysEnabled: !state.hotkeysEnabled 
        }), false, 'toggleHotkeys'),

      jumpToEvent: (eventIndex: number) => {
        const state = get();
        if (state.scenario && state.scenario.events[eventIndex]) {
          const event = state.scenario.events[eventIndex];
          set({ currentTime: event.timestamp }, false, 'jumpToEvent');
        }
      },

      // Hotkey triggers
      triggerAttentionDip: () => {
        const state = get();
        if (state.currentAnalytics) {
          const newAttention: AttentionDataPoint = {
            timestamp: Date.now(),
            value: Math.max(20, Math.random() * 40 + 20), // 20-60 range
            level: 'low'
          };
          
          set({
            currentAnalytics: {
              ...state.currentAnalytics,
              overallAttention: [...state.currentAnalytics.overallAttention, newAttention]
            }
          }, false, 'triggerAttentionDip');
        }
      },

      triggerParticipationBoost: () => {
        const state = get();
        if (state.currentAnalytics) {
          set({
            currentAnalytics: {
              ...state.currentAnalytics,
              participationSummary: {
                ...state.currentAnalytics.participationSummary,
                handRaises: state.currentAnalytics.participationSummary.handRaises + Math.floor(Math.random() * 5) + 2,
                chatActivity: state.currentAnalytics.participationSummary.chatActivity + Math.floor(Math.random() * 8) + 3
              }
            }
          }, false, 'triggerParticipationBoost');
        }
      },

      triggerEmotionShift: (emotion: string) => {
        const state = get();
        if (state.currentAnalytics) {
          const newEmotions = { ...state.currentAnalytics.emotionSummary.emotions };
          // Redistribute emotions with the target emotion becoming dominant
          Object.keys(newEmotions).forEach(key => {
            newEmotions[key as keyof typeof newEmotions] = key === emotion ? 
              Math.random() * 30 + 40 : // 40-70% for target emotion
              Math.random() * 20; // 0-20% for others
          });

          set({
            currentAnalytics: {
              ...state.currentAnalytics,
              emotionSummary: {
                ...state.currentAnalytics.emotionSummary,
                emotions: newEmotions,
                dominant: emotion as any
              }
            }
          }, false, 'triggerEmotionShift');
        }
      },

      triggerAlert: (type: string) => {
        const state = get();
        if (state.currentAnalytics) {
          const newAlert: Alert = {
            id: `alert-${Date.now()}`,
            type: type as any,
            severity: 'medium',
            title: `${type.charAt(0).toUpperCase() + type.slice(1)} Alert`,
            description: `Demo ${type} alert triggered`,
            timestamp: Date.now(),
            acknowledged: false
          };

          set({
            currentAnalytics: {
              ...state.currentAnalytics,
              alerts: [...state.currentAnalytics.alerts, newAlert]
            }
          }, false, 'triggerAlert');
        }
      },

      updateAnalytics: (analytics: Partial<ClassAnalytics>) => {
        const state = get();
        if (state.currentAnalytics) {
          set({
            currentAnalytics: {
              ...state.currentAnalytics,
              ...analytics
            }
          }, false, 'updateAnalytics');
        }
      },

      cleanupInterval: () => {
        const intervalId = get().intervalId;
        if (intervalId) {
          clearInterval(intervalId);
          set({ intervalId: null }, false, 'cleanupInterval');
        }
      }
    }),
    {
      name: 'demo-store'
    }
  )
);

// Selectors
export const useDemoControls = () => {
  const { 
    isPlaying, 
    currentTime, 
    playbackSpeed, 
    play, 
    pause, 
    reset, 
    setCurrentTime, 
    setPlaybackSpeed 
  } = useDemoStore();
  
  return {
    isPlaying,
    currentTime,
    playbackSpeed,
    play,
    pause,
    reset,
    setCurrentTime,
    setPlaybackSpeed
  };
};

export const useDemoAnalytics = () => {
  const currentAnalytics = useDemoStore((state) => state.currentAnalytics);
  return currentAnalytics;
};

export const useDemoHotkeys = () => {
  const {
    hotkeysEnabled,
    triggerAttentionDip,
    triggerParticipationBoost,
    triggerEmotionShift,
    triggerAlert
  } = useDemoStore();

  return {
    hotkeysEnabled,
    triggerAttentionDip,
    triggerParticipationBoost,
    triggerEmotionShift,
    triggerAlert
  };
};

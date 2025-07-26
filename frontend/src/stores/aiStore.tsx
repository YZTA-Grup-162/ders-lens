import React, { createContext, useContext, useEffect, useReducer } from 'react';
import { ClassroomMetrics, ModelPerformance, StudentAnalysis } from '../types/ai';
interface AIState {
  isAnalyzing: boolean;
  students: StudentAnalysis[];
  classroomMetrics: ClassroomMetrics;
  modelPerformance: ModelPerformance;
  cameraStream: MediaStream | null;
  websocket: WebSocket | null;
  errors: string[];
  lastUpdate: number;
}
interface AIAction {
  type: 'START_ANALYSIS' | 'STOP_ANALYSIS' | 'UPDATE_STUDENTS' | 'UPDATE_METRICS' | 'SET_CAMERA' | 'SET_WEBSOCKET' | 'ADD_ERROR' | 'CLEAR_ERRORS';
  payload?: any;
}
const initialState: AIState = {
  isAnalyzing: false,
  students: [],
  classroomMetrics: {
    totalStudents: 0,
    activeStudents: 0,
    averageAttention: 0,
    averageEngagement: 0,
    dominantEmotion: 'neutral',
    alertsCount: 0,
    sessionDuration: 0,
  },
  modelPerformance: {
    emotionAccuracy: 94.2,
    attentionAccuracy: 91.8,
    engagementAccuracy: 89.6,
    processingSpeed: 23,
    frameRate: 30,
    modelVersion: 'FER2013+',
  },
  cameraStream: null,
  websocket: null,
  errors: [],
  lastUpdate: Date.now(),
};
function aiReducer(state: AIState, action: AIAction): AIState {
  switch (action.type) {
    case 'START_ANALYSIS':
      return { ...state, isAnalyzing: true, errors: [] };
    case 'STOP_ANALYSIS':
      return { ...state, isAnalyzing: false };
    case 'UPDATE_STUDENTS':
      return { 
        ...state, 
        students: action.payload,
        lastUpdate: Date.now(),
      };
    case 'UPDATE_METRICS':
      return { 
        ...state, 
        classroomMetrics: action.payload,
        lastUpdate: Date.now(),
      };
    case 'SET_CAMERA':
      return { ...state, cameraStream: action.payload };
    case 'SET_WEBSOCKET':
      return { ...state, websocket: action.payload };
    case 'ADD_ERROR':
      return { 
        ...state, 
        errors: [...state.errors, action.payload].slice(-5) 
      };
    case 'CLEAR_ERRORS':
      return { ...state, errors: [] };
    default:
      return state;
  }
}
const AIContext = createContext<{
  state: AIState;
  dispatch: React.Dispatch<AIAction>;
  actions: {
    startAnalysis: () => Promise<void>;
    stopAnalysis: () => void;
    initializeCamera: () => Promise<void>;
    connectWebSocket: () => void;
    sendFrame: (frame: string) => void;
  };
} | null>(null);
export function AIProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(aiReducer, initialState);
  const startAnalysis = async () => {
    try {
      dispatch({ type: 'START_ANALYSIS' });
      await initializeCamera();
      connectWebSocket();
    } catch (error) {
      dispatch({ type: 'ADD_ERROR', payload: (error as Error).message });
    }
  };
  const stopAnalysis = () => {
    dispatch({ type: 'STOP_ANALYSIS' });
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(track => track.stop());
      dispatch({ type: 'SET_CAMERA', payload: null });
    }
    if (state.websocket) {
      state.websocket.close();
      dispatch({ type: 'SET_WEBSOCKET', payload: null });
    }
  };
  const initializeCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480, 
          frameRate: 30,
          facingMode: 'user'
        }
      });
      dispatch({ type: 'SET_CAMERA', payload: stream });
    } catch (error) {
      throw new Error('Kamera erişimi başarısız. Lütfen kamera izinlerini kontrol edin.');
    }
  };
  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/') // Added a forward slash at the end of the URL
    ws.onopen = () => {
      console.log('WebSocket bağlantısı kuruldu');
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.students) {
          dispatch({ type: 'UPDATE_STUDENTS', payload: data.students });
        }
        if (data.metrics) {
          dispatch({ type: 'UPDATE_METRICS', payload: data.metrics });
        }
      } catch (error) {
        dispatch({ type: 'ADD_ERROR', payload: 'WebSocket veri işleme hatası' });
      }
    };
    ws.onerror = () => {
      dispatch({ type: 'ADD_ERROR', payload: 'WebSocket bağlantı hatası' });
    };
    ws.onclose = () => {
      console.log('WebSocket bağlantısı kapatıldı');
      dispatch({ type: 'SET_WEBSOCKET', payload: null });
    };
    dispatch({ type: 'SET_WEBSOCKET', payload: ws });
  };
  const sendFrame = (frame: string) => {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
      state.websocket.send(JSON.stringify({
        type: 'frame',
        data: frame,
        timestamp: Date.now()
      }));
    }
  };
  useEffect(() => {
    return () => {
      stopAnalysis();
    };
  }, []);
  const actions = {
    startAnalysis,
    stopAnalysis,
    initializeCamera,
    connectWebSocket,
    sendFrame,
  };
  return (
    <AIContext.Provider value={{ state, dispatch, actions }}>
      {children}
    </AIContext.Provider>
  );
}
export function useAI() {
  const context = useContext(AIContext);
  if (!context) {
    throw new Error('useAI must be used within AIProvider');
  }
  return context;
}
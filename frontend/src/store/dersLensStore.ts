import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
interface AnalysisResult {
  attention: number;
  engagement: number;
  emotion: string;
  emotionConfidence: number;
  gazeDirection: string;
  faceDetected: boolean;
  timestamp: number;
}
interface StudentData {
  id: string;
  name: string;
  attention: number;
  engagement: number;
  emotion: string;
  gazeDirection: string;
  isActive: boolean;
}
interface AppState {
  isRecording: boolean;
  stream: MediaStream | null;
  isAnalyzing: boolean;
  currentAnalysis: AnalysisResult | null;
  analysisHistory: AnalysisResult[];
  students: StudentData[];
  overallStats: {
    averageAttention: number;
    averageEngagement: number;
    activeStudents: number;
    totalStudents: number;
  };
  setRecording: (recording: boolean) => void;
  setStream: (stream: MediaStream | null) => void;
  setAnalyzing: (analyzing: boolean) => void;
  setCurrentAnalysis: (analysis: AnalysisResult | null) => void;
  addAnalysisToHistory: (analysis: AnalysisResult) => void;
  updateStudents: (students: StudentData[]) => void;
  updateOverallStats: () => void;
}
export const useAppStore = create<AppState>()(
  devtools(
    (set, get) => ({
      isRecording: false,
      stream: null,
      isAnalyzing: false,
      currentAnalysis: null,
      analysisHistory: [],
      students: [],
      overallStats: {
        averageAttention: 0,
        averageEngagement: 0,
        activeStudents: 0,
        totalStudents: 0,
      },
      setRecording: (recording) => set({ isRecording: recording }),
      setStream: (stream) => set({ stream }),
      setAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
      setCurrentAnalysis: (analysis) => {
        set({ currentAnalysis: analysis });
        if (analysis) {
          get().addAnalysisToHistory(analysis);
        }
      },
      addAnalysisToHistory: (analysis) =>
        set((state) => ({
          analysisHistory: [...state.analysisHistory.slice(-49), analysis], 
        })),
      updateStudents: (students) => {
        set({ students });
        get().updateOverallStats();
      },
      updateOverallStats: () => {
        const { students } = get();
        const activeStudents = students.filter((s) => s.isActive);
        const avgAttention = activeStudents.length > 0
          ? activeStudents.reduce((sum, s) => sum + s.attention, 0) / activeStudents.length
          : 0;
        const avgEngagement = activeStudents.length > 0
          ? activeStudents.reduce((sum, s) => sum + s.engagement, 0) / activeStudents.length
          : 0;
        set({
          overallStats: {
            averageAttention: Math.round(avgAttention),
            averageEngagement: Math.round(avgEngagement),
            activeStudents: activeStudents.length,
            totalStudents: students.length,
          },
        });
      },
    }),
    {
      name: 'ders-lens-store',
    }
  )
);
export type { AnalysisResult, StudentData };
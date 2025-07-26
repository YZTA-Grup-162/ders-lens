import { create } from 'zustand'
import { devtools, subscribeWithSelector } from 'zustand/middleware'
export interface AttentionData {
  score: number
  state: 'attentive' | 'inattentive'
  confidence: number
  timestamp: string
}
export interface EngagementData {
  level: 0 | 1 | 2 | 3 
  probabilities: Record<string, number>
}
export interface EmotionData {
  class: 0 | 1 | 2 | 3 
  probabilities: Record<string, number>
}
export interface PredictionResult {
  frameId: string
  studentId?: string
  sessionId?: string
  attention: AttentionData
  engagement: EngagementData
  emotion: EmotionData
  faceDetected: boolean
  phoneUsage: boolean
  processingTimeMs: number
  timestamp: string
}
export interface StudentCard {
  studentId: string
  name?: string
  currentAttention: number
  attentionTrend: number[]
  currentEngagement: 0 | 1 | 2 | 3
  currentEmotion: 0 | 1 | 2 | 3
  lastUpdated: string
  sessionDuration: number
}
export interface ClassroomDashboard {
  classId: string
  timestamp: string
  activeStudents: number
  students: StudentCard[]
  classAverageAttention: number
  alerts: string[]
}
export interface SessionSummary {
  studentId: string
  sessionId: string
  startTime: string
  endTime: string
  durationMinutes: number
  averageAttention: number
  attentionDistribution: Record<string, number>
  engagementDistribution: Record<string, number>
  emotionDistribution: Record<string, number>
  recommendations: string[]
}
interface StudentState {
  isConnected: boolean
  studentId: string | null
  sessionId: string | null
  currentPrediction: PredictionResult | null
  attentionHistory: number[]
  feedbackColor: 'green' | 'yellow' | 'red'
  feedbackMessage: string
  webcamEnabled: boolean
  consentGiven: boolean
}
interface TeacherState {
  isConnected: boolean
  teacherId: string | null
  classId: string | null
  dashboard: ClassroomDashboard | null
  selectedStudentId: string | null
  sortBy: 'name' | 'attention' | 'lastUpdated'
  sortOrder: 'asc' | 'desc'
}
interface AppState {
  theme: 'light' | 'dark'
  sidebarOpen: boolean
  currentView: 'student' | 'teacher' | 'analytics'
  wsConnected: boolean
  reconnectAttempts: number
  student: StudentState
  teacher: TeacherState
  sessionSummaries: SessionSummary[]
  errors: string[]
}
interface AppActions {
  setTheme: (theme: 'light' | 'dark') => void
  toggleSidebar: () => void
  setCurrentView: (view: 'student' | 'teacher' | 'analytics') => void
  setWsConnected: (connected: boolean) => void
  incrementReconnectAttempts: () => void
  resetReconnectAttempts: () => void
  connectStudent: (studentId: string, sessionId: string) => void
  disconnectStudent: () => void
  updatePrediction: (prediction: PredictionResult) => void
  setFeedback: (color: 'green' | 'yellow' | 'red', message: string) => void
  setWebcamEnabled: (enabled: boolean) => void
  setConsentGiven: (given: boolean) => void
  connectTeacher: (teacherId: string, classId: string) => void
  disconnectTeacher: () => void
  updateDashboard: (dashboard: ClassroomDashboard) => void
  setSelectedStudent: (studentId: string | null) => void
  setSorting: (sortBy: 'name' | 'attention' | 'lastUpdated', order: 'asc' | 'desc') => void
  addSessionSummary: (summary: SessionSummary) => void
  clearSessionSummaries: () => void
  addError: (error: string) => void
  removeError: (index: number) => void
  clearErrors: () => void
}
type Store = AppState & AppActions
export const useAppStore = create<Store>()(
  devtools(
    subscribeWithSelector(
      (set, get) => ({
        theme: 'light',
        sidebarOpen: false,
        currentView: 'student',
        wsConnected: false,
        reconnectAttempts: 0,
        student: {
          isConnected: false,
          studentId: null,
          sessionId: null,
          currentPrediction: null,
          attentionHistory: [],
          feedbackColor: 'green',
          feedbackMessage: 'Ready to start',
          webcamEnabled: false,
          consentGiven: false,
        },
        teacher: {
          isConnected: false,
          teacherId: null,
          classId: null,
          dashboard: null,
          selectedStudentId: null,
          sortBy: 'attention',
          sortOrder: 'desc',
        },
        sessionSummaries: [],
        errors: [],
        setTheme: (theme) => set({ theme }),
        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
        setCurrentView: (currentView) => set({ currentView }),
        setWsConnected: (wsConnected) => set({ wsConnected }),
        incrementReconnectAttempts: () => 
          set((state) => ({ reconnectAttempts: state.reconnectAttempts + 1 })),
        resetReconnectAttempts: () => set({ reconnectAttempts: 0 }),
        connectStudent: (studentId, sessionId) =>
          set((state) => ({
            student: {
              ...state.student,
              isConnected: true,
              studentId,
              sessionId,
              attentionHistory: [],
            },
          })),
        disconnectStudent: () =>
          set((state) => ({
            student: {
              ...state.student,
              isConnected: false,
              studentId: null,
              sessionId: null,
              currentPrediction: null,
              attentionHistory: [],
            },
          })),
        updatePrediction: (prediction) =>
          set((state) => {
            const newHistory = [...state.student.attentionHistory, prediction.attention.score]
            if (newHistory.length > 300) {
              newHistory.shift()
            }
            return {
              student: {
                ...state.student,
                currentPrediction: prediction,
                attentionHistory: newHistory,
              },
            }
          }),
        setFeedback: (feedbackColor, feedbackMessage) =>
          set((state) => ({
            student: {
              ...state.student,
              feedbackColor,
              feedbackMessage,
            },
          })),
        setWebcamEnabled: (webcamEnabled) =>
          set((state) => ({
            student: {
              ...state.student,
              webcamEnabled,
            },
          })),
        setConsentGiven: (consentGiven) =>
          set((state) => ({
            student: {
              ...state.student,
              consentGiven,
            },
          })),
        connectTeacher: (teacherId, classId) =>
          set((state) => ({
            teacher: {
              ...state.teacher,
              isConnected: true,
              teacherId,
              classId,
            },
          })),
        disconnectTeacher: () =>
          set((state) => ({
            teacher: {
              ...state.teacher,
              isConnected: false,
              teacherId: null,
              classId: null,
              dashboard: null,
            },
          })),
        updateDashboard: (dashboard) =>
          set((state) => ({
            teacher: {
              ...state.teacher,
              dashboard,
            },
          })),
        setSelectedStudent: (selectedStudentId) =>
          set((state) => ({
            teacher: {
              ...state.teacher,
              selectedStudentId,
            },
          })),
        setSorting: (sortBy, sortOrder) =>
          set((state) => ({
            teacher: {
              ...state.teacher,
              sortBy,
              sortOrder,
            },
          })),
        addSessionSummary: (summary) =>
          set((state) => ({
            sessionSummaries: [...state.sessionSummaries, summary],
          })),
        clearSessionSummaries: () => set({ sessionSummaries: [] }),
        addError: (error) =>
          set((state) => ({
            errors: [...state.errors, error],
          })),
        removeError: (index) =>
          set((state) => ({
            errors: state.errors.filter((_, i) => i !== index),
          })),
        clearErrors: () => set({ errors: [] }),
      })
    ),
    {
      name: 'attention-pulse-store',
    }
  )
)
export const useStudentState = () => useAppStore((state) => state.student)
export const useTeacherState = () => useAppStore((state) => state.teacher)
export const useConnectionState = () => useAppStore((state) => ({
  wsConnected: state.wsConnected,
  reconnectAttempts: state.reconnectAttempts,
}))
export const useUIState = () => useAppStore((state) => ({
  theme: state.theme,
  sidebarOpen: state.sidebarOpen,
  currentView: state.currentView,
}))
export const useSortedStudents = () => {
  return useAppStore((state) => {
    const { dashboard, sortBy, sortOrder } = state.teacher
    if (!dashboard?.students) return []
    const sorted = [...dashboard.students].sort((a, b) => {
      let aVal: any = a[sortBy as keyof StudentCard]
      let bVal: any = b[sortBy as keyof StudentCard]
      if (sortBy === 'attention') {
        aVal = a.currentAttention
        bVal = b.currentAttention
      } else if (sortBy === 'lastUpdated') {
        aVal = new Date(a.lastUpdated).getTime()
        bVal = new Date(b.lastUpdated).getTime()
      } else if (sortBy === 'name') {
        aVal = a.name || `Student ${a.studentId}`
        bVal = b.name || `Student ${b.studentId}`
      }
      if (sortOrder === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0
      }
    })
    return sorted
  })
}
export const useAttentionStats = () => {
  return useAppStore((state) => {
    const { attentionHistory } = state.student
    if (attentionHistory.length === 0) {
      return {
        average: 0,
        min: 0,
        max: 0,
        latest: 0,
        trend: 'stable' as 'up' | 'down' | 'stable',
      }
    }
    const latest = attentionHistory[attentionHistory.length - 1]
    const average = attentionHistory.reduce((sum, val) => sum + val, 0) / attentionHistory.length
    const min = Math.min(...attentionHistory)
    const max = Math.max(...attentionHistory)
    let trend: 'up' | 'down' | 'stable' = 'stable'
    if (attentionHistory.length >= 20) {
      const recent = attentionHistory.slice(-10)
      const previous = attentionHistory.slice(-20, -10)
      const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length
      const previousAvg = previous.reduce((sum, val) => sum + val, 0) / previous.length
      const diff = recentAvg - previousAvg
      if (diff > 0.05) trend = 'up'
      else if (diff < -0.05) trend = 'down'
    }
    return {
      average: Math.round(average * 100) / 100,
      min: Math.round(min * 100) / 100,
      max: Math.round(max * 100) / 100,
      latest: Math.round(latest * 100) / 100,
      trend,
    }
  })
}
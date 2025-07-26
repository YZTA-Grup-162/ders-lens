import type { ClassroomDashboard, PredictionResult } from '@/store/appStore'
import { useAppStore } from '@/store/appStore'
import React from 'react'
import { io, Socket } from 'socket.io-client'
interface WebSocketConfig {
  url: string
  reconnectAttempts: number
  reconnectDelay: number
  timeout: number
}
interface StudentConnectData {
  studentId: string
  sessionId: string
}
interface TeacherConnectData {
  teacherId: string
  classId: string
}
interface FrameData {
  data: string
  width: number
  height: number
  timestamp: number
}
class WebSocketService {
  private socket: Socket | null = null
  private config: WebSocketConfig
  private reconnectTimer: number | null = null
  private isConnecting = false
  private studentUpdateHandlers: Array<(data: any) => void> = []
  private classroomMetricsHandlers: Array<(metrics: any) => void> = []
  private alertHandlers: Array<(alert: any) => void> = []

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = {
      url: 'ws://localhost:8000/ws',
      reconnectAttempts: 5,
      reconnectDelay: 2000,
      timeout: 10000,
      ...config,
    }
  }
  private initializeSocket(path: string): Socket {
    const socket = io(this.config.url + path, {
      transports: ['websocket'],
      timeout: this.config.timeout,
      forceNew: true,
    })
    socket.on('connect', () => {
      console.log('WebSocket connected')
      useAppStore.getState().setWsConnected(true)
      useAppStore.getState().resetReconnectAttempts()
      this.isConnecting = false
    })
    socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
      useAppStore.getState().setWsConnected(false)
      if (reason === 'io server disconnect') {
        return
      }
      this.handleReconnect()
    })
    socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      useAppStore.getState().addError('Connection failed: ' + error.message)
      this.handleReconnect()
    })
    return socket
  }
  private handleReconnect() {
    if (this.isConnecting || this.reconnectTimer) {
      return
    }
    const { reconnectAttempts } = useAppStore.getState()
    if (reconnectAttempts >= this.config.reconnectAttempts) {
      console.error('Max reconnection attempts reached')
      useAppStore.getState().addError('Unable to connect to server')
      return
    }
    console.log(`Reconnecting in ${this.config.reconnectDelay}ms... (attempt ${reconnectAttempts + 1})`)
    useAppStore.getState().incrementReconnectAttempts()
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null
      if (this.socket && !this.socket.connected) {
        this.isConnecting = true
        this.socket.connect()
      }
    }, this.config.reconnectDelay)
  }
  connectAsStudent(data: StudentConnectData): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = this.initializeSocket('/ws/student')
        this.socket.on('connect', () => {
          this.socket!.emit('join_session', data)
          useAppStore.getState().connectStudent(data.studentId, data.sessionId)
          resolve()
        })
        this.socket.on('attention_feedback', (feedback) => {
          const { color, message, attention_score, engagement_level, emotion, timestamp } = feedback.data
          useAppStore.getState().setFeedback(color, message)
          if (attention_score !== undefined) {
            const prediction: PredictionResult = {
              frameId: `frame_${Date.now()}`,
              studentId: data.studentId,
              sessionId: data.sessionId,
              attention: {
                score: attention_score,
                state: attention_score > 0.5 ? 'attentive' : 'inattentive',
                confidence: Math.max(attention_score, 1 - attention_score),
                timestamp,
              },
              engagement: {
                level: engagement_level || 0,
                probabilities: {},
              },
              emotion: {
                class: emotion === 'boredom' ? 0 : emotion === 'confusion' ? 1 : emotion === 'engagement' ? 2 : 3,
                probabilities: {},
              },
              faceDetected: true,
              phoneUsage: false,
              processingTimeMs: 0,
              timestamp,
            }
            useAppStore.getState().updatePrediction(prediction)
          }
        })
        this.socket.on('connect_error', reject)
      } catch (error) {
        reject(error)
      }
    })
  }
  connectAsTeacher(data: TeacherConnectData): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = this.initializeSocket('/ws/teacher')
        this.socket.on('connect', () => {
          this.socket!.emit('join_class', data)
          useAppStore.getState().connectTeacher(data.teacherId, data.classId)
          resolve()
        })
        this.socket.on('dashboard_update', (dashboardData) => {
          const dashboard: ClassroomDashboard = dashboardData.data
          useAppStore.getState().updateDashboard(dashboard)
        })
        this.socket.on('student_update', (data) => {
          this.studentUpdateHandlers.forEach(handler => handler(data))
        })
        this.socket.on('classroom_metrics', (metrics) => {
          this.classroomMetricsHandlers.forEach(handler => handler(metrics))
        })
        this.socket.on('alert', (alert) => {
          this.alertHandlers.forEach(handler => handler(alert))
        })
        this.socket.on('student_joined', (studentData) => {
          console.log('Student joined:', studentData)
          this.studentUpdateHandlers.forEach(handler => handler({
            type: 'joined',
            ...studentData
          }))
        })
        this.socket.on('student_left', (studentData) => {
          console.log('Student left:', studentData)
          this.studentUpdateHandlers.forEach(handler => handler({
            type: 'left',
            ...studentData
          }))
        })
        this.socket.on('connect_error', reject)
      } catch (error) {
        reject(error)
      }
    })
  }
  sendVideoFrame(frameData: FrameData): void {
    if (!this.socket || !this.socket.connected) {
      console.warn('Cannot send frame: WebSocket not connected')
      return
    }
    this.socket.emit('video_frame', {
      type: 'video_frame',
      data: frameData,
    })
  }
  requestDashboard(): void {
    if (!this.socket || !this.socket.connected) {
      console.warn('Cannot request dashboard: WebSocket not connected')
      return
    }
    this.socket.emit('get_dashboard', {
      type: 'get_dashboard',
    })
  }
  requestClassroomUpdate(): void {
    if (!this.socket || !this.socket.connected) {
      console.warn('Cannot request classroom update: WebSocket not connected')
      return
    }
    this.socket.emit('get_classroom_metrics', {
      type: 'get_classroom_metrics',
      timestamp: Date.now(),
    })
  }
  onStudentUpdate(handler: (data: any) => void): void {
    this.studentUpdateHandlers.push(handler)
  }
  onClassroomMetrics(handler: (metrics: any) => void): void {
    this.classroomMetricsHandlers.push(handler)
  }
  onAlert(handler: (alert: any) => void): void {
    this.alertHandlers.push(handler)
  }
  offStudentUpdate(handler: (data: any) => void): void {
    const index = this.studentUpdateHandlers.indexOf(handler)
    if (index > -1) {
      this.studentUpdateHandlers.splice(index, 1)
    }
  }
  offClassroomMetrics(handler: (metrics: any) => void): void {
    const index = this.classroomMetricsHandlers.indexOf(handler)
    if (index > -1) {
      this.classroomMetricsHandlers.splice(index, 1)
    }
  }
  offAlert(handler: (alert: any) => void): void {
    const index = this.alertHandlers.indexOf(handler)
    if (index > -1) {
      this.alertHandlers.splice(index, 1)
    }
  }
  ping(): void {
    if (!this.socket || !this.socket.connected) {
      return
    }
    this.socket.emit('ping', {
      type: 'ping',
      timestamp: Date.now(),
    })
  }
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    this.studentUpdateHandlers = []
    this.classroomMetricsHandlers = []
    this.alertHandlers = []
    useAppStore.getState().setWsConnected(false)
    useAppStore.getState().disconnectStudent()
    useAppStore.getState().disconnectTeacher()
  }
  isConnected(): boolean {
    return this.socket?.connected || false
  }
  getSocket(): Socket | null {
    return this.socket
  }
}
export const webSocketService = new WebSocketService()
export const useWebSocket = () => {
  const { wsConnected, reconnectAttempts } = useAppStore((state) => ({
    wsConnected: state.wsConnected,
    reconnectAttempts: state.reconnectAttempts,
  }))
  return {
    isConnected: wsConnected,
    reconnectAttempts,
    connect: webSocketService.connectAsStudent.bind(webSocketService),
    connectTeacher: webSocketService.connectAsTeacher.bind(webSocketService),
    disconnect: webSocketService.disconnect.bind(webSocketService),
    sendFrame: webSocketService.sendVideoFrame.bind(webSocketService),
    requestDashboard: webSocketService.requestDashboard.bind(webSocketService),
    ping: webSocketService.ping.bind(webSocketService),
  }
}
export const useFrameStreaming = (enabled: boolean = false) => {
  const { isConnected, sendFrame } = useWebSocket()
  React.useEffect(() => {
    if (!enabled || !isConnected) {
      return
    }
    const handleFrame = (event: CustomEvent<FrameData>) => {
      sendFrame(event.detail)
    }
    window.addEventListener('webcam-frame', handleFrame as EventListener)
    return () => {
      window.removeEventListener('webcam-frame', handleFrame as EventListener)
    }
  }, [enabled, isConnected, sendFrame])
}
export const useWebSocketKeepalive = (interval: number = 30000) => {
  const { isConnected, ping } = useWebSocket()
  React.useEffect(() => {
    if (!isConnected) {
      return
    }
    const pingInterval = setInterval(ping, interval)
    return () => {
      clearInterval(pingInterval)
    }
  }, [isConnected, ping, interval])
}
export const isWebSocketSupported = (): boolean => {
  return typeof WebSocket !== 'undefined' || typeof window.WebSocket !== 'undefined'
}
export const getWebSocketUrl = (path: string = ''): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.hostname
  const port = window.location.hostname === 'localhost' ? ':8000' : ''
  return `${protocol}//${host}${port}${path}`
}
export default webSocketService
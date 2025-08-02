import { AttentionRing } from '@/components/AttentionRing'
import { ConsentModal } from '@/components/ConsentModal'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { useWebcam, useWebcamConsent } from '@/hooks/useWebcam'
import { useFrameStreaming, useWebSocket, useWebSocketKeepalive } from '@/services/websocket'
import { useAppStore } from '@/store/appStore'
import {
    CheckCircleIcon,
    VideoCameraIcon,
    VideoCameraSlashIcon,
    XMarkIcon
} from '@heroicons/react/24/outline'
import { AnimatePresence, motion } from 'framer-motion'
import React, { useEffect, useState } from 'react'
import { toast } from 'react-hot-toast'
interface StudentPageProps {
  studentId?: string
  sessionId?: string
}
export const StudentPage: React.FC<StudentPageProps> = ({ 
  studentId = 'demo-student', 
  sessionId = 'demo-session' 
}) => {
  const [isInitialized, setIsInitialized] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected')
  const { 
    videoRef, 
    canvasRef, 
    isWebcamEnabled, 
    isCapturing, 
    error: webcamError,
    startWebcam,
    stopWebcam,
    startCapture,
    stopCapture
  } = useWebcam({
    width: 320,
    height: 240,
    frameRate: 15
  })
  const { 
    showConsent, 
    consentGiven, 
    requestConsent, 
    giveConsent, 
    denyConsent 
  } = useWebcamConsent()
  const { 
    isConnected, 
    connect, 
    disconnect 
  } = useWebSocket()
  useFrameStreaming(isWebcamEnabled && isConnected)
  useWebSocketKeepalive(30000)
  const { 
    student, 
    wsConnected, 
    errors 
  } = useAppStore((state) => ({
    student: state.student,
    wsConnected: state.wsConnected,
    errors: state.errors
  }))
  useEffect(() => {
    const initializeSession = async () => {
      if (!consentGiven) {
        requestConsent()
        return
      }
      try {
        setConnectionStatus('connecting')
        await connect({ studentId, sessionId })
        await startWebcam()
        setConnectionStatus('connected')
        setIsInitialized(true)
        toast.success('Connected successfully!')
      } catch (error) {
        console.error('Failed to initialize session:', error)
        setConnectionStatus('disconnected')
        toast.error('Failed to connect. Please try again.')
      }
    }
    if (consentGiven && !isInitialized) {
      initializeSession()
    }
  }, [consentGiven, isInitialized, connect, startWebcam, studentId, sessionId, requestConsent])
  useEffect(() => {
    if (isWebcamEnabled && isConnected && !isCapturing) {
      startCapture()
    } else if (!isConnected && isCapturing) {
      stopCapture()
    }
  }, [isWebcamEnabled, isConnected, isCapturing, startCapture, stopCapture])
  useEffect(() => {
    return () => {
      stopWebcam()
      disconnect()
    }
  }, [stopWebcam, disconnect])
  useEffect(() => {
    if (webcamError) {
      toast.error(`Webcam error: ${webcamError}`)
    }
  }, [webcamError])
  useEffect(() => {
    errors.forEach((error, index) => {
      toast.error(error)
      useAppStore.getState().removeError(index)
    })
  }, [errors])
  const handleStartSession = async () => {
    if (!consentGiven) {
      requestConsent()
      return
    }
    try {
      setConnectionStatus('connecting')
      await connect({ studentId, sessionId })
      await startWebcam()
      setConnectionStatus('connected')
      toast.success('Session started!')
    } catch (error) {
      setConnectionStatus('disconnected')
      toast.error('Failed to start session')
    }
  }
  const handleStopSession = () => {
    stopWebcam()
    disconnect()
    setConnectionStatus('disconnected')
    setIsInitialized(false)
    toast.success('Session ended')
  }
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500'
      case 'connecting': return 'text-yellow-500'
      case 'disconnected': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }
  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <CheckCircleIcon className="w-5 h-5" />
      case 'connecting': return <LoadingSpinner className="w-5 h-5" />
      case 'disconnected': return <XMarkIcon className="w-5 h-5" />
      default: return null
    }
  }
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
        <div className="max-w-4xl mx-auto">
          {}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">DersLens</h1>
                <p className="text-gray-600">Student Session</p>
              </div>
              <div className="flex items-center space-x-4">
                <div className={`flex items-center space-x-2 ${getConnectionStatusColor()}`}>
                  {getConnectionStatusIcon()}
                  <span className="capitalize">{connectionStatus}</span>
                </div>
                <div className="text-sm text-gray-500">
                  Student: {studentId}
                </div>
              </div>
            </div>
          </div>
          {}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2">
                  <div className="flex items-center justify-between">
                    <h2 className="text-white font-semibold">Live Video</h2>
                    <div className="flex items-center space-x-2">
                      {isWebcamEnabled ? (
                        <VideoCameraIcon className="w-5 h-5 text-green-400" />
                      ) : (
                        <VideoCameraSlashIcon className="w-5 h-5 text-red-400" />
                      )}
                      <span className="text-sm text-gray-300">
                        {isWebcamEnabled ? 'Camera On' : 'Camera Off'}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="relative aspect-video bg-gray-900 flex items-center justify-center">
                  {}
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    autoPlay
                    muted
                    playsInline
                  />
                  {}
                  <canvas
                    ref={canvasRef}
                    className="hidden"
                    width={320}
                    height={240}
                  />
                  {}
                  <AnimatePresence>
                    {isWebcamEnabled && student.currentPrediction && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="absolute top-4 right-4"
                      >
                        <AttentionRing
                          attentionScore={student.currentPrediction.attention.score}
                          engagementScore={student.currentPrediction.engagement?.level ? (student.currentPrediction.engagement.level * 25) : 0}
                          size={80}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                  {}
                  {!isWebcamEnabled && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center text-gray-400">
                        <VideoCameraSlashIcon className="w-16 h-16 mx-auto mb-4" />
                        <p>Camera is not active</p>
                      </div>
                    </div>
                  )}
                </div>
                {}
                <div className="p-4 bg-gray-50 flex justify-center space-x-4">
                  {!isInitialized ? (
                    <button
                      onClick={handleStartSession}
                      disabled={connectionStatus === 'connecting'}
                      className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg flex items-center space-x-2 transition-colors"
                    >
                      {connectionStatus === 'connecting' ? (
                        <>
                          <LoadingSpinner className="w-4 h-4" />
                          <span>Connecting...</span>
                        </>
                      ) : (
                        <>
                          <VideoCameraIcon className="w-4 h-4" />
                          <span>Start Session</span>
                        </>
                      )}
                    </button>
                  ) : (
                    <button
                      onClick={handleStopSession}
                      className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg flex items-center space-x-2 transition-colors"
                    >
                      <XMarkIcon className="w-4 h-4" />
                      <span>End Session</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
            {}
            <div className="space-y-6">
              {}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Attention Status</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Current Score</span>
                    <span className="text-2xl font-bold text-blue-600">
                      {student.currentPrediction ? 
                        Math.round(student.currentPrediction.attention.score * 100) : 0}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Status</span>
                    <span className={`font-semibold ${
                      student.feedbackColor === 'green' ? 'text-green-600' :
                      student.feedbackColor === 'yellow' ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {student.currentPrediction?.attention.state || 'Unknown'}
                    </span>
                  </div>
                  <div className="bg-gray-100 rounded-lg p-3">
                    <p className="text-sm text-center text-gray-700">
                      {student.feedbackMessage}
                    </p>
                  </div>
                </div>
              </div>
              {}
              {student.currentPrediction && (
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">Engagement</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Level</span>
                      <span className="font-semibold">
                        {['Very Low', 'Low', 'High', 'Very High'][student.currentPrediction.engagement.level]}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Emotion</span>
                      <span className="font-semibold">
                        {['Bored', 'Confused', 'Engaged', 'Frustrated'][student.currentPrediction.emotion.class]}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              {}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Session Stats</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Duration</span>
                    <span className="font-semibold">
                      {student.isConnected ? 
                        `${Math.floor(Date.now() / 60000)}m` : '0m'
                      }
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Frames Processed</span>
                    <span className="font-semibold">
                      {student.attentionHistory.length}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Avg. Attention</span>
                    <span className="font-semibold">
                      {student.attentionHistory.length > 0 ? 
                        Math.round(student.attentionHistory.reduce((sum, val) => sum + val, 0) / student.attentionHistory.length * 100) : 0
                      }%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        {}
        <ConsentModal
          isOpen={showConsent}
          onAccept={giveConsent}
          onDecline={denyConsent}
          onClose={denyConsent}
        />
      </div>
    </ErrorBoundary>
  )
}
export default StudentPage
import { useAppStore } from '@/store/appStore'
import { useCallback, useEffect, useRef, useState } from 'react'
interface WebcamConfig {
  width: number
  height: number
  frameRate: number
  facingMode: 'user' | 'environment'
}
interface UseWebcamReturn {
  videoRef: React.RefObject<HTMLVideoElement>
  canvasRef: React.RefObject<HTMLCanvasElement>
  isWebcamEnabled: boolean
  isCapturing: boolean
  error: string | null
  startWebcam: () => Promise<void>
  stopWebcam: () => void
  captureFrame: () => string | null
  startCapture: () => void
  stopCapture: () => void
}
const DEFAULT_CONFIG: WebcamConfig = {
  width: 320,
  height: 240,
  frameRate: 15,
  facingMode: 'user',
}
export const useWebcam = (config: Partial<WebcamConfig> = {}): UseWebcamReturn => {
  const finalConfig = { ...DEFAULT_CONFIG, ...config }
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const [isWebcamEnabled, setIsWebcamEnabled] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { setWebcamEnabled, setConsentGiven } = useAppStore()
  const startWebcam = useCallback(async () => {
    try {
      setError(null)
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser')
      }
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: finalConfig.width },
          height: { ideal: finalConfig.height },
          frameRate: { ideal: finalConfig.frameRate },
          facingMode: finalConfig.facingMode,
        },
        audio: false,
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      if (canvasRef.current) {
        canvasRef.current.width = finalConfig.width
        canvasRef.current.height = finalConfig.height
      }
      setIsWebcamEnabled(true)
      setWebcamEnabled(true)
      setConsentGiven(true)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access webcam'
      setError(errorMessage)
      setIsWebcamEnabled(false)
      setWebcamEnabled(false)
      console.error('Webcam error:', err)
    }
  }, [finalConfig, setWebcamEnabled, setConsentGiven])
  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    stopCapture()
    setIsWebcamEnabled(false)
    setWebcamEnabled(false)
    setError(null)
  }, [setWebcamEnabled])
  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current || !isWebcamEnabled) {
      return null
    }
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return null
    }
    try {
      ctx.drawImage(video, 0, 0, finalConfig.width, finalConfig.height)
      const dataURL = canvas.toDataURL('image/jpeg', 0.7)
      return dataURL.split(',')[1]
    } catch (err) {
      console.error('Frame capture error:', err)
      return null
    }
  }, [isWebcamEnabled, finalConfig.width, finalConfig.height])
  const startCapture = useCallback(() => {
    if (isCapturing || captureIntervalRef.current) {
      return
    }
    setIsCapturing(true)
    const interval = 1000 / finalConfig.frameRate
    captureIntervalRef.current = setInterval(() => {
      const frameData = captureFrame()
      if (frameData) {
        const event = new CustomEvent('webcam-frame', {
          detail: {
            data: frameData,
            width: finalConfig.width,
            height: finalConfig.height,
            timestamp: Date.now() / 1000,
          },
        })
        window.dispatchEvent(event)
      }
    }, interval)
  }, [isCapturing, captureFrame, finalConfig.frameRate, finalConfig.width, finalConfig.height])
  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    setIsCapturing(false)
  }, [])
  useEffect(() => {
    return () => {
      stopWebcam()
    }
  }, [stopWebcam])
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopCapture()
      } else if (isWebcamEnabled) {
        startCapture()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [isWebcamEnabled, startCapture, stopCapture])
  return {
    videoRef,
    canvasRef,
    isWebcamEnabled,
    isCapturing,
    error,
    startWebcam,
    stopWebcam,
    captureFrame,
    startCapture,
    stopCapture,
  }
}
export const useWebcamConsent = () => {
  const [showConsent, setShowConsent] = useState(false)
  const [consentGiven, setConsentGiven] = useState(false)
  const { setConsentGiven: setStoreConsent } = useAppStore()
  const requestConsent = useCallback(() => {
    setShowConsent(true)
  }, [])
  const giveConsent = useCallback(() => {
    setConsentGiven(true)
    setStoreConsent(true)
    setShowConsent(false)
  }, [setStoreConsent])
  const denyConsent = useCallback(() => {
    setConsentGiven(false)
    setStoreConsent(false)
    setShowConsent(false)
  }, [setStoreConsent])
  return {
    showConsent,
    consentGiven,
    requestConsent,
    giveConsent,
    denyConsent,
  }
}
export const useWebcamPermissions = () => {
  const [permissionState, setPermissionState] = useState<PermissionState | null>(null)
  const [isSupported, setIsSupported] = useState(false)
  useEffect(() => {
    if ('permissions' in navigator) {
      setIsSupported(true)
      navigator.permissions.query({ name: 'camera' as PermissionName }).then(permission => {
        setPermissionState(permission.state)
        permission.addEventListener('change', () => {
          setPermissionState(permission.state)
        })
      }).catch(() => {
        setIsSupported(false)
      })
    } else {
      setIsSupported(false)
    }
  }, [])
  return {
    permissionState,
    isSupported,
    isGranted: permissionState === 'granted',
    isDenied: permissionState === 'denied',
    isPrompt: permissionState === 'prompt',
  }
}
export const useWebcamDevices = () => {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null)
  useEffect(() => {
    const enumerateDevices = async () => {
      try {
        const allDevices = await navigator.mediaDevices.enumerateDevices()
        const videoDevices = allDevices.filter(device => device.kind === 'videoinput')
        setDevices(videoDevices)
        if (videoDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId)
        }
      } catch (err) {
        console.error('Failed to enumerate devices:', err)
      }
    }
    enumerateDevices()
    navigator.mediaDevices.addEventListener('devicechange', enumerateDevices)
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', enumerateDevices)
    }
  }, [selectedDeviceId])
  return {
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    hasMultipleDevices: devices.length > 1,
  }
}
export const assessFrameQuality = (canvas: HTMLCanvasElement): {
  brightness: number
  contrast: number
  sharpness: number
  quality: 'good' | 'fair' | 'poor'
} => {
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return { brightness: 0, contrast: 0, sharpness: 0, quality: 'poor' }
  }
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = imageData.data
  let brightness = 0
  let contrast = 0
  const pixelCount = data.length / 4
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    brightness += (r + g + b) / 3
  }
  brightness = brightness / pixelCount / 255
  const brightnessMean = brightness * 255
  let variance = 0
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    const gray = (r + g + b) / 3
    variance += Math.pow(gray - brightnessMean, 2)
  }
  contrast = Math.sqrt(variance / pixelCount) / 255
  let sharpness = 0
  for (let y = 1; y < canvas.height - 1; y++) {
    for (let x = 1; x < canvas.width - 1; x++) {
      const idx = (y * canvas.width + x) * 4
      const center = (data[idx] + data[idx + 1] + data[idx + 2]) / 3
      const neighbors = [
        (data[idx - 4] + data[idx - 3] + data[idx - 2]) / 3,
        (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3,
        (data[idx - canvas.width * 4] + data[idx - canvas.width * 4 + 1] + data[idx - canvas.width * 4 + 2]) / 3,
        (data[idx + canvas.width * 4] + data[idx + canvas.width * 4 + 1] + data[idx + canvas.width * 4 + 2]) / 3,
      ]
      const maxDiff = Math.max(...neighbors.map(n => Math.abs(center - n)))
      sharpness += maxDiff
    }
  }
  sharpness = sharpness / (canvas.width * canvas.height) / 255
  let quality: 'good' | 'fair' | 'poor' = 'good'
  if (brightness < 0.3 || brightness > 0.8 || contrast < 0.1 || sharpness < 0.05) {
    quality = 'poor'
  } else if (brightness < 0.4 || brightness > 0.7 || contrast < 0.15 || sharpness < 0.1) {
    quality = 'fair'
  }
  return {
    brightness: Math.round(brightness * 100) / 100,
    contrast: Math.round(contrast * 100) / 100,
    sharpness: Math.round(sharpness * 100) / 100,
    quality,
  }
}
/**
 * DersLens AI Service Configuration
 * Enhanced service with MPIIGaze and FER2013 models
 */

export const AI_SERVICE_CONFIG = {
  // Enhanced AI Service (NEW - with MPIIGaze + FER2013)
  ENHANCED_SERVICE: {
    baseUrl: 'http://localhost:5000',
    endpoints: {
      analyze: '/api/analyze',
      health: '/health',
      models: '/models/status',
      demo: '/demo'
    },
    features: {
      gazeTracking: true,
      emotionDetection: true,
      attentionAnalysis: true,
      headPoseNormalization: true,
      robustFaceDetection: true
    },
    models: {
      gaze: 'MPIIGaze Excellent (3.39° MAE)',
      emotion: 'FER2013 PyTorch (7 emotions)',
      engagement: 'DAiSEE Model (4 states)',
      attention: 'Mendeley Model (3 levels)',
      faceDetection: 'MediaPipe + OpenCV Fallback'
    }
  },
  
  // Legacy Service (OLD)
  LEGACY_SERVICE: {
    baseUrl: 'http://localhost:8000',
    endpoints: {
      analyze: '/api/analyze'
    },
    features: {
      gazeTracking: false,
      emotionDetection: true,
      attentionAnalysis: true,
      headPoseNormalization: false,
      robustFaceDetection: false
    }
  }
};

// Current active service - switch here to change service
export const ACTIVE_SERVICE = AI_SERVICE_CONFIG.ENHANCED_SERVICE;

// Helper function to get full endpoint URL
export const getEndpointUrl = (endpoint: keyof typeof ACTIVE_SERVICE.endpoints): string => {
  return `${ACTIVE_SERVICE.baseUrl}${ACTIVE_SERVICE.endpoints[endpoint]}`;
};

// Service status checker
export const checkServiceHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(getEndpointUrl('health'), { 
      method: 'GET',
      timeout: 5000 
    } as RequestInit);
    return response.ok;
  } catch (error) {
    console.error('Service health check failed:', error);
    return false;
  }
};

export default AI_SERVICE_CONFIG;

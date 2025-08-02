interface AnalysisResult {
  attention: number;
  engagement: number;
  emotion: string;
  emotionConfidence: number;
  gazeDirection: string;
  faceDetected: boolean;
  timestamp: number;
}
interface CalibrationPoint {
  x: number;
  y: number;
  timestamp: number;
}
class DersLensAPIService {
  private baseURL: string;
  constructor() {
    this.baseURL = process.env.NODE_ENV === 'production' 
      ? '/api' 
      : (process.env.REACT_APP_API_URL || 'http://localhost:8000/api');
  }
  /**
   * Analyzes a video frame using the AI service
   * @param frameBlob - The video frame as a Blob
   * @returns Analysis result with attention, engagement, and emotion data
   */
  async analyzeFrame(frameBlob: Blob): Promise<AnalysisResult> {
    try {
      const formData = new FormData();
      formData.append('frame', frameBlob, 'frame.jpg');
      
      const response = await fetch(`${this.baseURL}/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Analysis request failed with status: ${response.status}`);
      }
      
      const data = await response.json();
      return this.mapAIServiceResponse(data);
      
    } catch (error) {
      console.error('Analysis service error:', error instanceof Error ? error.message : 'Unknown error');
      throw error; // Let the caller handle the error appropriately
    }
  }
  async calibrateGaze(points: CalibrationPoint[]): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/calibrate-gaze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ calibration_points: points }),
      });
      return response.ok;
    } catch (error) {
      console.error('Calibration error:', error);
      return false;
    }
  }
  async getHealthStatus(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check error:', error);
      return false;
    }
  }
  async getModelInfo(): Promise<{
    emotion_model: string;
    attention_model: string;
    engagement_model: string;
    onnx_runtime: boolean;
  }> {
    try {
      const response = await fetch(`${this.baseURL}/models/info`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Model info error:', error);
    }
    return {
      emotion_model: 'FER2013',
      attention_model: 'DAiSEE',
      engagement_model: 'Mendeley',
      onnx_runtime: true
    };
  }
  private getMockAnalysis(): AnalysisResult {
    // Map all 8 emotions from AI service to Turkish
    const emotionMapping = {
      'neutral': 'Nötr',
      'happy': 'Mutlu', 
      'sad': 'Üzgün',
      'angry': 'Kızgın',
      'surprise': 'Şaşkın',
      'disgust': 'İğrenmiş',
      'fear': 'Korkmuş',
      'contempt': 'Küçümsemiş'
    };
    const emotions = Object.values(emotionMapping);
    const gazeDirections = ['Merkez', 'Sol', 'Sağ', 'Yukarı', 'Aşağı'];
    const baseAttention = 85;
    const baseEngagement = 78;
    const attentionVariation = (Math.random() - 0.5) * 20; 
    const engagementVariation = (Math.random() - 0.5) * 16; 
    const attention = Math.max(50, Math.min(100, baseAttention + attentionVariation));
    const engagement = Math.max(60, Math.min(100, baseEngagement + engagementVariation));
    let emotion;
    if (attention > 85 && engagement > 80) {
      emotion = Math.random() > 0.5 ? 'Mutlu' : 'Şaşkın';
    } else if (attention > 70) {
      emotion = Math.random() > 0.5 ? 'Nötr' : 'Mutlu';
    } else {
      emotion = Math.random() > 0.5 ? 'Üzgün' : 'Kızgın';
    }
    const gazeDirection = attention > 80 && Math.random() > 0.3 
      ? 'Merkez' 
      : gazeDirections[Math.floor(Math.random() * gazeDirections.length)];
    return {
      attention: Math.round(attention),
      engagement: Math.round(engagement),
      emotion,
      emotionConfidence: Math.round(Math.random() * 15 + 85), 
      gazeDirection,
      faceDetected: true,
      timestamp: Date.now()
    };
  }
  
  private mapAIServiceResponse(data: any): AnalysisResult {
    // Map all 8 emotions from AI service to Turkish
    const emotionMapping = {
      'neutral': 'Nötr',
      'happy': 'Mutlu', 
      'sad': 'Üzgün',
      'angry': 'Kızgın',
      'surprise': 'Şaşkın',
      'disgust': 'İğrenmiş',
      'fear': 'Korkmuş',
      'contempt': 'Küçümsemiş'
    };
    
    // Extract emotion data from AI service response
    const displayMetrics = data.display_metrics || {};
    const rawEmotion = displayMetrics.emotion || 'neutral';
    const mappedEmotion = emotionMapping[rawEmotion as keyof typeof emotionMapping] || emotionMapping.neutral;
    
    return {
      attention: Math.round(displayMetrics.attention || 0),
      engagement: Math.round(displayMetrics.engagement || 0),
      emotion: mappedEmotion,
      emotionConfidence: Math.round(displayMetrics.emotion_confidence || 0),
      gazeDirection: displayMetrics.gaze_direction || 'Merkez',
      faceDetected: data.face_detected || false,
      timestamp: Date.now()
    };
  }
}
export const apiService = new DersLensAPIService();
export type { AnalysisResult, CalibrationPoint };
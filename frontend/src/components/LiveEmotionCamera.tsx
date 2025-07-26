import React, { useEffect, useRef, useState } from 'react';
interface AttentionPrediction {
  attention_score: number;
  attention_state: string;
  confidence: number;
}
interface EngagementPrediction {
  engagement_level: number;
  engagement_probabilities: Record<string, number>;
}
interface EmotionPrediction {
  emotion_class: number;
  emotion_probabilities: Record<string, number>;
}
interface FaceFeatures {
  face_detected: boolean;
  face_confidence: number;
  phone_usage_detected: boolean;
}
interface PredictionResult {
  attention: AttentionPrediction;
  engagement: EngagementPrediction;
  emotion: EmotionPrediction;
  face_features: FaceFeatures;
  processing_time_ms: number;
  model_version: string;
  frame_id: string;
  timestamp: string;
}
const API_URL = 'http://localhost:8000/api/analyze'
const emotionLabels: Record<number, string> = {
  0: 'Boredom',
  1: 'Confusion',
  2: 'Engagement',
  3: 'Frustration',
  4: 'Happy',
  5: 'Sadness',
  6: 'Surprise',
  7: 'Disgust',
  8: 'Fear',
  9: 'Anger',
  10: 'Neutral',
};
const engagementLabels: Record<number, string> = {
  0: 'Very Low',
  1: 'Low',
  2: 'High',
  3: 'Very High',
};
const attentionStateColors: Record<string, string> = {
  attentive: '#4caf50',
  inattentive: '#e53935',
};
export const LiveEmotionCamera: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [attentionHistory, setAttentionHistory] = useState<number[]>([]);
  const [engagementHistory, setEngagementHistory] = useState<number[]>([]);
  const [emotionHistory, setEmotionHistory] = useState<number[]>([]);
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setConnected(true);
      } catch (e) {
        setError('Could not access webcam.');
        setConnected(false);
      }
    };
    startCamera();
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  useEffect(() => {
    if (!connected) return;
    const interval = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async (blob) => {
        if (!blob) return;
        setLoading(true);
        setError(null);
        try {
          const formData = new FormData();
          formData.append('frame', blob, 'frame.jpg');
          const res = await fetch(API_URL, {
            method: 'POST',
            body: formData,
          });
          if (!res.ok) throw new Error('Backend error');
          const data = await res.json();
          setPrediction(data);
          setAttentionHistory(h => [...h.slice(-29), data.attention.attention_score]);
          setEngagementHistory(h => [...h.slice(-29), data.engagement.engagement_level]);
          setEmotionHistory(h => [...h.slice(-29), data.emotion.emotion_class]);
        } catch (e) {
          setError('Prediction failed. Is the backend running?');
        } finally {
          setLoading(false);
        }
      }, 'image/jpeg');
    }, 1000);
    return () => clearInterval(interval);
  }, [connected]);
  const renderTrend = (history: number[], color: string, labels?: Record<number, string>) => (
    <svg width="100%" height="40" viewBox="0 0 120 40">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        points={history.map((v, i) => `${i * 4},${40 - v * 36}`).join(' ')}
      />
      {labels && history.length > 0 && (
        <text x="2" y="12" fontSize="10" fill={color}>{labels[history[history.length - 1]]}</text>
      )}
    </svg>
  );
  return (
    <div style={{ maxWidth: 480, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 16, boxShadow: '0 4px 24px rgba(0,0,0,0.08)' }}>
      <h2 style={{ textAlign: 'center', marginBottom: 16 }}>🎥 Live Attention & Emotion</h2>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', borderRadius: 12, background: '#222' }} />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
      <div style={{ marginTop: 24, textAlign: 'center' }}>
        {error && <div style={{ color: '#e53935', marginBottom: 12 }}>{error}</div>}
        {!connected && !error && <div style={{ color: '#757575' }}>Connecting to camera...</div>}
        {loading && <div style={{ color: '#2196f3' }}>Analyzing...</div>}
        {prediction && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 22, fontWeight: 600, marginBottom: 8 }}>
              <span style={{ color: attentionStateColors[prediction.attention.attention_state] || '#333' }}>
                {prediction.attention.attention_state.toUpperCase()}
              </span>
              {` (conf: ${(prediction.attention.confidence * 100).toFixed(1)}%)`}
            </div>
            <div style={{ fontSize: 18, margin: '8px 0' }}>
              Attention Score: <b>{(prediction.attention.attention_score * 100).toFixed(1)}%</b>
            </div>
            <div style={{ fontSize: 18, margin: '8px 0' }}>
              Engagement: <b>{engagementLabels[prediction.engagement.engagement_level]}</b>
            </div>
            <div style={{ fontSize: 18, margin: '8px 0' }}>
              Emotion: <b>{emotionLabels[prediction.emotion.emotion_class]}</b>
            </div>
            <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>
              Model: {prediction.model_version} | Response: {prediction.processing_time_ms.toFixed(0)} ms
            </div>
            <div style={{ marginTop: 10, fontSize: 14 }}>
              {prediction.face_features.face_detected ? 'Face Detected' : 'No Face Detected'}
              {prediction.face_features.phone_usage_detected && ' | Phone Usage Detected'}
            </div>
            {}
            <div style={{ marginTop: 10, fontSize: 13 }}>
              <b>Engagement Probabilities:</b>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {Object.entries(prediction.engagement.engagement_probabilities).map(([k, v]) => (
                  <li key={k}>{k}: {(v * 100).toFixed(1)}%</li>
                ))}
              </ul>
              <b>Emotion Probabilities:</b>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {Object.entries(prediction.emotion.emotion_probabilities).map(([k, v]) => (
                  <li key={k}>{k}: {(v * 100).toFixed(1)}%</li>
                ))}
              </ul>
            </div>
            {}
            <div style={{ marginTop: 18 }}>
              <div style={{ fontSize: 13, color: '#888' }}>Attention Trend (last 30s)</div>
              {renderTrend(attentionHistory, '#4caf50')}
              <div style={{ fontSize: 13, color: '#888', marginTop: 8 }}>Engagement Trend</div>
              {renderTrend(engagementHistory, '#2196f3', engagementLabels)}
              <div style={{ fontSize: 13, color: '#888', marginTop: 8 }}>Emotion Trend</div>
              {renderTrend(emotionHistory, '#e91e63', emotionLabels)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
export default LiveEmotionCamera; 
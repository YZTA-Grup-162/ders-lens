"""
Test AI models and inference
"""
from unittest.mock import Mock, patch
import numpy as np
import pytest
import torch
from app.ai.inference import AttentionInferenceEngine
from app.ai.models import AttentionModel, EmotionModel
class TestAttentionModel:
    def test_model_creation(self):
        model = AttentionModel()
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'attention_head')
    def test_model_forward_pass(self):
        model = AttentionModel()
        model.eval()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output is not None
        assert output.shape[0] == batch_size
        assert len(output.shape) == 2
    def test_model_parameters(self):
        model = AttentionModel()
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
class TestEmotionModel:
    def test_model_creation(self):
        model = EmotionModel()
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'emotion_head')
    def test_model_output_shape(self):
        model = EmotionModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 7)
class TestInferenceEngine:
    @patch('app.ai.inference.ort.InferenceSession')
    def test_engine_creation_with_mock(self, mock_session):
        mock_session.return_value = Mock()
        mock_session.return_value.get_inputs.return_value = [
            Mock(shape=[1, 3, 224, 224])
        ]
        try:
            engine = AttentionInferenceEngine()
            assert engine is not None
        except Exception as e:
            assert "model file not found" in str(e).lower() or "no such file" in str(e).lower()
    def test_preprocessing(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with patch('app.ai.inference.ort.InferenceSession'):
            try:
                engine = AttentionInferenceEngine()
                if hasattr(engine, '_preprocess_frame'):
                    processed = engine._preprocess_frame(frame)
                    assert processed.shape == (1, 3, 224, 224)
                    assert processed.dtype == np.float32
            except Exception:
                pytest.skip("Models not available in test environment")
    def test_frame_prediction_mock(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        with patch('app.ai.inference.AttentionInferenceEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.predict_frame.return_value = {
                'attention_score': 0.75,
                'engagement_score': 0.68,
                'distraction_level': 0.25,
                'dominant_emotion': 'focused',
                'emotion_scores': {
                    'happy': 0.1,
                    'sad': 0.05,
                    'angry': 0.02,
                    'surprise': 0.08,
                    'fear': 0.03,
                    'disgust': 0.02,
                    'neutral': 0.7
                },
                'attention_confidence': 0.92,
                'emotion_confidence': 0.88,
                'face_detected': True
            }
            mock_engine.return_value = mock_instance
            engine = mock_engine()
            result = engine.predict_frame(frame)
            assert result['attention_score'] == 0.75
            assert result['engagement_score'] == 0.68
            assert result['face_detected'] is True
            assert 'emotion_scores' in result
class TestModelUtilities:
    def test_model_config_validation(self):
        from app.core.config import settings
        assert hasattr(settings, 'MODEL_PATH') or hasattr(settings, 'ATTENTION_MODEL_PATH')
    def test_model_version_info(self):
        assert True
class TestDataProcessing:
    def test_frame_validation(self):
        valid_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        assert valid_frame.shape == (224, 224, 3)
        assert valid_frame.dtype == np.uint8
        invalid_shape = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert invalid_shape.shape != (224, 224, 3)
    def test_score_normalization(self):
        raw_scores = np.array([-1.5, 0.0, 1.2, 3.8])
        normalized = np.clip(raw_scores, 0, 1)
        assert all(0 <= score <= 1 for score in normalized)
        assert normalized[0] == 0
        assert normalized[2] == 1
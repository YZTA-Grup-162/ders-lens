
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from app.core.config import settings
from app.models.schemas import (AttentionPrediction, AttentionState,
                                EmotionClass, EmotionPrediction,
                                EngagementLevel, EngagementPrediction,
                                FaceFeatures, PredictionResult)

logger = logging.getLogger(__name__)
class ONNXModelOptimizer:
    def __init__(self):
        self.optimization_passes = [
            "eliminate_deadend",
            "eliminate_identity", 
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_log_softmax",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
            "lift_lexical_references"
        ]
    def optimize_model(self, input_path: str, output_path: str) -> None:
        try:
            model = onnx.load(input_path)
            optimized_model = onnx.helper.optimize(model, self.optimization_passes)
            onnx.save(optimized_model, output_path)
            onnx.checker.check_model(optimized_model)
            logger.info(f"Model optimized and saved to {output_path}")
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            raise
    def quantize_model(self, input_path: str, output_path: str) -> None:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QUInt8,
                optimize_model=True
            )
            logger.info(f"Model quantized and saved to {output_path}")
        except ImportError:
            logger.warning("ONNX quantization not available, skipping quantization")
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            raise
class FeatureExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    def extract_face_features(self, frame: np.ndarray) -> FaceFeatures:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            if len(faces) == 0:
                return FaceFeatures(
                    face_detected=False,
                    face_confidence=0.0,
                    face_bbox=None,
                    landmarks=None,
                    pose_features=None,
                    phone_usage_detected=False
                )
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_roi = gray[y:y+h, x:x+w]
            pose_features = self._extract_pose_features(face_roi)
            phone_usage = self._detect_phone_usage(frame, face)
            return FaceFeatures(
                face_detected=True,
                face_confidence=0.9,
                face_bbox=[int(x), int(y), int(w), int(h)],
                landmarks=None,
                pose_features=pose_features,
                phone_usage_detected=phone_usage
            )
        except Exception as e:
            logger.error(f"Face feature extraction failed: {e}")
            return FaceFeatures(
                face_detected=False,
                face_confidence=0.0,
                face_bbox=None,
                landmarks=None,
                pose_features=None,
                phone_usage_detected=False
            )
    def _extract_pose_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        if face_roi.size == 0:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        h, w = face_roi.shape
        left_half = face_roi[:, :w//2]
        right_half = face_roi[:, w//2:]
        left_intensity = np.mean(left_half)
        right_intensity = np.mean(right_half)
        yaw = (right_intensity - left_intensity) / (left_intensity + right_intensity + 1e-6)
        top_half = face_roi[:h//2, :]
        bottom_half = face_roi[h//2:, :]
        top_intensity = np.mean(top_half)
        bottom_intensity = np.mean(bottom_half)
        pitch = (bottom_intensity - top_intensity) / (top_intensity + bottom_intensity + 1e-6)
        return {
            "yaw": float(np.clip(yaw, -1.0, 1.0)),
            "pitch": float(np.clip(pitch, -1.0, 1.0)),
            "roll": 0.0  # Simplified
        }
    def _detect_phone_usage(self, frame: np.ndarray, face_bbox: np.ndarray) -> bool:
        try:
            x, y, w, h = face_bbox
            roi_y = min(y + h, frame.shape[0] - 20)
            roi = frame[roi_y:roi_y + 40, max(0, x-20):min(frame.shape[1], x+w+20)]
            if roi.size == 0:
                return False
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            return edge_ratio > 0.1
        except Exception:
            return False
    def extract_handcrafted_features(self, frame: np.ndarray, face_features: FaceFeatures) -> np.ndarray:
        features = []
        if face_features.face_detected and face_features.face_bbox:
            x, y, w, h = face_features.face_bbox
            frame_h, frame_w = frame.shape[:2]
            features.extend([
                x / frame_w,
                y / frame_h,
                w / frame_w,
                h / frame_h,
                (x + w/2) / frame_w,
                (y + h/2) / frame_h,
            ])
            face_area = w * h
            frame_area = frame_w * frame_h
            features.append(face_area / frame_area)
            if face_features.pose_features:
                features.extend([
                    face_features.pose_features["yaw"],
                    face_features.pose_features["pitch"],
                    face_features.pose_features["roll"]
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            features.extend([
                1.0 if face_features.phone_usage_detected else 0.0,
                face_features.face_confidence
            ])
        else:
            features = [0.0] * 12
        while len(features) < 20:
            features.append(0.0)
        features = features[:20]
        return np.array(features, dtype=np.float32)
class ONNXInferenceEngine:
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        self.session = None
        self.input_names = []
        self.output_names = []
        self.feature_extractor = FeatureExtractor()
        self._setup_session()
    def _setup_session(self):
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            logger.info(f"ONNX session initialized with providers: {self.session.get_providers()}")
            logger.info(f"Input names: {self.input_names}")
            logger.info(f"Output names: {self.output_names}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            raise
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, FaceFeatures]:
        target_size = (224, 224)
        frame_resized = cv2.resize(frame, target_size)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_chw = np.transpose(frame_normalized, (2, 0, 1))
        face_features = self.feature_extractor.extract_face_features(frame)
        handcrafted_features = self.feature_extractor.extract_handcrafted_features(
            frame, face_features
        )
        return frame_chw, handcrafted_features, face_features
    def preprocess_sequence(self, frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        sequence_length = min(len(frames), settings.sequence_length)
        if len(frames) < settings.sequence_length:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames = frames + [last_frame] * (settings.sequence_length - len(frames))
        else:
            frames = frames[-settings.sequence_length:]
        visual_features = []
        handcrafted_features = []
        face_features_list = []
        for frame in frames:
            frame_chw, handcrafted, face_feat = self.preprocess_frame(frame)
            visual_features.append(frame_chw)
            handcrafted_features.append(handcrafted)
            face_features_list.append(face_feat)
        visual_batch = np.stack(visual_features, axis=0)
        handcrafted_batch = np.stack(handcrafted_features, axis=0)
        visual_batch = np.expand_dims(visual_batch, axis=0)
        handcrafted_batch = np.expand_dims(handcrafted_batch, axis=0)
        return {
            "frames": visual_batch,
            "handcrafted_features": handcrafted_batch,
            "face_features": face_features_list[-1]  # Use last frame's face features
        }
    async def predict_async(self, frames: List[np.ndarray]) -> PredictionResult:
        try:
            start_time = time.time()
            if len(frames) == 1:
                frame_chw, handcrafted, face_features = self.preprocess_frame(frames[0])
                visual_input = np.expand_dims(np.expand_dims(frame_chw, axis=0), axis=0)
                handcrafted_input = np.expand_dims(np.expand_dims(handcrafted, axis=0), axis=0)
            else:
                preprocessed = self.preprocess_sequence(frames)
                visual_input = preprocessed["frames"]
                handcrafted_input = preprocessed["handcrafted_features"]
                face_features = preprocessed["face_features"]
            input_dict = {
                self.input_names[0]: visual_input,
                self.input_names[1]: handcrafted_input
            }
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                self.session.run,
                self.output_names,
                input_dict
            )
            attention_score = float(outputs[0][0])
            engagement_logits = outputs[1][0]
            emotion_logits = outputs[2][0]
            attention_pred = AttentionPrediction(
                attention_score=attention_score,
                attention_state=AttentionState.ATTENTIVE if attention_score > 0.5 else AttentionState.INATTENTIVE,
                confidence=max(attention_score, 1.0 - attention_score)
            )
            engagement_probs = torch.softmax(torch.tensor(engagement_logits), dim=0).numpy()
            engagement_pred = EngagementPrediction(
                engagement_level=EngagementLevel(int(np.argmax(engagement_logits))),
                engagement_probabilities={
                    "very_low": float(engagement_probs[0]),
                    "low": float(engagement_probs[1]),
                    "high": float(engagement_probs[2]),
                    "very_high": float(engagement_probs[3])
                }
            )
            emotion_probs = torch.softmax(torch.tensor(emotion_logits), dim=0).numpy()
            emotion_pred = EmotionPrediction(
                emotion_class=EmotionClass(int(np.argmax(emotion_logits))),
                emotion_probabilities={
                    "boredom": float(emotion_probs[0]),
                    "confusion": float(emotion_probs[1]),
                    "engagement": float(emotion_probs[2]),
                    "frustration": float(emotion_probs[3])
                }
            )
            processing_time = (time.time() - start_time) * 1000
            return PredictionResult(
                frame_id=f"frame_{int(time.time() * 1000)}",
                attention=attention_pred,
                engagement=engagement_pred,
                emotion=emotion_pred,
                face_features=face_features,
                processing_time_ms=processing_time,
                model_version="1.0.0"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    def predict_sync(self, frames: List[np.ndarray]) -> PredictionResult:
        return asyncio.run(self.predict_async(frames))
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        times = []
        for _ in range(num_iterations):
            start = time.time()
            try:
                self.predict_sync([dummy_frame])
                times.append((time.time() - start) * 1000)
            except Exception as e:
                logger.warning(f"Benchmark iteration failed: {e}")
        if not times:
            return {"error": "All benchmark iterations failed"}
        return {
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "p95_time_ms": np.percentile(times, 95),
            "p99_time_ms": np.percentile(times, 99)
        }
class ModelManager:
    def __init__(self):
        self.models: Dict[str, ONNXInferenceEngine] = {}
        self.primary_model: Optional[str] = None
        self.load_models()
    def load_models(self):
        onnx_dir = Path(settings.onnx_model_path)
        if not onnx_dir.exists():
            logger.warning(f"ONNX model directory not found: {onnx_dir}")
            return
        model_files = list(onnx_dir.glob("*.onnx"))
        for model_file in model_files:
            try:
                model_name = model_file.stem
                self.models[model_name] = ONNXInferenceEngine(str(model_file))
                if self.primary_model is None:
                    self.primary_model = model_name
                logger.info(f"Loaded ONNX model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    async def predict(self, frames: List[np.ndarray], model_name: Optional[str] = None) -> PredictionResult:
        if not self.models:
            raise RuntimeError("No ONNX models available")
        target_model = model_name or self.primary_model
        if target_model not in self.models:
            target_model = self.primary_model
        return await self.models[target_model].predict_async(frames)
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "available_models": list(self.models.keys()),
            "primary_model": self.primary_model,
            "total_models": len(self.models)
        }
model_manager = ModelManager()
class AttentionInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(Path(settings.onnx_model_path) / "best_model.onnx")
        self.engine = ONNXInferenceEngine(model_path)
        self.model_version = "1.0.0"
        self.model_path = model_path
        self.input_shape = (1, 3, 224, 224)
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        try:
            result = self.engine.predict_sync([frame])
            return {
                "attention_score": result.attention.attention_score,
                "engagement_score": result.engagement.engagement_level.value / 3.0,  # Normalize to [0,1]
                "distraction_level": 1.0 - result.attention.attention_score,
                "attention_confidence": result.attention.confidence,
                "dominant_emotion": result.emotion.emotion_class.name.lower(),
                "emotion_scores": result.emotion.emotion_probabilities,
                "emotion_confidence": 0.8,  # Default confidence
                "valence": 0.5,  # Default values
                "arousal": 0.5,
                "face_detected": result.face_features.face_detected,
                "head_pose": {
                    "x": result.face_features.pose_features.get("yaw", 0.0) if result.face_features.pose_features else 0.0,
                    "y": result.face_features.pose_features.get("pitch", 0.0) if result.face_features.pose_features else 0.0,
                    "z": result.face_features.pose_features.get("roll", 0.0) if result.face_features.pose_features else 0.0,
                } if result.face_features.pose_features else {"x": 0.0, "y": 0.0, "z": 0.0},
                "focus_regions": [],  # Default empty list
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "attention_score": 0.5,
                "engagement_score": 0.5,
                "distraction_level": 0.5,
                "attention_confidence": 0.0,
                "dominant_emotion": "neutral",
                "emotion_scores": {"neutral": 1.0},
                "emotion_confidence": 0.0,
                "valence": 0.5,
                "arousal": 0.5,
                "face_detected": False,
                "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                "focus_regions": [],
            }
    async def predict_async(self, frames: List[np.ndarray]) -> PredictionResult:
        return await self.engine.predict_async(frames)
    def predict_sync(self, frames: List[np.ndarray]) -> PredictionResult:
        return self.engine.predict_sync(frames)
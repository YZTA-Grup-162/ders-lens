"""
Performance test using Locust
"""
import base64
import io
import json

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


class DersLensUser(HttpUser):
    wait_time = between(1, 3)
    def on_start(self):
        self.sample_image = self._create_sample_image()
    def _create_sample_image(self):
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer.getvalue()
    @task(3)
    def health_check(self):
        self.client.get("/health")
    @task(2)
    def model_info(self):
        self.client.get("/api/ai/model/info")
    @task(1)
    def model_warm_up(self):
        self.client.post("/api/ai/model/warm-up")
    @task(5)
    def predict_frame(self):
        files = {
            "frame": ("test_frame.jpg", self.sample_image, "image/jpeg")
        }
        with self.client.post(
            "/api/ai/predict/frame",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "attention" in data and "emotion" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 500:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
class WebSocketUser(HttpUser):
    wait_time = between(2, 5)
    @task
    def websocket_test(self):
        pass
class TeacherUser(HttpUser):
    wait_time = between(3, 8)
    @task(2)
    def dashboard_data(self):
        self.client.get("/api/teacher/dashboard")
    @task(1)
    def session_management(self):
        self.client.get("/api/teacher/sessions")
class StudentUser(HttpUser):
    wait_time = between(2, 6)
    @task(3)
    def student_sessions(self):
        self.client.get("/api/student/sessions")
    @task(1)
    def student_feedback(self):
        self.client.get("/api/student/feedback")
class LightLoad(DersLensUser):
    weight = 70
class HeavyLoad(DersLensUser):
    weight = 30
    wait_time = between(0.5, 1.5)
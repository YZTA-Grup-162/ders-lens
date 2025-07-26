"""
Test API endpoints
"""
import json
import pytest
#from fastapi.testclient import TestClient
def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "AttentionPulse API" in data["message"]
    assert data["version"] == "1.0.0"
def test_api_docs(client: TestClient):
    response = client.get("/docs")
    assert response.status_code == 200
def test_model_info(client: TestClient):
    response = client.get("/api/ai/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "supported_emotions" in data
    assert "attention_classes" in data
def test_model_warm_up(client: TestClient):
    response = client.post("/api/ai/model/warm-up")
    assert response.status_code in [200, 500]
class TestPredictionAPI:
    def test_predict_frame_missing_file(self, client: TestClient):
        response = client.post("/api/ai/predict/frame")
        assert response.status_code == 422
    def test_predict_frame_invalid_file(self, client: TestClient):
        files = {"frame": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/ai/predict/frame", files=files)
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
class TestWebSocketAPI:
    def test_websocket_test_endpoint(self, client: TestClient):
        with client.websocket_connect("/ws/test") as websocket:
            data = websocket.receive_text()
            assert "Hello from test WebSocket!" in data
            websocket.send_text("test message")
            response = websocket.receive_text()
            assert "Echo: test message" == response
class TestStudentAPI:
    def test_student_endpoints_exist(self, client: TestClient):
        response = client.get("/api/student/sessions")
        assert response.status_code != 404
class TestTeacherAPI:
    def test_teacher_endpoints_exist(self, client: TestClient):
        response = client.get("/api/teacher/dashboard")
        assert response.status_code != 404
class TestAuthAPI:
    def test_auth_endpoints_exist(self, client: TestClient):
        response = client.post("/api/auth/login")
        assert response.status_code != 404
class TestCORS:
    def test_cors_preflight(self, client: TestClient):
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            #}
        #)
        assert response.status_code in [200, 204]
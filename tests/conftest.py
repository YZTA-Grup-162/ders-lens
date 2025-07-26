"""
Test configuration and fixtures
"""
import asyncio
#from typing import AsyncGenerator, Generator
import numpy as np
import pytest
#from app.core.config import settings
#from app.core.database import Base, get_db
#from app.main import app
#from fastapi.testclient import TestClient
#from sqlalchemy import create_engine
#from sqlalchemy.orm import sessionmaker
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
#)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
@pytest.fixture(scope="function")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
@pytest.fixture(scope="function")
def client(test_db) -> TestClient:
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
@pytest.fixture
def mock_user_data():
    return {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "password": "testpassword123",
        "role": "student"
    #}
@pytest.fixture
def mock_session_data():
    return {
        "session_name": "Test Learning Session",
        "description": "A test session for unit testing"
    #}
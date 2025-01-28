# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from main import app, get_embedding
import json

client = TestClient(app)

@pytest.fixture
def sample_faqs():
    return [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "What is FastAPI?", "answer": "FastAPI is a web framework."}
    ]

def test_embedding_generation():
    text = "Sample question"
    embedding = get_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_query_endpoint():
    response = client.post("/query", json={"text": "What is Python?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from main import app, get_embedding
import json

client = TestClient(app)

@pytest.fixture
def sample_faqs():
    return [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "What is FastAPI?", "answer": "FastAPI is a web framework."}
    ]

def test_embedding_generation():
    text = "Sample question"
    embedding = get_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_query_endpoint():
    response = client.post("/query", json={"text": "What is Python?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from main import app, get_embedding
import json

client = TestClient(app)

@pytest.fixture
def sample_faqs():
    return [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "What is FastAPI?", "answer": "FastAPI is a web framework."}
    ]

def test_embedding_generation():
    text = "Sample question"
    embedding = get_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_query_endpoint():
    response = client.post("/query", json={"text": "What is Python?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from main import app, get_embedding
import json

client = TestClient(app)

@pytest.fixture
def sample_faqs():
    return [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "What is FastAPI?", "answer": "FastAPI is a web framework."}
    ]

def test_embedding_generation():
    text = "Sample question"
    embedding = get_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_query_endpoint():
    response = client.post("/query", json={"text": "What is Python?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from main import app, get_embedding
import json

client = TestClient(app)

@pytest.fixture
def sample_faqs():
    return [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "What is FastAPI?", "answer": "FastAPI is a web framework."}
    ]

def test_embedding_generation():
    text = "Sample question"
    embedding = get_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_query_endpoint():
    response = client.post("/query", json={"text": "What is Python?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from main import app, get_embedding
import json

client = TestClient(app)

@pytest.fixture
def sample_faqs():
    return [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "What is FastAPI?", "answer": "FastAPI is a web framework."}
    ]

def test_embedding_generation():
    text = "Sample question"
    embedding = get_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_query_endpoint():
    response = client.post("/query", json={"text": "What is Python?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422
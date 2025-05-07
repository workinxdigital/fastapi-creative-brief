
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_collect_input():
    response = client.post("/collect-input", json={
        "website_url": "https://example.com",
        "product_type": "Skincare",
        "known_issues": "Dry skin",
        "target_audience": "Teens",
        "amazon_listing": ""
    })
    assert response.status_code == 200
    data = response.json()
    assert "form" in data
    assert "session_id" in data

def test_list_sessions():
    response = client.get("/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_session_by_id():
    post_resp = client.post("/collect-input", json={
        "website_url": "https://example.com",
        "product_type": "Supplements",
        "known_issues": "Fatigue",
        "target_audience": "Adults",
        "amazon_listing": ""
    })
    session_id = post_resp.json()["session_id"]
    get_resp = client.get(f"/sessions/{session_id}")
    assert get_resp.status_code == 200
    session_data = get_resp.json()
    assert session_data["id"] == session_id
    assert "brand_input" in session_data

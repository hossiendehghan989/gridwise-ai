from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
health = client.get('/health')
assert health.status_code == 200
quality = client.get('/quality')
assert quality.status_code == 200
forecast = client.post('/forecast', json={'horizon': 3})
assert forecast.status_code == 200
assert len(forecast.json()) == 3
print('api_smoke=passed')

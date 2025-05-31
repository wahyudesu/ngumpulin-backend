from prometheus_client import Counter, Histogram, Gauge
from fastapi import FastAPI
import time

# Prometheus metrics
REQUEST_COUNT = Counter('fastapi_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('fastapi_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('fastapi_active_connections', 'Active connections')
DOCUMENT_PROCESSING_DURATION = Histogram('document_processing_duration_seconds', 'Document processing duration')
ML_MODEL_PREDICTIONS = Counter('ml_model_predictions_total', 'Total ML model predictions')

def setup_metrics(app: FastAPI):
    @app.middleware("http")
    async def add_prometheus_metrics(request, call_next):
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path
            ).inc()
            
            REQUEST_DURATION.observe(time.time() - start_time)
            
            return response
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
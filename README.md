# Ngumpulin Backend - LLMOps Platform

## Fitur Utama
- [`FastAPI`](https://fastapi.tiangolo.com/) untuk API dan backend
- [`LangGraph`](https://github.com/langchain-ai/langgraph) untuk agentic workflow LLM
- [`Airflow`](https://airflow.apache.org/) untuk workflow ML pipeline
- [`MLflow`](https://mlflow.org/) untuk tracking experiment
- [`Prometheus`](https://prometheus.io/) & [`Grafana`](https://grafana.com/) untuk monitoring
- [`Docker Compose`](https://docs.docker.com/compose/) untuk orkestrasi multi-service
- [`uv`](https://docs.astral.sh/uv/) untuk manajemen environment Python

## Struktur Project

- `app/` : Source code FastAPI (API, core, routers, utils)
- `airflow/` : Konfigurasi, DAG, dan plugin Airflow
- `mlflow/` : Konfigurasi MLflow
- `monitoring/` : Konfigurasi Prometheus & Grafana
- `notebook/` : Notebook eksperimen
- `scripts/` : Script setup, backup, entrypoint, dsb
- `docker-compose.yaml` : Orkestrasi seluruh service
- `Dockerfile.*` : Dockerfile untuk masing-masing service
- `.env` : Environment variable (jangan commit ke repo publik)

## Prasyarat
- Sudah install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Sudah install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows: gunakan PowerShell)

## Instalasi & Setup (Local)

```powershell
# Clone repository
git clone https://github.com/yourusername/ngumpulin-backend.git
cd ngumpulin-backend

# Setup Python (opsional, untuk development)
uv python install 3.12.8
uv venv
.venv\Scripts\Activate.ps1
uv sync
uv pip install -e .
uv run pre-commit install

# Copy dan edit file .env
cp .env.example .env
# Edit .env sesuai kebutuhan

# Build & start semua service (FastAPI, Airflow, MLflow, monitoring, dsb)
docker compose up --build -d

# Cek status service
docker compose ps
```

## Akses Service
- FastAPI: [http://localhost:8000](http://localhost:8000)
- MLflow: [http://localhost:5000](http://localhost:5000)
- Airflow: [http://localhost:8080](http://localhost:8080) (user: admin/admin)
- Adminer: [http://localhost:8088](http://localhost:8088)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3030](http://localhost:3030) (user: admin/admin)

## Pengujian

```powershell
uv run pytest
```

## Penggunaan API

### Health check
```powershell
Invoke-WebRequest http://localhost:8000/health
```

### Upload file assignment
```powershell
Invoke-RestMethod -Uri http://localhost:8000/upload -Method Post -Form @{uuid='123'; file_url='https://...' }
```

### Generate Assignment Feedback (LangGraph Agent)
Endpoint ini menggunakan LangGraph untuk menghasilkan feedback assignment berbasis LLM.

**Endpoint:**
```
POST /feedback
```

**Contoh payload:**
```json
{
  "title": "Tugas Matematika Integral",
  "description": "Kerjakan soal integral berikut dengan metode substitusi.",
  "content": "\n\nSoal: ∫2x dx\nJawaban: ...",
  "persona": "Berikan feedback dengan gaya ramah dan membangun."
}
```

**Contoh request (PowerShell):**
```powershell
$body = @{ 
  title = 'Tugas Matematika Integral';
  description = 'Kerjakan soal integral berikut dengan metode substitusi.';
  content = 'Soal: ∫2x dx
Jawaban: ...';
  persona = 'Berikan feedback dengan gaya ramah dan membangun.'
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/feedback -Method Post -Body $body -ContentType 'application/json'
```

**Contoh response:**
```json
{
  "summary": "Ringkasan isi assignment ...",
  "relevance_analysis": "Analisis relevansi judul dan deskripsi ...",
  "feedback_analysis": "Feedback akademik ...",
  "personalized_feedback": "Feedback yang dipersonalisasi ...",
  "combined_output": "🎓 Final Personalized Feedback: ..."
}
```

Endpoint ini akan mengembalikan feedback assignment yang sudah dianalisis dan dipersonalisasi oleh LLM agent berbasis LangGraph.

## Deployment ke VPS/GCP

1. Deploy VM (Ubuntu) di GCP Compute Engine
2. Install Docker & Docker Compose di VM
3. Clone repo & copy `.env` ke VM
4. Jalankan:
   ```bash
   docker compose up --build -d
   ```
5. (Opsional) Setup Nginx reverse proxy & SSL untuk domain

## Backup & Maintenance

- Backup: jalankan `scripts/backup.sh`
- Cleanup: jalankan `scripts/cleanup.sh`
- Restore: manual dari hasil backup

---

> Project ini dikembangkan untuk workflow LLMOps/ML pipeline modern, siap untuk deployment di cloud maupun VPS.
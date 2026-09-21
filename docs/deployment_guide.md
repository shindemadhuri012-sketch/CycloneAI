# CycloneAI: Production Deployment Guide
**Smart India Hackathon 2026**  
**Document Version:** 1.0.0  
**Target Architecture:** Linux (Ubuntu 22.04 LTS+) / Windows 11 / macOS 14+ / Docker Container  

---

## 1. System Requirements

### Hardware Requirements
- **CPU**: Minimum 2 physical x86_64 or ARM64 cores (Intel Core i3 / AMD Ryzen 3 or AWS `t3.medium` equivalent).
- **RAM**: Minimum 2 GB available RAM (CycloneAI memory footprint is approximately 320 MB).
- **Disk Storage**: 2.5 GB free space (including Python virtual environment, dependencies, models, and processed IBTrACS datasets).
- **GPU**: **None required**. All feature extraction, gradient boosting regressions, calibrations, and TreeSHAP calculations are optimized for CPU execution.

### Software Prerequisites
- **Python**: 3.10, 3.11, or 3.12 (Python 3.13 supported).
- **Node.js** (Optional for building from source): Node.js v18.0.0+ / npm v9.0.0+. Pre-built distribution files are already compiled in `frontend/dist/`.
- **Docker** (Optional for containerized deployment): Docker Engine 24.0+ and Docker Compose v2.20+.

---

## 2. Quickstart Deployment (3 Minutes)

### Option A: Standard Native Deployment (Recommended for Local Judging)

1. **Clone and Enter Repository**:
   ```bash
   git clone https://github.com/CycloneAI/CycloneAI.git
   cd CycloneAI
   ```

2. **Set up Python Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv backend\.venv
   & "backend\.venv\Scripts\Activate.ps1"
   pip install -r backend\requirements.txt

   # Linux / macOS
   python3 -m venv backend/.venv
   source backend/.venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. **Launch Production Server**:
   ```bash
   # Windows (One-Click)
   start_production.bat

   # Cross-Platform Python
   python run_production.py
   ```
   *The server starts on `http://localhost:8000`, serving the React frontend SPA at `/` and the REST API at `/api`.*

---

### Option B: Docker Container Deployment

1. **Build and Run via Docker Compose**:
   ```bash
   docker compose up -d --build
   ```

2. **Verify Container Health**:
   ```bash
   docker ps
   # Container 'cycloneai-production' should show status: Up (healthy)
   ```

3. **View Live Server Logs**:
   ```bash
   docker compose logs -f
   ```

4. **Stop Container**:
   ```bash
   docker compose down
   ```

---

## 3. Environment Variables Reference

Configure runtime parameters by editing `backend/.env` (template in `backend/.env.example`):

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `ENVIRONMENT` | string | `production` | Operational environment mode (`production` or `development`). |
| `HOST` | string | `0.0.0.0` | Network interface address for the Uvicorn ASGI server. |
| `PORT` | integer | `8000` | Port on which the FastAPI application listens. |
| `SERVE_STATIC_SPA` | boolean | `True` | Automatically serves `frontend/dist` on the root `/` URL. |
| `CORS_ORIGINS` | list | `["*"]` | Allowed CORS origins for browser security. |

---

## 4. Production Hardening for State Disaster Centers (SEOCs)

### Systemd Service Configuration (Linux Ubuntu / Debian)
To run CycloneAI as an automated background daemon that restarts on failure or reboot:

1. Create service unit `/etc/systemd/system/cycloneai.service`:
   ```ini
   [Unit]
   Description=CycloneAI Intelligent Cyclone Analysis & Prediction System
   After=network.target

   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/opt/CycloneAI
   Environment="PORT=8000"
   Environment="HOST=127.0.0.1"
   Environment="ENVIRONMENT=production"
   ExecStart=/opt/CycloneAI/backend/.venv/bin/python /opt/CycloneAI/run_production.py
   Restart=always
   RestartSec=5s

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable cycloneai
   sudo systemctl start cycloneai
   sudo systemctl status cycloneai
   ```

---

### Nginx Reverse Proxy Configuration (with SSL / HTTPS)

```nginx
server {
    listen 80;
    server_name cycloneai.gov.in;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name cycloneai.gov.in;

    ssl_certificate /etc/letsencrypt/live/cycloneai.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cycloneai.gov.in/privkey.pem;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 5. Automated Health Checks & Monitoring

The system exposes live monitoring endpoints:
- **Liveness Probe**: `GET /api/health` returns HTTP 200 and subsystem statuses.
- **Model Inventory**: `GET /api/system/metrics` verifies model file versions and Phase 10 test benchmarks.
- **Automated Smoke Test**:
  ```bash
  python scripts/verify_deployment.py
  ```

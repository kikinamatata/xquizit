# Deployment & Operations Guide

## Overview

This guide covers deploying and operating the xquizit interview system in development and production environments, including infrastructure requirements, configuration, monitoring, and best practices.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Production Deployment](#production-deployment)
3. [Infrastructure Requirements](#infrastructure-requirements)
4. [External Service Setup](#external-service-setup)
5. [Environment Configuration](#environment-configuration)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Logging](#monitoring--logging)
8. [Performance Tuning](#performance-tuning)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## Development Setup

### Prerequisites

**Backend Requirements**:
- Python 3.9 - 3.12 (not 3.13+ due to RealtimeTTS compatibility)
- pip or poetry for package management
- Git

**Frontend Requirements**:
- Node.js 18+ and npm
- Modern web browser (Chrome, Firefox, Edge)

**Optional**:
- CUDA-capable GPU for faster TTS (falls back to CPU)
- Docker for containerized deployment

---

### Local Development - Backend

**1. Clone Repository**:
```bash
git clone <repository-url>
cd xquizit/backend
```

**2. Create Virtual Environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n xquizit python=3.11
conda activate xquizit
```

**3. Install Dependencies**:
```bash
pip install -r requirements.txt
```

**4. Configure Environment**:
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# Required: GEMINI_API_KEY, RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY
```

**5. Start Development Server**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Server Access**:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### Local Development - Frontend

**1. Navigate to Frontend**:
```bash
cd xquizit/frontend
```

**2. Install Dependencies**:
```bash
npm install
```

**3. Configure API Endpoints**:

Edit `frontend/src/config.js`:
```javascript
export const API_BASE_URL = "http://localhost:8000";
export const WS_BASE_URL = "ws://localhost:8000";
```

**4. Start Development Server**:
```bash
npm run dev
```

**Access**: http://localhost:5173

---

### Development Workflow

**Backend Hot Reload**:
- `--reload` flag auto-restarts on code changes
- State is lost (in-memory sessions cleared)

**Frontend Hot Module Replacement (HMR)**:
- Vite provides instant HMR
- State preserved during most updates

**Testing Interview Flow**:
1. Prepare sample PDF/DOCX files (resume + job description)
2. Upload documents
3. Test audio recording (requires microphone permission)
4. Monitor console for latency logs
5. Check browser DevTools Network tab for SSE/WebSocket traffic

---

## Production Deployment

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Load Balancer                       │
│                  (nginx, Traefik, AWS ALB)               │
└─────────────┬──────────────────────────┬─────────────────┘
              │                          │
              │ HTTPS/WSS               │
              │                          │
    ┌─────────▼──────────┐    ┌─────────▼──────────┐
    │  Backend Instance  │    │  Backend Instance  │
    │   (FastAPI + TTS)  │    │   (FastAPI + TTS)  │
    └─────────┬──────────┘    └─────────┬──────────┘
              │                          │
              │                          │
         ┌────▼──────────────────────────▼────┐
         │     Shared State (Redis/DB)        │
         │  - Session Storage                 │
         │  - Interview State                 │
         └────────────────────────────────────┘
              │                │
              │                │
    ┌─────────▼─────┐   ┌─────▼──────────┐
    │  Gemini API   │   │  RunPod API    │
    │  (External)   │   │  (External)    │
    └───────────────┘   └────────────────┘
```

---

### Backend Production Deployment

#### Option 1: Traditional Server (Recommended for Start)

**1. Install System Dependencies**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3-pip python3-venv nginx

# RedHat/CentOS
sudo yum install -y python3.11 python3-pip nginx
```

**2. Setup Application**:
```bash
# Create application directory
sudo mkdir -p /opt/xquizit/backend
sudo chown $USER:$USER /opt/xquizit/backend
cd /opt/xquizit/backend

# Clone and setup
git clone <repository-url> .
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure Environment**:
```bash
sudo nano /opt/xquizit/backend/.env

# Add production values
GEMINI_API_KEY=your_production_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id
RUNPOD_API_KEY=your_runpod_key

# TTS optimization for production
TTS_DEVICE=cuda  # or cpu if no GPU
KOKORO_VOICE=af_bella

# LLM optimization
GEMINI_THINKING_BUDGET=0
GEMINI_MAX_OUTPUT_TOKENS=1024
GEMINI_TEMPERATURE=0.7
```

**4. Create Systemd Service**:
```bash
sudo nano /etc/systemd/system/xquizit-backend.service
```

```ini
[Unit]
Description=xquizit Interview Backend
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/xquizit/backend
Environment="PATH=/opt/xquizit/backend/venv/bin"
ExecStart=/opt/xquizit/backend/venv/bin/gunicorn \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    --bind 0.0.0.0:8000 \
    --access-logfile /var/log/xquizit/access.log \
    --error-logfile /var/log/xquizit/error.log \
    --log-level info \
    main:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**5. Create Log Directory**:
```bash
sudo mkdir -p /var/log/xquizit
sudo chown www-data:www-data /var/log/xquizit
```

**6. Start Service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable xquizit-backend
sudo systemctl start xquizit-backend
sudo systemctl status xquizit-backend
```

**7. Configure Nginx Reverse Proxy**:
```bash
sudo nano /etc/nginx/sites-available/xquizit
```

```nginx
# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificates (use Let's Encrypt certbot)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # REST API and SSE endpoints
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket timeout settings
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # File upload size limit
    client_max_body_size 10M;
}
```

**8. Enable Site and Restart Nginx**:
```bash
sudo ln -s /etc/nginx/sites-available/xquizit /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

**9. Setup SSL Certificate (Let's Encrypt)**:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

#### Option 2: Docker Deployment

**1. Create Dockerfile** (`backend/Dockerfile`):
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "--bind", "0.0.0.0:8000", "main:app"]
```

**2. Create docker-compose.yml**:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}
      - RUNPOD_API_KEY=${RUNPOD_API_KEY}
      - TTS_DEVICE=cpu
    volumes:
      - ./backend:/app
      - backend-logs:/var/log
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  backend-logs:
```

**3. Build and Run**:
```bash
docker-compose up -d
```

---

### Frontend Production Deployment

#### Build for Production

**1. Update Configuration**:

Edit `frontend/src/config.js`:
```javascript
const isDev = import.meta.env.DEV;
export const API_BASE_URL = isDev
  ? "http://localhost:8000"
  : "https://api.your-domain.com";

export const WS_BASE_URL = isDev
  ? "ws://localhost:8000"
  : "wss://api.your-domain.com";
```

**2. Build**:
```bash
cd frontend
npm run build
```

**Output**: `frontend/dist/` directory

---

#### Option 1: Serve with Nginx

**1. Copy Build Files**:
```bash
sudo mkdir -p /var/www/xquizit
sudo cp -r frontend/dist/* /var/www/xquizit/
```

**2. Nginx Configuration**:
```nginx
server {
    listen 443 ssl http2;
    server_name app.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/app.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/app.your-domain.com/privkey.pem;

    root /var/www/xquizit;
    index index.html;

    # SPA routing - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
}
```

---

#### Option 2: Serve with CDN (Vercel, Netlify, Cloudflare Pages)

**Vercel**:
```bash
npm install -g vercel
cd frontend
vercel --prod
```

**Netlify**:
```bash
npm install -g netlify-cli
cd frontend
netlify deploy --prod --dir=dist
```

**Configuration**:
- Set environment variables: `VITE_API_BASE_URL`, `VITE_WS_BASE_URL`
- Configure redirects for SPA routing

---

## Infrastructure Requirements

### Compute Resources

#### Backend Server

**Minimum (Development)**:
- CPU: 2 cores
- RAM: 4 GB
- Storage: 20 GB
- Network: 10 Mbps

**Recommended (Production)**:
- CPU: 4-8 cores
- RAM: 8-16 GB (TTS uses significant memory)
- Storage: 50 GB SSD
- Network: 100 Mbps
- GPU: NVIDIA GPU with 4+ GB VRAM (optional, for faster TTS)

**Concurrent Users**:
- 1-10 users: 4 cores, 8 GB RAM
- 10-50 users: 8 cores, 16 GB RAM
- 50+ users: Scale horizontally with load balancer

---

#### Frontend Hosting

**Static Site**:
- Minimal resources (CDN-based)
- Global edge network for low latency
- Auto-scaling

---

### Network Requirements

**Bandwidth Estimation (Per Interview)**:
- Document upload: ~2-5 MB
- Audio upload (30 min): ~30-50 MB
- Audio download (responses): ~10-20 MB
- **Total per interview**: ~50-75 MB

**Concurrent Interviews**:
- 10 concurrent: ~500-750 MB/session (15-25 Mbps avg)
- 50 concurrent: ~2.5-3.75 GB/session (75-125 Mbps avg)

---

### Storage Requirements

**Session Data** (In-Memory):
- Per session: ~1-5 MB (conversation history, state)
- 100 concurrent sessions: ~100-500 MB RAM

**Logs**:
- Access logs: ~100 MB/day (1000 interviews)
- Error logs: Minimal (< 10 MB/day)
- RunPod logs: Stored locally in `~/.cache/whisper-live/logs/`

**Persistent Storage** (If Implemented):
- Interview transcripts: ~5-10 KB per interview
- Audio recordings (optional): ~50 MB per interview

---

## External Service Setup

### 1. Google Gemini API

**Setup**:
1. Visit https://aistudio.google.com/app/apikey
2. Create new API key
3. Add to `backend/.env`: `GEMINI_API_KEY=...`

**Pricing** (as of 2025):
- Gemini 2.5 Flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
- Thinking tokens (if enabled): More expensive
- Estimated cost per interview: $0.05-$0.15

**Quotas**:
- Free tier: 15 requests/minute, 1M tokens/day
- Production: Request quota increase

**Optimization**:
- Set `GEMINI_THINKING_BUDGET=0` for fastest/cheapest
- Limit `GEMINI_MAX_OUTPUT_TOKENS=1024`

---

### 2. RunPod Serverless (Whisper Transcription)

**Setup**:

1. **Create Account**:
   - Visit https://runpod.io
   - Sign up and add payment method

2. **Deploy Faster Whisper Endpoint**:
   - Go to **Serverless** → **Deploy a New Endpoint**
   - Search "Faster Whisper" in template marketplace
   - Configure:
     - **Model**: `small` (development), `large-v3` (production)
     - **Min Workers**: 0 (auto-scale)
     - **Max Workers**: 5-10 (based on expected concurrent users)
     - **GPU**: A40 or A100 (better performance)
     - **Idle Timeout**: 60 seconds

3. **Get Credentials**:
   - Copy **Endpoint ID** (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
   - Create API key in **Settings** → **API Keys**

4. **Configure Backend**:
   ```env
   RUNPOD_ENDPOINT_ID=your-endpoint-id
   RUNPOD_API_KEY=your-api-key
   WHISPERLIVE_MODEL=small  # Must match deployed model
   ```

**Pricing**:
- Pay per second of worker runtime
- A40: ~$0.00016/second (~$0.60/hour)
- A100: ~$0.00035/second (~$1.26/hour)
- Estimated cost per 30-min interview: $0.50-$1.00
- Workers auto-scale down when idle (no cost)

**Performance**:
- Cold start: ~5-10 seconds (first request)
- Warm execution: ~1-3 seconds per transcription
- Concurrent capacity: Unlimited (serverless)

---

### 3. RealtimeTTS (Local TTS)

**Setup**:
- Included in `requirements.txt`
- Automatically downloads Kokoro model on first run
- Model stored in `~/.cache/realtimetts/`

**GPU Acceleration** (Optional):
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Configuration**:
```env
TTS_DEVICE=auto  # auto-detects CUDA/CPU
KOKORO_VOICE=af_bella  # af_bella, af_sarah, am_adam, am_michael
KOKORO_SPEED=1.0
```

**Resource Usage**:
- CPU mode: ~1-2 seconds per sentence, high CPU usage
- GPU mode: ~0.5-1 seconds per sentence, ~2-4 GB VRAM

---

## Environment Configuration

### Production .env Template

```env
# ============================================
# Core API Keys (REQUIRED)
# ============================================
GEMINI_API_KEY=your_production_gemini_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
RUNPOD_API_KEY=your_runpod_api_key

# ============================================
# RunPod Transcription Settings
# ============================================
RUNPOD_MAX_BUFFER_SECONDS=30
WHISPERLIVE_LANGUAGE=en
WHISPERLIVE_MODEL=large-v3  # Production: large-v3, Dev: small
WHISPERLIVE_USE_VAD=true
WHISPERLIVE_NO_SPEECH_THRESH=0.45
WHISPERLIVE_SAME_OUTPUT_THRESHOLD=7
WHISPERLIVE_TRANSCRIPTION_INTERVAL=3.0

# ============================================
# TTS Configuration
# ============================================
TTS_DEVICE=cuda  # cuda, cpu, or auto
KOKORO_VOICE=af_bella  # af_bella, af_sarah, am_adam, am_michael
KOKORO_SPEED=1.0

# ============================================
# LLM Performance Optimization
# ============================================
GEMINI_THINKING_BUDGET=0  # 0=fastest, 512-1024=balanced, 2048+=highest quality
GEMINI_INCLUDE_THOUGHTS=false
GEMINI_MAX_OUTPUT_TOKENS=1024  # Lower=faster, Higher=more detailed
GEMINI_TEMPERATURE=0.7

# ============================================
# Legacy (Optional, no longer used)
# ============================================
DEEPINFRA_API_KEY=
```

---

### Frontend Environment Variables

**Vite Environment Files**:

**.env.development**:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

**.env.production**:
```env
VITE_API_BASE_URL=https://api.your-domain.com
VITE_WS_BASE_URL=wss://api.your-domain.com
```

**Usage in Code**:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
```

---

## Security Hardening

### Backend Security

**1. CORS Configuration**:

**Development** (permissive):
```python
allow_origins=["*"]
```

**Production** (restrictive):
```python
allow_origins=[
    "https://app.your-domain.com",
    "https://www.your-domain.com"
]
```

---

**2. Input Validation**:
- File upload size limits (already enforced)
- File type validation (PDF, DOCX only)
- Audio chunk size limits
- Answer length limits

---

**3. Authentication** (Recommended Addition):

**JWT-based authentication**:
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Validate JWT token
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return get_user_from_token(token)

@app.post("/upload-documents")
async def upload_documents(user=Depends(verify_token)):
    # Only authenticated users can upload
    ...
```

---

**4. Rate Limiting**:

**Using slowapi**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/upload-documents")
@limiter.limit("5/minute")
async def upload_documents(request: Request):
    ...
```

---

**5. Secret Management**:

**Use environment-specific secret managers**:
- AWS: Secrets Manager, Parameter Store
- Azure: Key Vault
- GCP: Secret Manager
- Self-hosted: HashiCorp Vault

**Example with AWS Secrets Manager**:
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

secrets = get_secret('xquizit/production')
GEMINI_API_KEY = secrets['GEMINI_API_KEY']
```

---

**6. Security Headers** (Nginx):
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "geolocation=(), microphone=(self), camera=()" always;
```

---

### Frontend Security

**1. Content Security Policy (CSP)**:
```html
<meta http-equiv="Content-Security-Policy" content="
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline';
  connect-src 'self' https://api.your-domain.com wss://api.your-domain.com;
  media-src 'self' blob:;
">
```

**2. Secure Cookie Settings** (if using cookies):
```javascript
document.cookie = "session=...; Secure; HttpOnly; SameSite=Strict";
```

**3. Input Sanitization**:
- Sanitize user inputs before display
- Use React's built-in XSS protection (JSX escaping)

---

## Monitoring & Logging

### Application Logging

**Backend Logging Configuration**:

**main.py**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/xquizit/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

**Log Important Events**:
```python
logger.info(f"Interview started: session_id={session_id}")
logger.warning(f"Transcription error: {error}")
logger.error(f"LLM API failure: {exception}")
```

---

### Metrics Collection

**Using Prometheus**:

**1. Install dependencies**:
```bash
pip install prometheus-client
```

**2. Add metrics**:
```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
interview_counter = Counter('interviews_total', 'Total interviews started')
question_duration = Histogram('question_generation_seconds', 'Time to generate question')
transcription_errors = Counter('transcription_errors_total', 'Total transcription errors')

# Instrument code
interview_counter.inc()
with question_duration.time():
    generate_question()

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**3. Configure Prometheus**:
```yaml
scrape_configs:
  - job_name: 'xquizit-backend'
    static_configs:
      - targets: ['backend:8000']
```

---

### Error Tracking

**Using Sentry**:

**1. Install**:
```bash
pip install sentry-sdk[fastapi]
```

**2. Configure**:
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="https://your-sentry-dsn@sentry.io/project",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
    environment="production"
)
```

**3. Automatic error capture** for unhandled exceptions

---

### Health Checks

**Add health check endpoint**:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "tts": TTS_SERVICE is not None,
            "runpod": RUNPOD_SERVICE is not None,
            "gemini": settings.gemini_api_key != ""
        }
    }
```

**Configure load balancer health check**:
- Path: `/health`
- Interval: 30 seconds
- Timeout: 5 seconds
- Healthy threshold: 2
- Unhealthy threshold: 3

---

## Performance Tuning

### Backend Optimization

**1. Worker Configuration**:
```bash
# CPU-bound: workers = (2 × num_cores) + 1
# I/O-bound: workers = (4 × num_cores) + 1
gunicorn -k uvicorn.workers.UvicornWorker -w 8 main:app
```

**2. Connection Pooling** (if using database):
```python
from sqlalchemy import create_engine
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0
)
```

**3. Caching** (Redis):
```python
import redis
cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Cache TTS audio for common phrases
def get_tts_audio(text):
    cached = cache.get(f"tts:{text}")
    if cached:
        return cached
    audio = generate_tts(text)
    cache.setex(f"tts:{text}", 3600, audio)  # Cache 1 hour
    return audio
```

**4. LLM Optimization**:
- `GEMINI_THINKING_BUDGET=0` (fastest)
- `GEMINI_MAX_OUTPUT_TOKENS=1024` (prevent verbosity)
- Consider caching frequent question patterns

---

### Frontend Optimization

**1. Code Splitting**:
```javascript
// React lazy loading
const ChatInterface = lazy(() => import('./components/ChatInterface'));
const UploadScreen = lazy(() => import('./components/UploadScreen'));
```

**2. Asset Optimization**:
```bash
# Optimize images
npm install -D vite-plugin-imagemin

# vite.config.js
import viteImagemin from 'vite-plugin-imagemin';

export default {
  plugins: [
    viteImagemin({
      gifsicle: { optimizationLevel: 7 },
      optipng: { optimizationLevel: 7 },
      mozjpeg: { quality: 80 }
    })
  ]
};
```

**3. CDN for Static Assets**:
- Serve JS/CSS/images from CDN
- Enable browser caching
- Use HTTP/2 or HTTP/3

---

## Backup & Recovery

### Current State (In-Memory)

**Risk**: All sessions lost on server restart

**Mitigation**:
1. Implement persistent storage (Redis, PostgreSQL)
2. Regular session snapshots
3. Graceful shutdown handling

---

### Recommended: Persistent Storage

**Redis-based Session Storage**:

**1. Install Redis**:
```bash
sudo apt install redis-server
```

**2. Modify Backend**:
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def save_session(session_id, session_data):
    redis_client.setex(
        f"session:{session_id}",
        3600,  # TTL: 1 hour
        json.dumps(session_data)
    )

def get_session(session_id):
    data = redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None
```

**3. Backup Redis**:
```bash
# Enable AOF (Append-Only File)
redis-cli CONFIG SET appendonly yes

# Backup RDB snapshot
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backup/dump-$(date +%Y%m%d).rdb
```

---

### Application Logs

**Backup Strategy**:
```bash
# Rotate logs daily
sudo logrotate /etc/logrotate.d/xquizit

# logrotate config
/var/log/xquizit/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload xquizit-backend > /dev/null
    endscript
}
```

---

## Troubleshooting

### Common Issues

#### 1. TTS Initialization Fails

**Symptom**: `RuntimeError: TTS service not initialized`

**Solutions**:
- Check Python version (must be < 3.13)
- Verify `realtimetts[kokoro]` installed: `pip list | grep realtimetts`
- Check disk space in `~/.cache/realtimetts/`
- Review startup logs for errors

---

#### 2. WebSocket Connection Refused

**Symptom**: Frontend can't connect to `/ws/transcribe/`

**Solutions**:
- Check backend is running: `sudo systemctl status xquizit-backend`
- Verify nginx WebSocket config (Upgrade headers)
- Check firewall rules: `sudo ufw status`
- Test directly: `wscat -c ws://localhost:8000/ws/transcribe/test-session`

---

#### 3. SSE Stream Not Receiving Events

**Symptom**: Frontend EventSource connects but no events received

**Solutions**:
- Check nginx buffering: `proxy_buffering off;`
- Verify CORS headers
- Check backend logs for errors
- Test with curl: `curl -N http://localhost:8000/stream-answer?session_id=...&answer=test`

---

#### 4. RunPod Transcription Timeout

**Symptom**: Transcription requests timeout

**Solutions**:
- Check RunPod endpoint status in dashboard
- Verify endpoint ID and API key in `.env`
- Increase worker count in RunPod settings
- Check RunPod logs: `~/.cache/whisper-live/logs/runpod/`
- Reduce `WHISPERLIVE_TRANSCRIPTION_INTERVAL` if too frequent

---

#### 5. High Memory Usage

**Symptom**: Backend consuming excessive RAM

**Solutions**:
- Implement session cleanup (delete old sessions)
- Reduce `RUNPOD_MAX_BUFFER_SECONDS` (default: 30)
- Limit concurrent interviews
- Use persistent storage instead of in-memory
- Monitor with: `htop` or `free -h`

---

#### 6. Slow LLM Responses

**Symptom**: Interview questions take > 5 seconds

**Solutions**:
- Verify `GEMINI_THINKING_BUDGET=0`
- Check `GEMINI_MAX_OUTPUT_TOKENS=1024` (not too high)
- Monitor Gemini API quotas
- Check network latency to Gemini API
- Consider caching common question patterns

---

### Diagnostic Commands

**Check Backend Status**:
```bash
sudo systemctl status xquizit-backend
sudo journalctl -u xquizit-backend -f  # Follow logs
```

**Test API Endpoints**:
```bash
# Health check
curl http://localhost:8000/health

# Test document upload
curl -X POST -F "resume=@resume.pdf" -F "job_description=@jd.pdf" http://localhost:8000/upload-documents

# Test WebSocket
wscat -c ws://localhost:8000/ws/transcribe/test-session-id
```

**Monitor Resources**:
```bash
htop  # CPU and memory
iotop  # Disk I/O
iftop  # Network I/O
```

**Check Logs**:
```bash
tail -f /var/log/xquizit/access.log
tail -f /var/log/xquizit/error.log
tail -f /var/log/nginx/error.log
```

---

## Summary

This deployment guide covers:

- **Development setup** for rapid iteration
- **Production deployment** with systemd, nginx, and SSL
- **Infrastructure requirements** for scaling
- **External service configuration** (Gemini, RunPod, TTS)
- **Security hardening** with CORS, authentication, rate limiting
- **Monitoring** with logging, metrics, and error tracking
- **Performance tuning** for optimal latency and throughput
- **Backup strategies** for data persistence
- **Troubleshooting** common operational issues

For production deployment, prioritize:
1. SSL/TLS encryption (HTTPS/WSS)
2. Authentication and authorization
3. Persistent session storage (Redis)
4. Monitoring and alerting
5. Regular backups
6. Performance optimization
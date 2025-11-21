# xquizit - AI-Powered Interview Screening System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![React](https://img.shields.io/badge/react-18-blue)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/fastapi-latest-green)](https://fastapi.tiangolo.com)

xquizit is a sophisticated real-time AI interview screening system that conducts automated technical interviews with live audio and text streaming. The system analyzes candidate resumes and job descriptions to create tailored interview strategies, conducts adaptive interviews with intelligent follow-up questions, and provides a seamless candidate experience with real-time transcription and natural conversational audio.

---

## Key Features

### Intelligent Interview Orchestration
- **LangGraph-based multiagent workflow** for adaptive interview logic
- **Document analysis** to create personalized interview strategies
- **Context-aware follow-up questions** with intelligent topic rotation
- **Time-constrained interviews** (45 minutes, 16 questions max)

### Real-Time Communication
- **Live audio transcription** via RunPod serverless Whisper
- **Streaming text responses** via Server-Sent Events (SSE)
- **Natural TTS audio** using RealtimeTTS with Kokoro engine
- **WebSocket-based audio streaming** for low-latency transcription

### Optimized Performance
- **LLM optimization** with thinking budgets and token limits (30-60% faster)
- **Streaming architecture** for instant feedback and progressive rendering
- **Serverless integrations** for unlimited scalability
- **GPU-accelerated TTS** (optional, falls back to CPU)

---

## Technology Stack

### Backend
- **FastAPI** - High-performance async web framework
- **LangGraph** - State machine orchestration for interview workflow
- **LangChain** - LLM integration framework
- **Google Gemini 2.5 Flash** - Natural language understanding and generation
- **RealtimeTTS + Kokoro** - High-quality text-to-speech synthesis
- **RunPod Serverless** - Whisper-based audio transcription
- **PyPDF2 + python-docx** - Document text extraction

### Frontend
- **React 18** - Modern component-based UI
- **Vite** - Lightning-fast build tool with HMR
- **EventSource API** - Server-Sent Events for streaming
- **WebSocket API** - Real-time bidirectional communication
- **Web Audio API** - Audio capture and processing

### External Services
- **Google Gemini API** - LLM provider
- **RunPod API** - Serverless transcription
- **Kokoro Engine** - Local TTS synthesis

---

## Quick Start

### Prerequisites

- Python 3.9 - 3.12 (not 3.13+)
- Node.js 18+
- Google Gemini API key
- RunPod account with Whisper endpoint

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Access**: http://localhost:5173

---

## System Architecture

```
┌─────────────┐
│   FRONTEND  │  React + Vite (http://localhost:5173)
│   (React)   │  - Document upload UI
└──────┬──────┘  - Real-time chat interface
       │         - Audio recording and playback
       │         - Live transcription display
       │
       │ REST, SSE, WebSocket
       │
┌──────▼──────┐
│   BACKEND   │  FastAPI (http://localhost:8000)
│  (FastAPI)  │  - LangGraph interview engine
└──────┬──────┘  - Document processing
       │         - TTS service (Kokoro)
       │         - RunPod transcription service
       │
       ├──────────────────────────────┬─────────────────────┐
       │                              │                     │
┌──────▼────────┐          ┌──────────▼──────┐   ┌────────▼──────────┐
│ Google Gemini │          │  RunPod API     │   │  Kokoro TTS       │
│  (LLM API)    │          │  (Whisper)      │   │  (Local Engine)   │
└───────────────┘          └─────────────────┘   └───────────────────┘
```

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Core Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture, technology stack, design patterns, and scalability considerations
- **[DATA_FLOW.md](docs/DATA_FLOW.md)** - Complete interview workflow, data flow diagrams, and sequence diagrams with Mermaid visualizations

### Technical Documentation

- **[BACKEND.md](docs/BACKEND.md)** - Backend module documentation, LangGraph state machine, API endpoints, and core services
- **[FRONTEND.md](docs/FRONTEND.md)** - Frontend component documentation, custom hooks, audio processing, and streaming implementation
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API reference for REST, WebSocket, and SSE protocols

### Operations Documentation

- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide, infrastructure requirements, security hardening, and troubleshooting

---

## Configuration

### Backend Environment Variables

Create `backend/.env` from `backend/.env.example`:

```env
# Core API Keys (REQUIRED)
GEMINI_API_KEY=your_gemini_api_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
RUNPOD_API_KEY=your_runpod_api_key

# RunPod Transcription Settings
WHISPERLIVE_MODEL=small              # small (dev), large-v3 (prod)
WHISPERLIVE_LANGUAGE=en
WHISPERLIVE_TRANSCRIPTION_INTERVAL=3.0
RUNPOD_MAX_BUFFER_SECONDS=30

# TTS Configuration
TTS_DEVICE=auto                      # auto, cuda, cpu
KOKORO_VOICE=af_bella                # af_bella, af_sarah, am_adam, am_michael
KOKORO_SPEED=1.0

# LLM Performance Optimization
GEMINI_THINKING_BUDGET=0             # 0=fastest, 1024+=highest quality
GEMINI_MAX_OUTPUT_TOKENS=1024
GEMINI_TEMPERATURE=0.7
```

### Frontend Configuration

Edit `frontend/src/config.js`:

```javascript
export const API_BASE_URL = "http://localhost:8000";
export const WS_BASE_URL = "ws://localhost:8000";
```

---

## Usage Example

### 1. Upload Documents

Navigate to http://localhost:5173 and upload:
- **Resume** (PDF or DOCX)
- **Job Description** (PDF or DOCX)
- **Custom Instructions** (optional guidance for interview)

### 2. Start Interview

Click "Start Interview" to begin. The system will:
- Analyze your documents
- Create a personalized interview strategy
- Generate the first question with audio

### 3. Conduct Interview

For each question:
1. Click **"Start Recording"** to capture your answer
2. Speak your response (live transcription displayed)
3. Click **"Stop Recording"** when finished
4. System evaluates your answer and generates next question
5. Text and audio stream in real-time

### 4. Interview Conclusion

The interview automatically concludes when:
- 45 minutes elapsed
- 16 questions asked
- System generates a warm conclusion message

---

## API Endpoints

### REST API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload-documents` | POST | Upload resume and job description |
| `/start-interview` | POST | Initialize interview session |
| `/stream-answer` | GET | Submit answer and stream response (SSE) |
| `/submit-answer` | POST | Submit answer (legacy, non-streaming) |
| `/interview-status/{session_id}` | GET | Check interview progress |

### WebSocket

| Endpoint | Purpose |
|----------|---------|
| `/ws/transcribe/{session_id}` | Real-time audio transcription |

**Full API documentation**: [API_REFERENCE.md](docs/API_REFERENCE.md)

---

## Performance

### Typical Latencies

| Operation | Latency |
|-----------|---------|
| Document Analysis | 2-5 seconds (one-time) |
| Question Generation | 1-3 seconds |
| Transcription Cycle | 1-3 seconds (every 3 seconds) |
| TTS Generation | 1-2 seconds |
| **Total Round-Trip** | **7-12 seconds** (stop → audio playback) |

### Optimization Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| LLM Latency | 3-5s | 1-3s | 30-60% faster |
| LLM Cost | High | Low | 600% reduction |
| Time-to-First-Token | 2-3s | 0.5-1.5s | 50% faster |

**Details**: See [Performance Tuning](docs/DEPLOYMENT.md#performance-tuning)

---

## Development

### Backend Development

```bash
cd backend

# Run with auto-reload
uvicorn main:app --reload

# Run tests (if implemented)
pytest

# Check code quality
flake8 .
black .
```

### Frontend Development

```bash
cd frontend

# Start dev server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint
```

---

## Production Deployment

### Backend Deployment

**Using systemd + nginx** (recommended):

```bash
# Create systemd service
sudo systemctl enable xquizit-backend
sudo systemctl start xquizit-backend

# Configure nginx reverse proxy
sudo nano /etc/nginx/sites-available/xquizit
sudo systemctl restart nginx

# Setup SSL with Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

**Using Docker**:

```bash
docker-compose up -d
```

### Frontend Deployment

**Static hosting** (nginx, Vercel, Netlify):

```bash
npm run build
# Deploy dist/ directory
```

**Full deployment guide**: [DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## External Service Setup

### 1. Google Gemini API

1. Visit https://aistudio.google.com/app/apikey
2. Create API key
3. Add to `backend/.env`: `GEMINI_API_KEY=...`

**Pricing**: ~$0.05-$0.15 per interview

### 2. RunPod Serverless (Whisper)

1. Create account at https://runpod.io
2. Deploy **Faster Whisper** endpoint
3. Copy **Endpoint ID** and **API Key**
4. Add to `backend/.env`:
   ```env
   RUNPOD_ENDPOINT_ID=your-endpoint-id
   RUNPOD_API_KEY=your-api-key
   ```

**Pricing**: ~$0.50-$1.00 per 30-minute interview

**Setup guide**: [DEPLOYMENT.md#external-service-setup](docs/DEPLOYMENT.md#external-service-setup)

---

## Architecture Highlights

### LangGraph State Machine

The interview logic is orchestrated using a **LangGraph state machine**:

```
START → analyze_documents → generate_question → END

User Answer → process_answer → check_time → generate_question OR conclude_interview → END
```

**Nodes**:
- **analyze_documents**: Parse resume/JD, create strategy
- **generate_question**: Generate next question
- **process_answer**: Evaluate answer, decide follow-up
- **check_time**: Monitor time/question limits
- **conclude_interview**: Generate conclusion

**State**: Shared `InterviewState` with message history, strategy, and tracking data

**Details**: [BACKEND.md#langgraph-state-machine](docs/BACKEND.md#interview_graphpy---langgraph-state-machine)

---

### Real-Time Transcription

**Flow**:
1. Frontend captures audio (16kHz mono, int16 PCM)
2. WebSocket streams to backend (~256ms chunks)
3. Backend buffers and sends to RunPod every 3 seconds
4. RunPod returns transcription segments
5. Frontend displays live transcription
6. On stop, backend flushes remaining audio
7. Frontend auto-submits final transcript

**Details**: [DATA_FLOW.md#real-time-transcription-flow](docs/DATA_FLOW.md#2-real-time-transcription-flow-runpod-websocket)

---

### Streaming Responses

**SSE Architecture**:
- Backend streams text chunks as LLM generates
- Backend generates TTS audio for complete sentences
- Backend sends audio chunks via SSE events
- Frontend displays text with typing cursor
- Frontend queues audio for seamless playback

**Event Types**: `text_chunk`, `audio_chunk`, `metadata`, `done`

**Details**: [DATA_FLOW.md#streaming-response-flow](docs/DATA_FLOW.md#3-streaming-response-flow-sse)

---

## Security Considerations

### Development Mode

⚠️ **Current configuration is for development only**:
- CORS allows all origins (`allow_origins=["*"]`)
- No authentication required
- HTTP/WS (not HTTPS/WSS)
- No rate limiting

### Production Recommendations

✅ **Before production deployment**:
- Implement JWT-based authentication
- Restrict CORS to allowed domains
- Enable HTTPS/WSS with SSL certificates
- Add rate limiting (e.g., slowapi)
- Use environment-specific secret managers
- Implement session validation

**Full security guide**: [DEPLOYMENT.md#security-hardening](docs/DEPLOYMENT.md#security-hardening)

---

## Troubleshooting

### Common Issues

#### Backend won't start
```bash
# Check Python version (must be < 3.13)
python --version

# Verify dependencies
pip install -r requirements.txt

# Check environment variables
cat .env
```

#### WebSocket connection fails
```bash
# Test WebSocket directly
wscat -c ws://localhost:8000/ws/transcribe/test-session

# Check nginx WebSocket config
sudo nginx -t
```

#### TTS errors
```bash
# Check TTS device config
echo $TTS_DEVICE

# Verify Kokoro model downloaded
ls ~/.cache/realtimetts/
```

**Full troubleshooting guide**: [DEPLOYMENT.md#troubleshooting](docs/DEPLOYMENT.md#troubleshooting)

---

## Project Structure

```
xquizit/
├── backend/
│   ├── main.py                          # FastAPI application
│   ├── interview_graph_v3.py            # LangGraph workflow (V3 architecture)
│   ├── agents/                          # Specialized agent modules
│   ├── prompts/                         # Prompt templates and schemas
│   ├── models.py                        # Pydantic models
│   ├── document_processor.py            # PDF/DOCX extraction
│   ├── tts_service.py                   # Text-to-speech
│   ├── runpod_transcription_service.py  # Transcription service
│   ├── timing_utils.py                  # Performance instrumentation
│   ├── requirements.txt                 # Python dependencies
│   └── .env.example                     # Environment template
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Root component
│   │   ├── components/
│   │   │   ├── UploadScreen.jsx         # Document upload UI
│   │   │   ├── ChatInterface.jsx        # Main interview UI
│   │   │   ├── AudioRecorder.jsx        # Audio recording
│   │   │   ├── StreamingMessage.jsx     # Real-time message
│   │   │   ├── Message.jsx              # Static message
│   │   │   └── Timer.jsx                # Interview timer
│   │   ├── hooks/
│   │   │   └── useAudioPlayback.js      # Audio queue hook
│   │   └── config.js                    # API configuration
│   ├── package.json
│   └── vite.config.js
│
├── docs/
│   ├── ARCHITECTURE.md                  # System architecture
│   ├── BACKEND.md                       # Backend modules
│   ├── FRONTEND.md                      # Frontend components
│   ├── DATA_FLOW.md                     # Workflow diagrams
│   ├── DEPLOYMENT.md                    # Deployment guide
│   └── API_REFERENCE.md                 # API documentation
│
├── CLAUDE.md                            # Claude Code instructions
└── README.md                            # This file
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LangGraph** and **LangChain** for workflow orchestration
- **Google Gemini** for powerful LLM capabilities
- **RunPod** for serverless Whisper transcription
- **RealtimeTTS** and **Kokoro** for natural TTS synthesis
- **FastAPI** for the excellent async web framework
- **React** and **Vite** for modern frontend development

---

## Support

For issues, questions, or feature requests:

- **Documentation**: See [docs/](docs/) directory
- **API Reference**: [API_REFERENCE.md](docs/API_REFERENCE.md)
- **Deployment Help**: [DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Troubleshooting**: [DEPLOYMENT.md#troubleshooting](docs/DEPLOYMENT.md#troubleshooting)

---

## Roadmap

### Short-Term
- [ ] Persistent session storage (Redis)
- [ ] JWT-based authentication
- [ ] Rate limiting implementation
- [ ] Interview analytics dashboard

### Medium-Term
- [ ] Multi-tenancy support
- [ ] Custom interview templates
- [ ] Post-interview reports
- [ ] Video streaming support

### Long-Term
- [ ] Microservices architecture
- [ ] Event-driven processing (Kafka)
- [ ] Custom ML model training
- [ ] Global multi-region deployment

---

**Built with ❤️ for better interview experiences**

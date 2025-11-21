"""
FastAPI Main Application
Backend server for the screening interview chatbot.
"""

import os
import uuid
import logging
import time
import tempfile
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
import requests
from langchain_core.messages import HumanMessage, AIMessage

from app.config import Settings
from app.core.models import (
    SessionData,
    UploadDocumentsResponse,
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    InterviewStatusResponse,
    ErrorResponse,
    Message,
    TranscriptionSegment,
    TranscriptionMessage,
)
from app.services.document_processor import extract_text_from_document, DocumentProcessingError
from app.core.graph import create_interview_graph_v3, InterviewGraphBuilderV3
from app.services.tts_service import initialize_tts_service, get_tts_service, cleanup_tts_service
from app.services.transcription_service import (
    initialize_runpod_service,
    get_runpod_service,
    cleanup_runpod_service,
)
from app.services.local_whisper_service import (
    initialize_local_whisper_service,
    get_local_whisper_service,
    cleanup_local_whisper_service,
)
from app.services.base_transcription_service import TranscriptionService
from app.utils.timing import time_operation, TimingSummary
from app.utils.logging_config import setup_logging, get_logger
from app.utils.metrics import (
    active_sessions,
    record_session_metrics,
    http_requests_total,
    http_request_duration_seconds
)

# Configure logging (clean format by default, JSON for production via env var)
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=os.getenv("LOG_FORMAT", "").lower() == "json",  # Only JSON if explicitly set
    transcription_logging_enabled=os.getenv("TRANSCRIPTION_LOGGING_ENABLED", "true").lower() == "true"
)
logger = logging.getLogger(__name__)


# Global state
settings: Optional[Settings] = None
interview_graph: Optional[InterviewGraphBuilderV3] = None
sessions: Dict[str, SessionData] = {}
transcription_service: Optional[TranscriptionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global settings, interview_graph, transcription_service
    try:
        settings = Settings()
        logger.info("Settings loaded successfully")

        # Initialize interview graph with Gemini V3 (Hybrid Modular State Machine)
        interview_graph = create_interview_graph_v3(
            gemini_api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            thinking_budget=settings.gemini_thinking_budget,
            include_thoughts=settings.gemini_include_thoughts,
            max_output_tokens=settings.gemini_max_output_tokens,
            temperature=settings.gemini_temperature
        )
        logger.info("✓ Interview graph initialized with Gemini")
        logger.info("  → Hybrid Modular State Machine Architecture")
        logger.info("  → Features: Conversational turn handling + Strategic time allocation + Quality-driven follow-ups")

        # Initialize TTS service using factory pattern
        tts_backend = settings.tts_backend.lower()
        logger.info(f"Initializing TTS service (backend={tts_backend})...")

        try:
            if tts_backend == "kokoro":
                initialize_tts_service(
                    backend="kokoro",
                    device=settings.tts_device,
                    voice=settings.kokoro_voice,
                    speed=settings.kokoro_speed
                )
                logger.info("✓ TTS service initialized successfully with Kokoro")
                logger.info(f"  - Device: {settings.tts_device}")
                logger.info(f"  - Voice: {settings.kokoro_voice}")
                logger.info(f"  - Speed: {settings.kokoro_speed}")

            elif tts_backend == "websocket":
                initialize_tts_service(
                    backend="websocket",
                    server_url=settings.tts_server_url
                )
                logger.info("✓ TTS service initialized successfully with WebSocket")
                logger.info(f"  - Server URL: {settings.tts_server_url}")
                logger.info("  - Note: Will connect to server on first use")

            else:
                logger.error(f"✗ Unknown TTS backend: {tts_backend}")
                logger.error("  Valid options: 'kokoro', 'websocket'")
                logger.warning("  → Streaming will be text-only (no audio)")

        except ImportError as e:
            logger.error(f"✗ TTS backend '{tts_backend}' dependencies not installed!")
            logger.error(f"  Error: {str(e)}")
            if tts_backend == "kokoro":
                logger.error("  Please install with: pip install realtimetts[kokoro]")
            elif tts_backend == "websocket":
                logger.error("  Please install with: pip install websockets")
            logger.warning("  → Streaming will be text-only (no audio)")

        except Exception as e:
            logger.error(f"✗ Failed to initialize TTS service: {str(e)}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.warning("  → Streaming will be text-only (no audio)")
            import traceback
            logger.debug(traceback.format_exc())

        # Initialize transcription service (RunPod or Local WhisperLive)
        backend = settings.transcription_backend.lower()
        logger.info(f"Initializing transcription service (backend={backend})...")

        try:
            if backend == "runpod":
                # Initialize RunPod transcription service
                if not settings.runpod_endpoint_id or not settings.runpod_api_key:
                    raise ValueError("RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY are required for RunPod backend")

                transcription_service = initialize_runpod_service(
                    runpod_endpoint_id=settings.runpod_endpoint_id,
                    runpod_api_key=settings.runpod_api_key,
                    lang=settings.whisperlive_language,
                    model=settings.whisperlive_model,
                    use_vad=settings.whisperlive_use_vad,
                    no_speech_thresh=settings.whisperlive_no_speech_thresh,
                    same_output_threshold=settings.whisperlive_same_output_threshold,
                    transcription_interval=settings.whisperlive_transcription_interval,
                    max_buffer_seconds=settings.runpod_max_buffer_seconds,
                    transcription_logging_enabled=settings.transcription_logging_enabled
                )
                logger.info("✓ RunPod transcription service initialized successfully")
                logger.info(f"  - Endpoint: {settings.runpod_endpoint_id[:8]}...")
                logger.info(f"  - Model: {settings.whisperlive_model}")
                logger.info(f"  - Language: {settings.whisperlive_language}")
                logger.info(f"  - VAD enabled: {settings.whisperlive_use_vad}")

            elif backend == "local":
                # Initialize local WhisperLive transcription service
                if not settings.whisperlive_server_url and not settings.whisperlive_server_host:
                    raise ValueError("Either WHISPERLIVE_SERVER_URL or WHISPERLIVE_SERVER_HOST is required for local backend")

                transcription_service = initialize_local_whisper_service(
                    server_url=settings.whisperlive_server_url,
                    server_host=settings.whisperlive_server_host,
                    server_port=settings.whisperlive_server_port,
                    use_ssl=settings.whisperlive_use_ssl,
                    lang=settings.whisperlive_language,
                    model=settings.whisperlive_model,
                    use_vad=settings.whisperlive_use_vad,
                    no_speech_thresh=settings.whisperlive_no_speech_thresh,
                    send_last_n_segments=settings.whisperlive_send_last_n_segments,
                    same_output_threshold=settings.whisperlive_same_output_threshold,
                    clip_audio=settings.whisperlive_clip_audio,
                    chunking_mode=settings.whisperlive_chunking_mode,
                    chunk_interval=settings.whisperlive_chunk_interval,
                    enable_translation=settings.whisperlive_enable_translation,
                    target_language=settings.whisperlive_target_language,
                    transcription_logging_enabled=settings.transcription_logging_enabled
                )
                logger.info("✓ Local WhisperLive transcription service initialized successfully")
                if settings.whisperlive_server_url:
                    logger.info(f"  - Server URL: {settings.whisperlive_server_url}")
                else:
                    logger.info(f"  - Server: {settings.whisperlive_server_host}:{settings.whisperlive_server_port}")
                logger.info(f"  - Protocol: {'wss' if settings.whisperlive_use_ssl else 'ws'}://")
                logger.info(f"  - Model: {settings.whisperlive_model}")
                logger.info(f"  - Language: {settings.whisperlive_language}")
                logger.info(f"  - VAD enabled: {settings.whisperlive_use_vad}")

            else:
                raise ValueError(f"Unknown transcription backend: {backend}. Must be 'runpod' or 'local'")

        except Exception as e:
            logger.error(f"✗ Failed to initialize transcription service: {str(e)}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.warning("  → Real-time transcription will not be available")
            if backend == "runpod":
                logger.warning("  → Ensure RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY are configured correctly")
            else:
                logger.warning("  → Ensure WHISPERLIVE_SERVER_HOST is configured and server is running")
            import traceback
            logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

    yield

    # Shutdown - cleanup resources
    logger.info("Shutting down application...")
    try:
        # Clear session data
        sessions.clear()
        logger.info("Cleared session data")

        # Cleanup transcription service (RunPod or Local WhisperLive)
        try:
            if transcription_service:
                await transcription_service.close_all()
                transcription_service = None
                logger.info("Transcription service cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up transcription service: {str(e)}")

        # Cleanup TTS service
        cleanup_tts_service()
        logger.info("TTS service cleaned up")

        # Set globals to None for cleanup
        interview_graph = None
        settings = None

        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Screening Interview Chatbot API",
    description="Backend API for conducting automated screening interviews",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for network access (allows all origins for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - simple API info."""
    return {
        "message": "Screening Interview Chatbot API",
        "status": "operational",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.

    Returns:
        Health status with component checks
    """
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "components": {
            "interview_graph": interview_graph is not None,
            "tts_service": get_tts_service() is not None,
            "transcription_service": get_runpod_service() is not None,
        },
        "metrics": {
            "active_sessions": len(sessions),
            "total_sessions": len(sessions)
        }
    }

    # Check if critical components are initialized
    all_healthy = all(health_status["components"].values())

    if not all_healthy:
        health_status["status"] = "degraded"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )

    return health_status


@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        Text-formatted Prometheus metrics
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    metrics_output = generate_latest()
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


@app.post(
    "/upload-documents",
    response_model=UploadDocumentsResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_documents(
    resume: UploadFile = File(..., description="Resume file (PDF or DOCX)"),
    job_description: UploadFile = File(..., description="Job description file (PDF or DOCX)"),
    custom_instructions: str = Form(None, description="Optional custom instructions for interview strategy")
):
    """
    Upload resume and job description documents to create a new interview session.

    Args:
        resume: Resume file (PDF or DOCX format)
        job_description: Job description file (PDF or DOCX format)
        custom_instructions: Optional custom instructions to guide interview strategy

    Returns:
        Session ID and document processing confirmation
    """
    logger.info(f"Received document upload request - Resume: {resume.filename}, JD: {job_description.filename}, Custom Instructions: {'Yes' if custom_instructions else 'No'}")

    temp_files = []
    try:
        # Validate file types
        resume_ext = Path(resume.filename).suffix.lower()
        jd_ext = Path(job_description.filename).suffix.lower()

        allowed_extensions = {'.pdf', '.docx', '.doc'}
        if resume_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resume file type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        if jd_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job description file type not supported. Allowed: {', '.join(allowed_extensions)}"
            )

        # Save resume to temporary file
        resume_temp = tempfile.NamedTemporaryFile(delete=False, suffix=resume_ext)
        temp_files.append(resume_temp.name)
        resume_content = await resume.read()
        resume_temp.write(resume_content)
        resume_temp.close()

        # Save job description to temporary file
        jd_temp = tempfile.NamedTemporaryFile(delete=False, suffix=jd_ext)
        temp_files.append(jd_temp.name)
        jd_content = await job_description.read()
        jd_temp.write(jd_content)
        jd_temp.close()

        # Extract text from documents
        resume_text = extract_text_from_document(resume_temp.name)
        jd_text = extract_text_from_document(jd_temp.name)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Create session data
        session_data = SessionData(
            session_id=session_id,
            resume_text=resume_text,
            job_description_text=jd_text,
            custom_instructions=custom_instructions or ""  # Default to empty string
        )

        # Store session
        sessions[session_id] = session_data

        # Log custom instructions details
        if custom_instructions:
            logger.info(f"Created session {session_id} - Resume: {len(resume_text)} chars, JD: {len(jd_text)} chars, Custom Instructions: {len(custom_instructions)} chars")
            logger.info(f"Custom instructions preview: {custom_instructions[:100]}..." if len(custom_instructions) > 100 else f"Custom instructions: {custom_instructions}")
        else:
            logger.info(f"Created session {session_id} - Resume: {len(resume_text)} chars, JD: {len(jd_text)} chars, No custom instructions")

        return UploadDocumentsResponse(
            session_id=session_id,
            message="Documents uploaded and processed successfully",
            resume_length=len(resume_text),
            job_description_length=len(jd_text),
            has_custom_instructions=bool(custom_instructions),
            custom_instructions_length=len(custom_instructions) if custom_instructions else 0
        )

    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in upload_documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")


@app.post(
    "/start-interview",
    response_model=StartInterviewResponse,
    status_code=status.HTTP_200_OK
)
async def start_interview(request: StartInterviewRequest):
    """
    Initialize an interview with the given session ID and return the first question.

    Args:
        request: Request containing session_id

    Returns:
        First interview question
    """
    # Create timing summary for this request
    timing_summary = TimingSummary(session_id=request.session_id)

    with time_operation("ENDPOINT - Start Interview (Total)", log_result=False) as endpoint_timing:
        logger.info(f"Starting interview for session {request.session_id}")

        # Validate session exists
        if request.session_id not in sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {request.session_id} not found"
            )

        session = sessions[request.session_id]

        # Check if interview already started
        if session.start_time is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Interview has already been started for this session"
            )

        try:
            # Initialize interview state
            from datetime import datetime
            session.start_time = datetime.now()

            # Log custom instructions usage
            if session.custom_instructions:
                logger.info(f"Starting interview with custom instructions ({len(session.custom_instructions)} chars)")
            else:
                logger.info("Starting interview without custom instructions")

            initial_state = {
                "session_id": request.session_id,
                "resume_text": session.resume_text,
                "job_description_text": session.job_description_text,
                "custom_instructions": session.custom_instructions or "",  # Ensure it's never None
                "messages": [],  # Start empty - graph will add first question
                # V3 fields
                "interview_time_strategy": "",  # JSON string from strategic_planner
                "key_topics": [],
                "questions_asked": 0,
                "current_question": None,
                "current_topic": None,
                # V3 state tracking
                "topic_statistics": {},
                "topics_completed": [],
                "critical_topics_covered": [],
                # V3 transient fields
                "last_turn_analysis": None,
                "pending_clarification": False,
                "pending_continuation": False,
                "current_topic_quality": None,
                # Timing
                "start_time": time.time(),
                "time_elapsed": 0.0,
                "is_concluded": False,
                "conclusion_reason": None
            }

            # Run the graph to generate first question
            result = interview_graph.invoke(initial_state)

            # Update session with V3 results
            session.interview_time_strategy = result.get("interview_time_strategy", "")
            session.key_topics = result.get("key_topics", [])
            session.questions_asked = result.get("questions_asked", 0)
            session.topic_statistics = result.get("topic_statistics", {})
            session.topics_completed = result.get("topics_completed", [])
            # critical_topics_covered is tracked in topic_statistics, not as separate session field
            # Persist current state for next invocation
            session.current_topic = result.get("current_topic")
            session.current_question = result.get("current_question")

            # Extract the first question
            first_question = result.get("current_question", "")

            # Add to conversation history
            if first_question:
                session.conversation_history.append(Message(
                    role="assistant",
                    content=first_question
                ))

            # Generate TTS audio for first question
            audio_chunks = []
            if first_question:
                try:
                    logger.info(f"Generating TTS audio for first question: '{first_question[:50]}...'")
                    tts_service = get_tts_service()

                    # Generate audio chunks
                    async for audio_data in tts_service.generate_stream(first_question, audio_index=0):
                        if audio_data.get("type") == "audio_chunk":
                            audio_chunks.append(audio_data["audio"])

                    logger.info(f"Generated {len(audio_chunks)} audio chunks for first question")

                except Exception as e:
                    logger.error(f"Error generating TTS audio for first question: {str(e)}")
                    # Continue without audio - not a fatal error
                    audio_chunks = []

            logger.info(f"Interview started for session {request.session_id}")

            # Log endpoint timing
            endpoint_timing.metadata["audio_chunks"] = len(audio_chunks)
            endpoint_timing.metadata["question_length"] = len(first_question)
            timing_summary.log_summary()

            return StartInterviewResponse(
                session_id=request.session_id,
                first_question=first_question,
                message="Interview started successfully",
                audio_chunks=audio_chunks if audio_chunks else None
            )

        except Exception as e:
            logger.error(f"Error starting interview: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start interview: {str(e)}"
            )


@app.websocket("/ws/transcribe/{session_id}")
async def websocket_transcribe(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time audio transcription.

    Supports multiple backends:
    - RunPod: Serverless transcription via HTTP API
    - Local WhisperLive: WebSocket connection to local server

    Flow:
    1. Accept WebSocket connection from frontend
    2. Validate session_id exists
    3. Create transcription session (backend determined by config)
    4. Receive audio chunks from frontend (binary, int16 PCM, 16kHz)
    5. Forward audio to transcription backend
    6. Stream transcription segments back to frontend (JSON)
    7. Handle disconnection and cleanup

    Args:
        websocket: WebSocket connection instance
        session_id: Interview session identifier

    Message Format (Backend -> Frontend):
        {
            "type": "transcript",
            "text": "Hello, I have experience",
            "is_final": false,
            "start": 0.0,
            "end": 2.5
        }

    Message Format (Frontend -> Backend):
        Binary audio data (int16 PCM, 16kHz, mono)
    """
    logger.info(f"WebSocket transcription request for session {session_id}")

    # Validate session exists
    if session_id not in sessions:
        await websocket.close(code=4004, reason=f"Session {session_id} not found")
        return

    # Accept WebSocket connection
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session {session_id}")

    transcription_session = None

    try:
        # Get transcription service (RunPod or Local WhisperLive)
        if not transcription_service:
            raise RuntimeError("Transcription service not initialized")

        # Define callback for transcription updates
        async def transcription_callback(segment_data: dict):
            """Send transcription segments to frontend via WebSocket."""
            try:
                message = TranscriptionMessage(
                    type=segment_data["type"],
                    text=segment_data.get("text"),
                    segment=TranscriptionSegment(
                        text=segment_data.get("text", ""),
                        start=segment_data.get("start", 0.0),
                        end=segment_data.get("end", 0.0),
                        is_final=segment_data.get("is_final", False)
                    ) if segment_data["type"] == "transcript" else None,
                    error=segment_data.get("error")
                )
                await websocket.send_json(message.model_dump())
            except Exception as e:
                logger.error(f"Error sending transcription to WebSocket: {str(e)}")

        # Create transcription session with callback
        transcription_session = transcription_service.create_session(
            session_id=session_id,
            fastapi_websocket=websocket,
            callback=lambda data: asyncio.create_task(transcription_callback(data))
        )

        # Connect to transcription backend
        if not await transcription_session.connect():
            await websocket.close(code=4503, reason="Failed to connect to transcription service")
            return

        logger.info(f"Transcription session connected for {session_id}")

        # Send ready message to frontend
        await websocket.send_json({
            "type": "ready",
            "message": "Transcription service ready"
        })

        # Track whether final flush was completed (to avoid redundant cleanup)
        final_flush_completed = False

        # Receive and forward audio chunks
        chunk_count = 0
        while True:
            # Receive message (binary audio or JSON control message)
            message = await websocket.receive()

            # Handle binary audio data
            if 'bytes' in message:
                audio_data = message['bytes']
                chunk_count += 1

                # Log periodically (every 10 chunks to avoid spam)
                if chunk_count % 10 == 0:
                    logger.debug(f"Session {session_id}: Received chunk #{chunk_count} ({len(audio_data)} bytes)")

                # Forward to transcription backend
                await transcription_session.send_audio(audio_data)

            # Handle JSON control messages
            elif 'text' in message:
                try:
                    control_msg = json.loads(message['text'])
                    msg_type = control_msg.get('type')

                    if msg_type == 'stop_recording':
                        logger.info(f"⏹️  Received stop_recording signal for session {session_id}")
                        # Trigger final flush without closing WebSocket
                        await transcription_session.trigger_final_flush()
                        logger.info(f"✅ Final flush completed for session {session_id}")
                        final_flush_completed = True
                        break  # Exit receive loop - frontend will close WebSocket
                    else:
                        logger.warning(f"Unknown control message type: {msg_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse control message: {e}")

            else:
                logger.warning(f"Received unknown message type from session {session_id}")

    except WebSocketDisconnect:
        disconnect_time = time.time()
        logger.info(f"⏱️  WebSocket disconnected for session {session_id} at {disconnect_time}")

    except RuntimeError as e:
        logger.error(f"Transcription service not initialized: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "error": "Transcription service not available"
        })
        await websocket.close(code=4503, reason="Service not available")

    except Exception as e:
        logger.error(f"Error in WebSocket transcription: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
        await websocket.close(code=4500, reason="Internal error")

    finally:
        # Cleanup transcription session
        if transcription_session:
            try:
                if not final_flush_completed:
                    # Final flush NOT done (disconnect without stop_recording signal)
                    cleanup_start = time.time()
                    logger.info(f"⏱️  Starting cleanup for session {session_id} at {cleanup_start}")

                    # Close session via service (includes final flush)
                    await transcription_service.close_session(session_id)

                    cleanup_end = time.time()
                    cleanup_duration = cleanup_end - cleanup_start
                    logger.info(f"⏱️  [FINAL FLUSH COMPLETE] Session {session_id} - Duration: {cleanup_duration:.3f}s")
                    logger.info(f"Transcription session closed and removed for {session_id}")
                else:
                    # Final flush already done - just close session
                    logger.info(f"Final flush already completed for session {session_id}, closing session")
                    await transcription_session.close()
                    transcription_service.sessions.pop(session_id, None)
                    logger.info(f"Transcription session closed for {session_id}")

                # Close WebSocket connection
                try:
                    await websocket.close()
                    logger.info(f"WebSocket closed for session {session_id}")
                except Exception as e:
                    logger.warning(f"Error closing WebSocket for {session_id}: {e}")

            except Exception as e:
                logger.error(f"Error in cleanup: {str(e)}")
                # Still try to close WebSocket even if cleanup failed
                try:
                    await websocket.close()
                except:
                    pass
        else:
            # No transcription session, just close WebSocket
            try:
                await websocket.close()
            except:
                pass


@app.post(
    "/submit-answer",
    response_model=SubmitAnswerResponse,
    status_code=status.HTTP_200_OK
)
async def submit_answer(request: SubmitAnswerRequest):
    """
    Submit candidate's answer and receive the next question.

    Args:
        request: Request containing session_id and answer text

    Returns:
        Next question or interview conclusion
    """
    logger.info(f"Received answer submission for session {request.session_id}")

    # Validate session exists
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        )

    session = sessions[request.session_id]

    # Check if interview has been started
    if session.start_time is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview has not been started yet. Call /start-interview first."
        )

    # Check if interview is already concluded
    if session.is_concluded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview has already been concluded"
        )

    try:
        # Add user answer to conversation history
        session.conversation_history.append(Message(
            role="user",
            content=request.answer
        ))

        # Build messages for graph state using LangChain message objects
        # CRITICAL: add_messages reducer expects HumanMessage/AIMessage objects, not plain dicts
        messages = []
        for msg in session.conversation_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            # Skip any other role types (e.g., "system" if present)

        # Diagnostic logging: verify messages are properly constructed
        logger.info(
            f"[{request.session_id}] Built {len(messages)} messages for graph: "
            f"{sum(1 for m in messages if isinstance(m, HumanMessage))} user, "
            f"{sum(1 for m in messages if isinstance(m, AIMessage))} assistant"
        )

        # Calculate time elapsed
        from datetime import datetime
        time_elapsed = (datetime.now() - session.start_time).total_seconds()

        # Create state for graph (V3 fields)
        current_state = {
            "session_id": request.session_id,
            "resume_text": session.resume_text,
            "job_description_text": session.job_description_text,
            "custom_instructions": session.custom_instructions or "",  # Ensure it's never None
            "messages": messages,
            # V3 fields from session
            "interview_time_strategy": session.interview_time_strategy or "",  # CRITICAL for routing
            "key_topics": session.key_topics,
            "questions_asked": session.questions_asked,
            "current_question": session.current_question,  # Restore from session
            "current_topic": session.current_topic,  # Restore from session
            # V3 state tracking from session
            "topic_statistics": getattr(session, 'topic_statistics', {}),
            "topics_completed": getattr(session, 'topics_completed', []),
            "critical_topics_covered": [],  # Derived from topic_statistics
            # V3 transient fields (reset between turns)
            "last_turn_analysis": None,
            "pending_clarification": False,
            "pending_continuation": False,
            "current_topic_quality": None,
            # Timing
            "start_time": session.start_time.timestamp(),
            "time_elapsed": time_elapsed,
            "is_concluded": False,
            "conclusion_reason": None
        }

        # Process through graph
        result = interview_graph.invoke(current_state)

        # Update session with V3 fields
        session.questions_asked = result.get("questions_asked", session.questions_asked)
        session.topic_statistics = result.get("topic_statistics", getattr(session, 'topic_statistics', {}))
        session.topics_completed = result.get("topics_completed", getattr(session, 'topics_completed', []))
        session.is_concluded = result.get("is_concluded", False)
        session.conclusion_reason = result.get("conclusion_reason")
        # Persist current state for next invocation
        session.current_topic = result.get("current_topic")
        session.current_question = result.get("current_question")

        # Get next question or conclusion
        next_question = result.get("current_question")

        # Add assistant response to history
        if next_question:
            session.conversation_history.append(Message(
                role="assistant",
                content=next_question
            ))

        # Calculate time remaining
        max_time = 45 * 60  # 45 minutes in seconds
        time_remaining = max(0, max_time - time_elapsed)

        logger.info(f"Processed answer for session {request.session_id}, concluded: {session.is_concluded}")

        return SubmitAnswerResponse(
            session_id=request.session_id,
            next_question=next_question if not session.is_concluded else None,
            is_concluded=session.is_concluded,
            conclusion_message=next_question if session.is_concluded else None,
            time_remaining_seconds=time_remaining
        )

    except Exception as e:
        logger.error(f"Error processing answer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process answer: {str(e)}"
        )


@app.get(
    "/stream-answer",
    status_code=status.HTTP_200_OK
)
async def stream_answer(session_id: str, answer: str):
    """
    Submit candidate's answer and stream the next question with real-time TTS audio.

    This endpoint uses Server-Sent Events (SSE) to stream:
    1. Text chunks as the LLM generates them
    2. Audio chunks as TTS processes complete sentences
    3. Metadata when streaming completes

    Args:
        session_id: Interview session identifier (query parameter)
        answer: Candidate's answer text (query parameter)

    Returns:
        StreamingResponse with SSE events
    """
    logger.info(f"Received streaming answer submission for session {session_id}")

    # Validate session exists
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    session = sessions[session_id]

    # Check if interview has been started
    if session.start_time is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview has not been started yet. Call /start-interview first."
        )

    # Check if interview is already concluded
    if session.is_concluded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview has already been concluded"
        )

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events for streaming response."""
        try:
            # Add user answer to conversation history
            session.conversation_history.append(Message(
                role="user",
                content=answer
            ))

            # Build messages for graph state using LangChain message objects
            # CRITICAL: add_messages reducer expects HumanMessage/AIMessage objects, not plain dicts
            messages = []
            for msg in session.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
                # Skip any other role types (e.g., "system" if present)

            # Diagnostic logging: verify messages are properly constructed
            logger.info(
                f"[{session_id}] Built {len(messages)} messages for graph: "
                f"{sum(1 for m in messages if isinstance(m, HumanMessage))} user, "
                f"{sum(1 for m in messages if isinstance(m, AIMessage))} assistant"
            )

            # Calculate time elapsed
            from datetime import datetime
            time_elapsed = (datetime.now() - session.start_time).total_seconds()

            # Create state for graph (V3 fields)
            current_state = {
                "session_id": session_id,
                "resume_text": session.resume_text,
                "job_description_text": session.job_description_text,
                "custom_instructions": session.custom_instructions or "",  # Ensure it's never None
                "messages": messages,
                # V3 fields from session
                "interview_time_strategy": session.interview_time_strategy or "",  # CRITICAL for routing
                "key_topics": session.key_topics,
                "questions_asked": session.questions_asked,
                "current_question": session.current_question,  # Restore from session
                "current_topic": session.current_topic,  # Restore from session
                # V3 state tracking from session
                "topic_statistics": getattr(session, 'topic_statistics', {}),
                "topics_completed": getattr(session, 'topics_completed', []),
                "critical_topics_covered": [],  # Derived from topic_statistics
                # V3 transient fields (reset between turns)
                "last_turn_analysis": None,
                "pending_clarification": False,
                "pending_continuation": False,
                "current_topic_quality": None,
                # Timing
                "start_time": session.start_time.timestamp(),
                "time_elapsed": time_elapsed,
                "is_concluded": False,
                "conclusion_reason": None
            }

            # Process answer through graph synchronously first
            # (This handles _process_answer and routing logic)
            result = interview_graph.invoke(current_state)

            # Update session with V3 fields
            session.questions_asked = result.get("questions_asked", session.questions_asked)
            session.topic_statistics = result.get("topic_statistics", getattr(session, 'topic_statistics', {}))
            session.topics_completed = result.get("topics_completed", getattr(session, 'topics_completed', []))
            session.is_concluded = result.get("is_concluded", False)
            session.conclusion_reason = result.get("conclusion_reason")
            # Persist current state for next invocation
            session.current_topic = result.get("current_topic")
            session.current_question = result.get("current_question")

            # Check if a question was already generated during invoke() (follow-up case)
            existing_question = result.get("current_question")

            # If already concluded, send the conclusion message
            if session.is_concluded:
                conclusion_message = result.get("current_question", "Thank you for your time!")

                # Send conclusion as text chunks (named SSE event)
                yield f"event: text_chunk\ndata: {json.dumps({'chunk': conclusion_message})}\n\n"

                # Try to generate TTS for conclusion
                try:
                    tts_service = get_tts_service()
                    async for audio_event in tts_service.generate_stream(conclusion_message, audio_index=0):
                        # Send as named SSE event
                        yield f"event: audio_chunk\ndata: {json.dumps(audio_event)}\n\n"
                except Exception as e:
                    logger.warning(f"TTS not available for conclusion: {str(e)}")

                # Send metadata (named SSE events)
                yield f"event: metadata\ndata: {json.dumps({'is_concluded': True, 'time_remaining': 0})}\n\n"

                # Allow SSE buffer to flush audio chunks before sending done event
                await asyncio.sleep(0.1)

                yield f"event: done\ndata: {{}}\n\n"

                # Add to conversation history
                session.conversation_history.append(Message(
                    role="assistant",
                    content=conclusion_message
                ))
                return

            # If a question already exists (follow-up case), stream it instead of regenerating
            elif existing_question:
                logger.info(f"Question already generated during invoke, streaming existing: {existing_question[:80]}...")

                # Split into words for streaming effect
                words = existing_question.split()
                sentence_buffer = ""
                audio_index = 0

                for word in words:
                    chunk_text = word + " "
                    sentence_buffer += chunk_text

                    # Send text chunk via SSE
                    yield f"event: text_chunk\ndata: {json.dumps({'chunk': chunk_text})}\n\n"

                    # Small delay for streaming effect
                    await asyncio.sleep(0.05)

                    # Check for complete sentences and generate TTS
                    sentence = _extract_complete_sentence(sentence_buffer)
                    if sentence:
                        sentence_buffer = sentence_buffer[len(sentence):].lstrip()
                        try:
                            tts_service = get_tts_service()
                            async for audio_event in tts_service.generate_stream(sentence, audio_index=audio_index):
                                yield f"event: audio_chunk\ndata: {json.dumps(audio_event)}\n\n"
                            audio_index += 1
                        except Exception as e:
                            logger.warning(f"TTS not available: {str(e)}")

                # Process remaining text in buffer
                if sentence_buffer.strip():
                    try:
                        tts_service = get_tts_service()
                        async for audio_event in tts_service.generate_stream(sentence_buffer.strip(), audio_index=audio_index):
                            yield f"event: audio_chunk\ndata: {json.dumps(audio_event)}\n\n"
                    except Exception as e:
                        logger.warning(f"TTS not available for remaining: {str(e)}")

                # Sync conversation history (check to avoid duplicates - graph may have already added it)
                if not session.conversation_history or session.conversation_history[-1].content != existing_question:
                    session.conversation_history.append(Message(
                        role="assistant",
                        content=existing_question
                    ))

                # Calculate time remaining
                max_time = 45 * 60  # 45 minutes in seconds
                time_remaining = max(0, max_time - time_elapsed)

                # Send metadata
                yield f"event: metadata\ndata: {json.dumps({'is_concluded': False, 'time_remaining': time_remaining})}\n\n"

                # Allow SSE buffer to flush audio chunks before sending done event
                await asyncio.sleep(0.1)

                yield f"event: done\ndata: {{}}\n\n"

                logger.info(f"Streamed existing question successfully for session {session_id}")
                return

            # Stream the next question generation (no existing question - generate new)
            accumulated_text = ""
            audio_index = 0
            question_metadata = None

            async for chunk in interview_graph._generate_question_stream(result):
                if chunk["type"] == "text_chunk":
                    text_content = chunk["content"]
                    accumulated_text += text_content

                    # Send text chunk to client (named SSE event)
                    yield f"event: text_chunk\ndata: {json.dumps({'chunk': text_content})}\n\n"

                    # Send to TTS IMMEDIATELY (no buffering for near-zero latency)
                    try:
                        tts_service = get_tts_service()
                        async for audio_event in tts_service.generate_stream(text_content, audio_index=audio_index):
                            # Send as named SSE event
                            yield f"event: audio_chunk\ndata: {json.dumps(audio_event)}\n\n"
                        audio_index += 1
                    except Exception as e:
                        logger.warning(f"TTS not available: {str(e)}")

                elif chunk["type"] == "question_complete":
                    question_metadata = chunk

            # Update session with question metadata
            if question_metadata:
                final_question = question_metadata["question"]
                current_topic = question_metadata["current_topic"]
                questions_asked = question_metadata["questions_asked"]

                # Update session state
                session.questions_asked = questions_asked
                if question_metadata.get("followup_count", 0) > 0:
                    topic_followup_counts = session.topic_followup_counts
                    topic_followup_counts[current_topic] = question_metadata["followup_count"]
                    session.topic_followup_counts = topic_followup_counts

                # Add to conversation history
                session.conversation_history.append(Message(
                    role="assistant",
                    content=final_question
                ))

            # Calculate time remaining
            max_time = 45 * 60  # 45 minutes in seconds
            time_remaining = max(0, max_time - time_elapsed)

            # Send final metadata (named SSE events)
            yield f"event: metadata\ndata: {json.dumps({'is_concluded': False, 'time_remaining': time_remaining})}\n\n"

            # Allow SSE buffer to flush audio chunks before sending done event
            await asyncio.sleep(0.1)

            yield f"event: done\ndata: {{}}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            error_msg = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


def _extract_complete_sentence(text: str) -> Optional[str]:
    """
    Extract a complete sentence from text buffer.

    Args:
        text: Text buffer to check

    Returns:
        Complete sentence if found, None otherwise
    """
    # Look for sentence-ending punctuation
    match = re.search(r'^(.*?[.!?])\s+', text)
    if match:
        return match.group(1)

    # Check if the entire buffer ends with punctuation (last sentence)
    if text.rstrip() and text.rstrip()[-1] in '.!?':
        return text.rstrip()

    return None


@app.get(
    "/interview-status/{session_id}",
    response_model=InterviewStatusResponse,
    status_code=status.HTTP_200_OK
)
async def get_interview_status(session_id: str):
    """
    Get the current status of an interview session.

    Args:
        session_id: ID of the session to check

    Returns:
        Interview status information
    """
    logger.info(f"Status check for session {session_id}")

    # Validate session exists
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    session = sessions[session_id]

    # Calculate time metrics
    time_elapsed_seconds = None
    time_remaining_seconds = None

    if session.start_time is not None:
        from datetime import datetime
        time_elapsed_seconds = (datetime.now() - session.start_time).total_seconds()
        max_time = 45 * 60  # 45 minutes
        time_remaining_seconds = max(0, max_time - time_elapsed_seconds)

    return InterviewStatusResponse(
        session_id=session_id,
        is_active=session.is_active,
        is_concluded=session.is_concluded,
        questions_asked=session.questions_asked,
        time_elapsed_seconds=time_elapsed_seconds,
        time_remaining_seconds=time_remaining_seconds,
        conclusion_reason=session.conclusion_reason
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": str(exc.detail)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom exception handler for unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

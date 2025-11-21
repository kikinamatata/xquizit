# Interview Flow & Data Flow Documentation

## Overview

This document details the complete data flow and workflow sequences for the xquizit interview system, including document processing, real-time transcription, streaming responses, and interview orchestration.

---

## Complete Interview Workflow

```mermaid
sequenceDiagram
    actor C as Candidate
    participant F as Frontend
    participant B as Backend
    participant G as LangGraph
    participant L as Gemini LLM
    participant T as TTS Service
    participant R as RunPod API

    Note over C,R: Phase 1: Initialization

    C->>F: Upload resume + job description + custom instructions
    F->>B: POST /upload-documents (multipart/form-data)
    B->>B: Extract text from documents
    B->>B: Create session (UUID)
    B-->>F: { session_id }

    F->>B: POST /start-interview (session_id)
    B->>G: Invoke graph: analyze_documents node
    G->>L: Analyze resume + job description + instructions
    L-->>G: Interview strategy + key topics
    G->>G: Store strategy in state
    G->>G: generate_question node (first question)
    G->>L: Generate introductory question
    L-->>G: "Hello! I'm excited to learn about..."
    G->>T: Generate audio for first question
    T-->>G: [audio_chunk_1, audio_chunk_2, ...]
    G-->>B: First question + audio chunks
    B-->>F: { first_question, audio_chunks, start_time }

    F->>F: Display first question
    F->>F: Queue and play audio chunks

    Note over C,R: Phase 2: Question-Answer Loop

    loop Until interview concludes
        C->>F: Click "Record"
        F->>B: Connect WebSocket /ws/transcribe/{session_id}
        B->>R: Create transcription session
        B-->>F: {"type": "ready"}

        F->>F: Start audio capture (16kHz mono)

        loop Audio Streaming
            F->>B: Send audio chunk (int16 PCM, ~256ms)
            B->>B: Buffer audio in memory

            Note over B,R: Every 3 seconds
            B->>R: POST transcribe (buffered audio as WAV)
            R-->>B: Transcription segments
            B-->>F: {"type": "transcript", "segment": {...}}
            F->>F: Update live transcript display
        end

        C->>F: Click "Stop Recording"
        F->>B: {"type": "stop_recording"}
        B->>R: Final flush (remaining audio)
        R-->>B: Final transcription segments
        B-->>F: {"type": "transcript", ...} (final segments)
        B-->>F: {"type": "complete"}

        F->>F: Auto-submit final transcript
        F->>F: Add candidate message to chat

        F->>B: GET /stream-answer?session_id=...&answer=...
        Note over F,B: EventSource SSE connection opened

        B->>G: Add answer to conversation
        B->>G: Invoke graph: process_answer node
        G->>L: Evaluate answer quality
        L-->>G: Decision: follow_up or next_topic
        G->>G: Update follow-up counts

        B->>G: Invoke graph: check_time node
        G->>G: Calculate elapsed time
        G->>G: Check question count

        alt Time/Questions OK
            G->>G: generate_question node (stream)
            G->>L: Generate next question (streaming)

            loop Text Streaming
                L-->>G: Text chunk
                G-->>B: Text chunk
                B-->>F: event: text_chunk
                F->>F: Append to streaming message
                F->>F: Display with typing cursor
            end

            Note over B,T: Sentence Completion Detection
            B->>B: Extract complete sentences from buffer
            B->>T: Generate audio for sentence

            loop Audio Streaming
                T-->>B: Audio chunk (base64 WAV)
                B-->>F: event: audio_chunk
                F->>F: Queue audio chunk
                F->>F: Play sequentially
            end

            B-->>F: event: metadata (questions_asked, is_concluded)
            B-->>F: event: done

            F->>F: Finalize streaming message
            F->>F: Add to message history
        else Time/Question Limit Reached
            G->>G: conclude_interview node
            G->>L: Generate conclusion message
            L-->>G: Conclusion text
            G-->>B: Conclusion + is_concluded = true
            B->>T: Generate conclusion audio

            loop Conclusion Streaming
                B-->>F: event: text_chunk (conclusion)
                B-->>F: event: audio_chunk (conclusion)
            end

            B-->>F: event: metadata (is_concluded: true)
            B-->>F: event: done

            F->>F: Display conclusion
            F->>F: Disable recording button
        end
    end

    Note over C,R: Interview Complete
```

---

## Detailed Flow Breakdowns

### 1. Document Upload and Analysis Flow

```mermaid
sequenceDiagram
    participant F as Frontend
    participant B as Backend (main.py)
    participant D as DocumentProcessor
    participant G as LangGraph
    participant L as Gemini LLM

    F->>B: POST /upload-documents
    Note over F,B: resume (File)<br/>job_description (File)<br/>custom_instructions (str)

    B->>D: extract_text_from_document(resume)
    D->>D: Detect file type (PDF/DOCX)
    alt PDF
        D->>D: PyPDF2.PdfReader(file)
        D->>D: Extract text from all pages
    else DOCX
        D->>D: python-docx.Document(file)
        D->>D: Extract paragraphs + tables
    end
    D-->>B: resume_text

    B->>D: extract_text_from_document(job_description)
    D-->>B: job_description_text

    B->>B: Create session (UUID)
    B->>B: Store in sessions dict
    Note over B: sessions[session_id] = SessionData(...)

    B-->>F: { "session_id": "...", "message": "Success" }

    Note over F,B: === Start Interview ===

    F->>B: POST /start-interview (session_id)
    B->>B: Retrieve session from sessions dict

    B->>G: Invoke graph with initial state
    Note over B,G: State: {<br/>  resume_text,<br/>  job_description_text,<br/>  custom_instructions<br/>}

    G->>G: analyze_documents node
    G->>L: Analyze documents
    Note over G,L: Prompt includes:<br/>- Resume text<br/>- Job description<br/>- Custom instructions<br/>- Output format instructions

    L->>L: Generate interview strategy
    L-->>G: Strategy + key topics
    Note over L,G: STRATEGY: Focus on Python...<br/>TOPICS:<br/>- Python Experience<br/>- Team Leadership<br/>- System Design

    G->>G: Parse and extract topics
    G->>G: Update state with strategy

    G->>G: generate_question node
    G->>L: Generate first question (introductory)
    Note over G,L: First question is always<br/>welcoming and introductory

    L-->>G: "Hello! I'm excited to learn about your background..."

    G-->>B: Updated state with first question
    B->>B: Extract question from messages

    B->>T: Generate TTS audio
    Note over B,T: text = first_question

    T->>T: Synthesize audio (Kokoro)
    T->>T: Convert to WAV bytes
    T->>T: Base64 encode
    T-->>B: [base64_chunk_1, base64_chunk_2, ...]

    B->>B: Update session with state + start_time
    B-->>F: {<br/>  first_question,<br/>  audio_chunks,<br/>  start_time<br/>}

    F->>F: Display question
    F->>F: Queue and play audio
```

**Key Points**:
- Document analysis happens **once** per interview
- Strategy guides all subsequent questions
- Key topics extracted for rotation logic
- First question always introductory

---

### 2. Real-Time Transcription Flow (RunPod WebSocket)

```mermaid
sequenceDiagram
    participant F as Frontend
    participant W as WebSocket Connection
    participant B as Backend
    participant S as RunPodTranscriptionSession
    participant R as RunPod API

    Note over F,B: User clicks "Record"

    F->>W: Connect ws://backend/ws/transcribe/{session_id}
    W->>B: WebSocket connection established
    B->>B: Validate session exists
    B->>S: Create transcription session
    S->>S: Initialize audio buffer
    S->>S: Start periodic transcription task
    B->>W: {"type": "ready"}
    W->>F: ready message received

    F->>F: Request microphone access
    F->>F: Create AudioContext (16kHz)
    F->>F: Setup ScriptProcessor (4096 samples)

    loop Audio Streaming (every ~256ms)
        F->>F: Capture audio chunk
        F->>F: Convert float32 → int16
        F->>W: Send binary audio (ArrayBuffer)
        W->>B: Binary WebSocket frame
        B->>S: add_audio_chunk(bytes)
        S->>S: Append to audio_buffer
        S->>S: Trim if exceeds max_buffer_seconds
    end

    Note over S,R: Periodic Task (every 3 seconds)

    loop While recording active
        S->>S: Wait 3 seconds
        S->>S: Check if buffer has audio

        alt Buffer not empty
            S->>S: Concatenate audio chunks
            S->>S: Convert int16 → float32 normalized
            S->>S: Encode as WAV (scipy)
            S->>S: Base64 encode WAV

            S->>R: POST /run (audio + config)
            Note over S,R: Request body:<br/>{<br/>  "input": {<br/>    "audio": "base64-wav",<br/>    "model": "small",<br/>    "language": "en",<br/>    "use_vad": true<br/>  }<br/>}

            R->>R: Decode audio
            R->>R: Run Whisper model
            R->>R: Apply VAD filtering
            R-->>S: {<br/>  "output": {<br/>    "segments": [...]<br/>  }<br/>}

            S->>S: Parse segments
            S->>S: Transform to frontend format
            S->>B: Forward segments
            B->>W: {"type": "transcript", "segment": {...}}
            W->>F: Transcript message

            F->>F: Check if is_final
            alt is_final = true
                F->>F: Add to finalTranscript (with dedup)
            else is_final = false
                F->>F: Update lastInterimSegment
            end
            F->>F: Update live transcript display
        end
    end

    Note over F,B: User clicks "Stop Recording"

    F->>F: Record stop timestamp
    F->>W: {"type": "stop_recording"}
    W->>B: JSON control message
    B->>S: stop_recording()

    S->>S: Stop periodic task
    S->>S: Check if buffer has remaining audio

    alt Buffer not empty
        S->>S: Transcribe remaining audio
        S->>R: POST /run (final audio)
        R-->>S: Final segments
        S->>B: Forward final segments
        B->>W: {"type": "transcript", ...} (multiple)
    end

    S->>B: Transcription finalized
    B->>W: {"type": "complete"}
    W->>F: Complete message

    F->>F: Concatenate all final segments
    F->>F: Auto-submit transcript
    F->>W: Close WebSocket
    W->>B: Connection closed
    B->>S: Cleanup session
```

**Key Points**:
- **Buffering**: Audio buffered in backend, sent to RunPod every 3 seconds
- **Deduplication**: Frontend uses timestamps to prevent duplicate segments
- **Final Flush**: Remaining audio processed on stop_recording
- **Auto-Submit**: Frontend automatically submits when complete message received

**Performance**:
- **Latency**: 1-3 seconds per transcription cycle
- **Accuracy**: Improved with VAD (Voice Activity Detection)
- **Cost**: ~$0.50-$1.00 per 30-minute interview

---

### 3. Streaming Response Flow (SSE)

```mermaid
sequenceDiagram
    participant F as Frontend
    participant E as EventSource
    participant B as Backend
    participant G as LangGraph
    participant L as Gemini LLM
    participant T as TTS Service

    Note over F,B: User submitted answer via transcription

    F->>E: Create EventSource
    Note over F,E: GET /stream-answer?<br/>session_id=...&<br/>answer=...

    E->>B: HTTP GET (SSE connection)
    B->>B: Retrieve session
    B->>B: Add answer to conversation

    B->>G: Invoke graph: process_answer
    G->>L: Evaluate answer
    Note over G,L: Prompt:<br/>- Last question<br/>- Candidate answer<br/>- Current topic<br/><br/>Return:<br/>DECISION: follow_up | next_topic

    L-->>G: Decision + reasoning
    G->>G: Update needs_followup
    G->>G: Update topic_followup_counts

    alt Follow-up needed and count < 2
        G->>G: Keep current_topic_index same
    else Move to next topic
        G->>G: Increment current_topic_index
    end

    G->>G: Transition to check_time node
    G->>G: Calculate time elapsed
    G->>G: Check questions_asked count

    alt Time OK and questions < 16
        G->>G: Transition to generate_question
        G->>L: Generate question (streaming)
        Note over G,L: Stream mode:<br/>astream() instead of invoke()

        loop Text Streaming
            L-->>G: Chunk of text
            G-->>B: Chunk
            B->>B: Accumulate in text buffer
            B->>E: event: text_chunk\ndata: {"text": "..."}
            E->>F: text_chunk event
            F->>F: Append to streamingMessage.text
            F->>F: Render with typing cursor

            Note over B: Sentence Detection
            B->>B: Check if buffer has complete sentence
            alt Complete sentence found
                B->>B: Extract sentence
                B->>T: generate_stream(sentence)

                loop Audio Chunks
                    T-->>B: Base64 audio chunk
                    B->>E: event: audio_chunk\ndata: {"audio": "..."}
                    E->>F: audio_chunk event
                    F->>F: addAudioChunk(base64)
                    F->>F: Queue for playback
                end

                B->>B: Clear sentence from buffer
            end
        end

        G-->>B: Stream complete

        B->>B: Generate audio for remaining text
        B->>T: generate_stream(remaining_text)
        T-->>B: Final audio chunks
        B->>E: event: audio_chunk (final chunks)

        B->>B: Prepare metadata
        B->>E: event: metadata\ndata: {<br/>  "is_concluded": false,<br/>  "questions_asked": N<br/>}
        E->>F: metadata event
        F->>F: Update question counter

        B->>E: event: done\ndata: {"status": "complete"}
        E->>F: done event

        F->>F: Finalize streamingMessage
        F->>F: Add to messages array
        F->>F: Clear streamingMessage
        F->>E: Close EventSource

    else Time limit or max questions reached
        G->>G: Transition to conclude_interview
        G->>L: Generate conclusion message
        L-->>G: Conclusion text
        G-->>B: Conclusion + is_concluded = true

        B->>E: event: text_chunk (conclusion text)
        E->>F: Display conclusion with typing effect

        B->>T: generate_stream(conclusion)
        T-->>B: Audio chunks
        B->>E: event: audio_chunk
        E->>F: Play conclusion audio

        B->>E: event: metadata\ndata: {<br/>  "is_concluded": true,<br/>  "questions_asked": N<br/>}
        E->>F: metadata event
        F->>F: Set interviewComplete = true
        F->>F: Disable recording button

        B->>E: event: done
        F->>E: Close EventSource
    end
```

**Key Points**:
- **Streaming Text**: Word-by-word or chunk-by-chunk delivery
- **Parallel Audio**: Audio generated and sent alongside text
- **Sentence-Based Audio**: TTS triggered on complete sentences
- **Metadata**: Question count and conclusion status sent separately
- **Graceful Conclusion**: Time/question limits trigger conclusion flow

**Visual Feedback**:
- Typing cursor during streaming
- "Streaming..." badge with pulse animation
- Audio playback queue prevents gaps

---

### 4. LangGraph State Transitions

```mermaid
stateDiagram-v2
    [*] --> analyze_documents: Upload Documents

    analyze_documents --> generate_question: Strategy Created

    generate_question --> [*]: Return First Question

    note right of generate_question
        First question is always
        introductory/welcoming
    end note

    [*] --> process_answer: User Submits Answer

    process_answer --> check_time: Answer Evaluated

    check_time --> generate_question: Time/Questions OK
    check_time --> conclude_interview: Limit Reached

    generate_question --> [*]: Return Next Question

    conclude_interview --> [*]: Interview Complete

    note right of check_time
        Checks:
        - Time < 45 minutes
        - Questions < 16
    end note

    note left of process_answer
        Evaluates answer quality
        Decides follow-up strategy
        Updates topic counts
    end note

    note right of conclude_interview
        Generates warm conclusion
        Sets is_concluded = true
    end note
```

---

### 5. Follow-Up Question Logic Flow

```mermaid
flowchart TD
    A[Answer Submitted] --> B[process_answer node]
    B --> C{LLM Evaluation}

    C -->|Answer adequate| D[Decision: next_topic]
    C -->|Answer incomplete| E[Decision: follow_up]

    D --> F[Increment current_topic_index]
    E --> G{Check follow-up count for topic}

    G -->|Count < 2| H[Increment followup count]
    G -->|Count >= 2| I[Force next topic]

    H --> J[Keep current_topic_index same]
    I --> F

    F --> K[generate_question node]
    J --> K

    K --> L{Question type}
    L -->|Next topic| M[Select topic from rotation]
    L -->|Follow-up| N[Ask deeper question on same topic]

    M --> O[Generate question using LLM]
    N --> O

    O --> P[Return question to user]

    style C fill:#FFE5B4
    style G fill:#FFE5B4
    style O fill:#87CEEB
```

**Key Rules**:
1. Maximum **2 follow-ups per topic**
2. LLM evaluates answer quality to decide follow-up
3. If follow-up limit reached, force move to next topic
4. Topic selection uses round-robin rotation

**Example Scenario**:
```
Topic: "Python Experience"
Q1: "Describe your Python experience"
A1: "I've used Python for 5 years"
  → LLM: Answer brief, needs follow-up
  → followup_count["Python Experience"] = 1

Q2: "Can you describe a complex Python project you worked on?"
A2: "I built a data pipeline using Pandas and Airflow..."
  → LLM: Good detail, but could probe deeper
  → followup_count["Python Experience"] = 2

Q3: "What challenges did you face and how did you solve them?"
A3: [Detailed response]
  → followup_count["Python Experience"] = 2 (max reached)
  → Force move to next topic: "Team Leadership"
```

---

### 6. Audio Format Conversion Pipeline

```mermaid
flowchart LR
    subgraph Frontend
        A[Microphone] -->|Capture| B[AudioContext<br/>48kHz default]
        B -->|Resample| C[16kHz mono<br/>float32 [-1, 1]]
        C -->|ScriptProcessor| D[4096 sample chunks<br/>~256ms]
        D -->|Convert| E[int16 PCM<br/>[-32768, 32767]]
        E -->|Binary| F[WebSocket<br/>ArrayBuffer]
    end

    subgraph Backend
        F -->|Receive| G[bytes data]
        G -->|Buffer| H[numpy array<br/>int16]
        H -->|Concatenate| I[Full buffer<br/>max 30 seconds]
        I -->|Convert| J[float32 normalized<br/>[-1.0, 1.0]]
        J -->|Encode| K[WAV file<br/>scipy.io.wavfile]
        K -->|Base64| L[String payload]
    end

    subgraph RunPod API
        L -->|HTTP POST| M[Decode Base64]
        M -->|Parse WAV| N[Whisper Model<br/>16kHz expected]
        N -->|Transcribe| O[Segments]
        O -->|Return JSON| P[Backend]
    end

    subgraph TTS Pipeline
        Q[Generated Text] -->|Kokoro Engine| R[Synthesize<br/>24kHz audio]
        R -->|numpy array| S[Convert to<br/>int16 WAV]
        S -->|Base64 encode| T[SSE Event]
        T -->|EventSource| U[Frontend]
        U -->|Decode| V[Blob + Object URL]
        V -->|Audio Element| W[Playback]
    end

    style A fill:#FFE5B4
    style N fill:#87CEEB
    style Q fill:#90EE90
    style W fill:#FFB6C1
```

**Format Summary**:

| Stage | Format | Sample Rate | Encoding |
|-------|--------|-------------|----------|
| Microphone Capture | float32 | 48kHz (browser default) | Raw PCM |
| Frontend Processing | int16 | 16kHz | Raw PCM |
| WebSocket Transmission | int16 | 16kHz | Binary ArrayBuffer |
| Backend Buffer | int16 | 16kHz | numpy array |
| RunPod Payload | float32 | 16kHz | WAV (Base64) |
| TTS Output | int16 | 24kHz | WAV (Base64) |
| Frontend Playback | int16 | 24kHz | WAV (Blob URL) |

---

### 7. Session Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant S as sessions dict

    Note over U,S: Session Creation

    U->>F: Upload documents
    F->>B: POST /upload-documents
    B->>B: Generate UUID session_id
    B->>S: sessions[session_id] = SessionData(...)
    Note over S: {<br/>  session_id,<br/>  resume_text,<br/>  job_description_text,<br/>  custom_instructions,<br/>  interview_state: {},<br/>  start_time: None,<br/>  last_activity: now<br/>}
    B-->>F: { session_id }

    Note over U,S: Session Initialization

    F->>B: POST /start-interview
    B->>S: Retrieve session by session_id
    S-->>B: SessionData
    B->>B: Run LangGraph workflow
    B->>S: Update interview_state
    B->>S: Set start_time
    Note over S: {<br/>  ...,<br/>  interview_state: {<br/>    messages: [...],<br/>    interview_strategy: "...",<br/>    key_topics: [...],<br/>    questions_asked: 1<br/>  },<br/>  start_time: now<br/>}
    B-->>F: First question + audio

    Note over U,S: Active Interview

    loop Question-Answer Cycles
        U->>F: Record and submit answer
        F->>B: GET /stream-answer
        B->>S: Retrieve session
        S-->>B: SessionData
        B->>B: Add answer to state
        B->>B: Run LangGraph (process → check → generate)
        B->>S: Update interview_state
        Note over S: {<br/>  ...,<br/>  interview_state: {<br/>    messages: [..., new_answer, new_question],<br/>    questions_asked: N,<br/>    current_topic_index: M,<br/>    topic_followup_counts: {...}<br/>  },<br/>  last_activity: now<br/>}
        B-->>F: Stream response
    end

    Note over U,S: Session Termination

    alt Interview Concludes Normally
        B->>B: Time/question limit reached
        B->>S: Update interview_state (is_concluded: true)
        B-->>F: Conclusion message
    else User Closes Browser
        Note over F: No explicit cleanup
        Note over S: Session remains in memory
        Note over B: Could implement TTL cleanup
    else Server Restart
        Note over S: All sessions lost (in-memory)
    end
```

**Session Storage Characteristics**:
- **In-Memory**: Stored in Python dict (`sessions`)
- **Not Persistent**: Lost on server restart
- **No Expiration**: Sessions never auto-deleted (manual cleanup needed)
- **Concurrency**: Thread-safe for async FastAPI (GIL)

**Production Recommendations**:
1. **Use Redis** for persistent session storage
2. **Implement TTL** (time-to-live) for session expiration
3. **Add cleanup tasks** to remove stale sessions
4. **Session recovery** on server restart

---

### 8. Error Recovery Flows

```mermaid
flowchart TD
    A[User Action] --> B{Action Type}

    B -->|SSE Stream| C[EventSource Connection]
    B -->|WebSocket| D[WebSocket Connection]
    B -->|HTTP Request| E[Axios Request]

    C --> C1{Connection Status}
    C1 -->|Success| C2[Receive Events]
    C1 -->|Error/Close| C3[EventSource.onerror]

    C3 --> C4[Save Partial Response]
    C4 --> C5[Display Error Message]
    C5 --> C6[Close EventSource]
    C6 --> C7[Reset UI State]

    D --> D1{Connection Status}
    D1 -->|Success| D2[Stream Audio/Receive Transcripts]
    D1 -->|Error/Close| D3[WebSocket.onerror]

    D3 --> D4{Recording Active?}
    D4 -->|Yes| D5[Stop Recording]
    D4 -->|No| D7[Display Error]
    D5 --> D6[Save Partial Transcript]
    D6 --> D7
    D7 --> D8[Close WebSocket]
    D8 --> D9[Reset UI State]

    E --> E1{Request Status}
    E1 -->|200 Success| E2[Process Response]
    E1 -->|4xx Client Error| E3[Display Validation Error]
    E1 -->|5xx Server Error| E4[Display Server Error]
    E1 -->|Network Error| E5[Display Connection Error]

    E3 --> E6[Log Error]
    E4 --> E6
    E5 --> E6
    E6 --> E7[User Can Retry]

    style C3 fill:#FFB6C1
    style D3 fill:#FFB6C1
    style E3 fill:#FFB6C1
    style E4 fill:#FFB6C1
    style E5 fill:#FFB6C1
```

**Error Handling Strategies**:

1. **SSE Connection Loss**:
   - Save partial streaming response
   - Display "[Connection lost]" indicator
   - Clean up EventSource
   - Allow user to retry

2. **WebSocket Disconnect**:
   - Stop audio recording gracefully
   - Save partial transcript if available
   - Display error message
   - Prevent auto-submit if incomplete

3. **HTTP Request Failures**:
   - Parse error details from response
   - Display user-friendly message
   - Log technical details to console
   - Enable retry if appropriate

4. **Audio Playback Errors**:
   - Skip problematic chunk
   - Continue with next chunk
   - Log error (don't block interview)

---

## Performance Metrics

### Typical Latencies

| Operation | Latency | Notes |
|-----------|---------|-------|
| Document Upload | < 1s | File I/O + text extraction |
| Document Analysis (LLM) | 2-5s | One-time per interview |
| First Question Generation | 1-3s | Includes TTS |
| **Transcription Cycle** | **1-3s** | Every 3 seconds during recording |
| **Stop → Transcript Ready** | **2-5s** | Final flush + complete signal |
| Answer Evaluation (LLM) | 1-2s | Fast with thinking_budget=0 |
| Question Generation (LLM) | 1-3s | Streaming reduces perceived latency |
| TTS Generation | 1-2s | Per sentence |
| **Total Round-Trip** | **7-12s** | Stop → Audio playback starts |

### Optimization Impact

| Metric | Before Optimization | After Optimization | Improvement |
|--------|---------------------|---------------------|-------------|
| LLM Latency | 3-5s | 1-3s | 30-60% faster |
| LLM Cost | High (thinking tokens) | Low (no thinking) | 600% reduction |
| Time-to-First-Token | 2-3s | 0.5-1.5s | 50% faster |
| User Perceived Latency | High | Low | Streaming effect |

---

## Data Models

### Interview State

```python
{
  # Document Data
  "resume_text": str,
  "job_description_text": str,
  "custom_instructions": str,

  # Strategy
  "interview_strategy": str,
  "key_topics": List[str],

  # Conversation
  "messages": [
    {"role": "ai", "content": "Hello! I'm excited..."},
    {"role": "human", "content": "I have 5 years of experience..."},
    ...
  ],

  # State Tracking
  "questions_asked": int,
  "current_topic_index": int,
  "topic_followup_counts": {
    "Python Experience": 2,
    "Team Leadership": 0,
    "System Design": 1
  },
  "last_question_topic": str,

  # Interview Control
  "start_time": "2025-11-04T10:30:00Z",
  "should_conclude": bool,
  "is_concluded": bool,
  "needs_followup": bool
}
```

### Session Data

```python
{
  "session_id": str,
  "resume_text": str,
  "job_description_text": str,
  "custom_instructions": str,
  "interview_state": Dict[str, Any],
  "start_time": datetime,
  "last_activity": datetime
}
```

### Transcription Segment

```python
{
  "text": str,
  "start": float,      # seconds
  "end": float,        # seconds
  "is_final": bool
}
```

### SSE Event Payloads

**text_chunk**:
```json
{
  "text": "word or phrase"
}
```

**audio_chunk**:
```json
{
  "audio": "base64-encoded-wav-data"
}
```

**metadata**:
```json
{
  "is_concluded": false,
  "questions_asked": 5,
  "time_elapsed": 300.5
}
```

**done**:
```json
{
  "status": "complete"
}
```

---

## Summary

The xquizit data flow demonstrates:

- **Multi-Protocol Communication**: REST, SSE, WebSocket orchestrated seamlessly
- **Real-Time Streaming**: Text and audio delivered progressively for optimal UX
- **State Machine Orchestration**: LangGraph manages complex interview logic
- **Serverless Integration**: RunPod and Gemini provide scalable, cost-effective services
- **Graceful Error Handling**: Robust recovery mechanisms for network and service failures
- **Performance Optimization**: Thinking budgets, streaming, and audio queuing minimize latency

The system balances **real-time responsiveness** with **intelligent interview orchestration**, creating a natural and engaging candidate experience.
# API Reference Documentation

## Overview

This document provides complete API reference for the xquizit interview system, including REST endpoints, WebSocket protocol, and Server-Sent Events (SSE) specifications.

**Base URL**: `http://localhost:8000` (development) or `https://api.your-domain.com` (production)

**API Version**: 1.0

---

## Table of Contents

1. [REST API Endpoints](#rest-api-endpoints)
2. [WebSocket Protocol](#websocket-protocol)
3. [Server-Sent Events (SSE)](#server-sent-events-sse)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## REST API Endpoints

### POST /upload-documents

Upload resume and job description documents to create an interview session.

**Request**:
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body Parameters**:
  - `resume` (File, required): PDF or DOCX file containing candidate resume
  - `job_description` (File, required): PDF or DOCX file containing job description
  - `custom_instructions` (string, optional): Additional guidance for the interview (max 2000 characters)

**Example Request** (curl):
```bash
curl -X POST http://localhost:8000/upload-documents \
  -F "resume=@john_doe_resume.pdf" \
  -F "job_description=@senior_engineer_jd.docx" \
  -F "custom_instructions=Focus on leadership experience and team collaboration skills"
```

**Example Request** (JavaScript):
```javascript
const formData = new FormData();
formData.append('resume', resumeFile);
formData.append('job_description', jobDescFile);
formData.append('custom_instructions', 'Focus on technical depth');

const response = await fetch('http://localhost:8000/upload-documents', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data.session_id);
```

**Response** (200 OK):
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Documents uploaded successfully"
}
```

**Error Responses**:

- **400 Bad Request** - Invalid file format:
  ```json
  {
    "detail": "Only PDF and DOCX files are supported"
  }
  ```

- **500 Internal Server Error** - Document processing failed:
  ```json
  {
    "detail": "Failed to extract text from document: [error details]"
  }
  ```

**File Constraints**:
- Supported formats: PDF (.pdf), DOCX (.docx, .doc)
- Max file size: 10 MB (configurable in nginx)
- Max custom instructions: 2000 characters

---

### POST /start-interview

Initialize interview session and receive the first question.

**Request**:
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
  ```

**Example Request** (curl):
```bash
curl -X POST http://localhost:8000/start-interview \
  -H "Content-Type: application/json" \
  -d '{"session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}'
```

**Example Request** (JavaScript):
```javascript
const response = await fetch('http://localhost:8000/start-interview', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    session_id: sessionId
  })
});

const data = await response.json();
console.log(data.first_question);
```

**Response** (200 OK):
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "first_question": "Hello! I'm excited to learn about your background and experience. To start, could you tell me a bit about yourself and what attracted you to this position?",
  "audio_chunks": [
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
  ],
  "start_time": "2025-11-04T10:30:00.123456Z"
}
```

**Response Fields**:
- `session_id` (string): Interview session identifier
- `first_question` (string): The introductory interview question
- `audio_chunks` (array of strings): Base64-encoded WAV audio chunks for the question
- `start_time` (string): ISO 8601 timestamp of interview start

**Error Responses**:

- **404 Not Found** - Session not found:
  ```json
  {
    "detail": "Session not found"
  }
  ```

- **500 Internal Server Error** - LLM or TTS failure:
  ```json
  {
    "detail": "Failed to generate first question: [error details]"
  }
  ```

**Processing Time**: 4-10 seconds (includes document analysis, question generation, and TTS)

---

### GET /stream-answer

Submit candidate answer and receive streaming interview response via Server-Sent Events.

**Request**:
- **Method**: `GET`
- **Query Parameters**:
  - `session_id` (string, required): Interview session identifier
  - `answer` (string, required): Candidate's answer text (URL-encoded)

**Example Request** (curl):
```bash
curl -N http://localhost:8000/stream-answer \
  ?session_id=a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  &answer=I%20have%205%20years%20of%20experience
```

**Example Request** (JavaScript):
```javascript
const sessionId = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';
const answer = 'I have 5 years of experience with Python...';

const eventSource = new EventSource(
  `http://localhost:8000/stream-answer?session_id=${sessionId}&answer=${encodeURIComponent(answer)}`
);

eventSource.addEventListener('text_chunk', (event) => {
  const data = JSON.parse(event.data);
  console.log('Text chunk:', data.text);
});

eventSource.addEventListener('audio_chunk', (event) => {
  const data = JSON.parse(event.data);
  console.log('Audio chunk:', data.audio);
});

eventSource.addEventListener('metadata', (event) => {
  const data = JSON.parse(event.data);
  console.log('Metadata:', data);
});

eventSource.addEventListener('done', (event) => {
  console.log('Stream complete');
  eventSource.close();
});

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  eventSource.close();
};
```

**Response**: Server-Sent Events stream

**Event Types**:

1. **text_chunk** - Incremental text segments:
   ```
   event: text_chunk
   data: {"text": "That's an interesting "}

   event: text_chunk
   data: {"text": "answer. "}
   ```

2. **audio_chunk** - Base64-encoded audio segments:
   ```
   event: audio_chunk
   data: {"audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."}
   ```

3. **metadata** - Interview metadata:
   ```
   event: metadata
   data: {
     "is_concluded": false,
     "questions_asked": 5,
     "time_elapsed": 301.5
   }
   ```

4. **done** - Stream completion:
   ```
   event: done
   data: {"status": "complete"}
   ```

**Fallback (Unnamed Events)**:

For compatibility, the endpoint also emits unnamed events that can be handled via `EventSource.onmessage`:

```javascript
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle based on data structure
};
```

**Error Responses**:

- **404 Not Found** - Session not found
- **500 Internal Server Error** - Processing error (stream will close with error event)

**Connection Duration**: Variable (until question generation completes, typically 5-15 seconds)

---

### POST /submit-answer (Legacy)

Submit candidate answer and receive next question (non-streaming version).

**⚠️ Deprecated**: Use `/stream-answer` for better user experience.

**Request**:
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "answer": "I have 5 years of experience with Python, focusing on backend development..."
  }
  ```

**Response** (200 OK):
```json
{
  "question": "Can you describe a challenging Python project you worked on?",
  "audio_chunks": [
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
  ],
  "is_concluded": false,
  "questions_asked": 2,
  "time_elapsed": 120.5
}
```

**Response Fields**:
- `question` (string): Next interview question
- `audio_chunks` (array): Base64-encoded audio for question
- `is_concluded` (boolean): Whether interview has concluded
- `questions_asked` (integer): Total questions asked so far
- `time_elapsed` (float): Seconds since interview started

---

### GET /interview-status/{session_id}

Check interview progress and status.

**Request**:
- **Method**: `GET`
- **Path Parameter**:
  - `session_id` (string): Interview session identifier

**Example Request**:
```bash
curl http://localhost:8000/interview-status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response** (200 OK):
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "questions_asked": 8,
  "is_concluded": false,
  "time_elapsed": 540.2,
  "start_time": "2025-11-04T10:30:00.123456Z"
}
```

**Error Responses**:

- **404 Not Found**:
  ```json
  {
    "detail": "Session not found"
  }
  ```

---

## WebSocket Protocol

### WS /ws/transcribe/{session_id}

Real-time audio transcription via WebSocket connection.

**Connection**:
```
ws://localhost:8000/ws/transcribe/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Example (JavaScript)**:
```javascript
const ws = new WebSocket(
  `ws://localhost:8000/ws/transcribe/${sessionId}`
);

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'ready') {
    console.log('Server ready to receive audio');
    startAudioCapture();
  } else if (data.type === 'transcript') {
    console.log('Transcript segment:', data.segment);
  } else if (data.type === 'complete') {
    console.log('Transcription complete');
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed');
};
```

---

### Message Types (Client → Server)

#### 1. Binary Audio Data

**Format**: int16 PCM, 16kHz, mono

**Encoding**: Raw binary (ArrayBuffer)

**Chunk Size**: 4096 samples (~256ms)

**Example**:
```javascript
// Capture audio from microphone
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  const inputData = e.inputBuffer.getChannelData(0);

  // Convert float32 to int16
  const int16Data = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i++) {
    const s = Math.max(-1, Math.min(1, inputData[i]));
    int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  // Send binary data
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(int16Data.buffer);
  }
};
```

---

#### 2. Control Messages (JSON)

**Stop Recording**:
```json
{
  "type": "stop_recording"
}
```

**Example**:
```javascript
// Signal end of recording
ws.send(JSON.stringify({ type: 'stop_recording' }));
```

---

### Message Types (Server → Client)

#### 1. Ready

**Description**: Server is ready to receive audio

**Format**:
```json
{
  "type": "ready"
}
```

**Usage**: Client should wait for this message before sending audio

---

#### 2. Transcript Segment

**Description**: Transcription result for audio segment

**Format**:
```json
{
  "type": "transcript",
  "segment": {
    "text": "Hello, I have experience with Python",
    "is_final": false,
    "start": "0.0",
    "end": "2.5"
  }
}
```

**Fields**:
- `text` (string): Transcribed text
- `is_final` (boolean): Whether this is a final (non-interim) result
- `start` (string): Start timestamp in seconds
- `end` (string): End timestamp in seconds

**Segment Types**:
- **Interim** (`is_final: false`): Temporary transcription, may change
- **Final** (`is_final: true`): Confirmed transcription, won't change

---

#### 3. Complete

**Description**: Transcription finalized, all segments sent

**Format**:
```json
{
  "type": "complete"
}
```

**Usage**: Client should concatenate all final segments and submit transcript

---

### Transcription Flow

1. **Connect**: Client opens WebSocket connection
2. **Ready**: Server sends `{"type": "ready"}`
3. **Stream**: Client sends audio chunks (binary)
4. **Transcribe**: Server periodically sends transcript segments (every 3 seconds)
5. **Stop**: Client sends `{"type": "stop_recording"}`
6. **Flush**: Server processes remaining audio and sends final segments
7. **Complete**: Server sends `{"type": "complete"}`
8. **Close**: Client closes WebSocket

---

### Error Handling

**Connection Errors**:
- **404 Not Found**: Session doesn't exist
- **403 Forbidden**: Session validation failed

**WebSocket Errors**:
```javascript
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  // Stop recording, display error to user
};

ws.onclose = (event) => {
  if (event.code !== 1000) {
    console.error('Abnormal close:', event.code, event.reason);
  }
};
```

**Close Codes**:
- `1000`: Normal closure
- `1001`: Going away (server shutdown)
- `1006`: Abnormal closure (network error)

---

## Server-Sent Events (SSE)

### Event: text_chunk

**Description**: Incremental text segment from LLM

**Event Name**: `text_chunk`

**Data Format**:
```json
{
  "text": "word or phrase"
}
```

**Example**:
```
event: text_chunk
data: {"text": "Can you "}

event: text_chunk
data: {"text": "describe "}

event: text_chunk
data: {"text": "your experience "}

event: text_chunk
data: {"text": "with Python?"}
```

**Usage**: Accumulate `text` fields to build complete question

---

### Event: audio_chunk

**Description**: Base64-encoded audio segment

**Event Name**: `audio_chunk`

**Data Format**:
```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
}
```

**Audio Format**:
- Encoding: WAV, Base64-encoded
- Sample Rate: 24kHz
- Bit Depth: 16-bit
- Channels: Mono

**Example**:
```
event: audio_chunk
data: {"audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."}
```

**Usage**: Decode Base64, create Blob, and play via Audio element

**JavaScript Example**:
```javascript
eventSource.addEventListener('audio_chunk', (event) => {
  const data = JSON.parse(event.data);

  // Decode Base64
  const binaryString = atob(data.audio);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Create Blob and Object URL
  const blob = new Blob([bytes], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);

  // Play audio
  const audio = new Audio(url);
  audio.play();

  // Clean up after playback
  audio.onended = () => {
    URL.revokeObjectURL(url);
  };
});
```

---

### Event: metadata

**Description**: Interview metadata and progress

**Event Name**: `metadata`

**Data Format**:
```json
{
  "is_concluded": false,
  "questions_asked": 5,
  "time_elapsed": 301.5
}
```

**Fields**:
- `is_concluded` (boolean): Whether interview has concluded (time/question limit reached)
- `questions_asked` (integer): Total questions asked so far
- `time_elapsed` (float): Seconds since interview started

**Example**:
```
event: metadata
data: {"is_concluded": false, "questions_asked": 5, "time_elapsed": 301.5}
```

---

### Event: done

**Description**: Stream completion signal

**Event Name**: `done`

**Data Format**:
```json
{
  "status": "complete"
}
```

**Example**:
```
event: done
data: {"status": "complete"}
```

**Usage**: Close EventSource connection

```javascript
eventSource.addEventListener('done', () => {
  eventSource.close();
});
```

---

### SSE Connection Management

**Automatic Reconnection**:

EventSource API automatically reconnects on connection loss with exponential backoff.

**Disable Reconnection**:
```javascript
eventSource.addEventListener('done', () => {
  eventSource.close();  // Prevents reconnection
});
```

**Manual Reconnection**:
```javascript
eventSource.onerror = (error) => {
  console.error('SSE error:', error);

  if (eventSource.readyState === EventSource.CLOSED) {
    // Manually reconnect if needed
    setTimeout(() => {
      createNewEventSource();
    }, 5000);
  }
};
```

---

## Data Models

### UploadDocumentsRequest

**Format**: multipart/form-data

```typescript
{
  resume: File,                       // PDF or DOCX
  job_description: File,              // PDF or DOCX
  custom_instructions?: string        // Optional, max 2000 chars
}
```

---

### UploadDocumentsResponse

```typescript
{
  session_id: string,                 // UUID format
  message: string                     // Success message
}
```

---

### StartInterviewRequest

```typescript
{
  session_id: string                  // UUID from upload response
}
```

---

### StartInterviewResponse

```typescript
{
  session_id: string,                 // UUID
  first_question: string,             // Introductory question text
  audio_chunks: string[],             // Base64-encoded WAV chunks
  start_time: string                  // ISO 8601 timestamp
}
```

---

### SubmitAnswerRequest

```typescript
{
  session_id: string,                 // UUID
  answer: string                      // Candidate's answer text
}
```

---

### SubmitAnswerResponse

```typescript
{
  question: string,                   // Next question text
  audio_chunks: string[],             // Base64-encoded WAV chunks
  is_concluded: boolean,              // Interview concluded flag
  questions_asked: number,            // Total questions count
  time_elapsed: number                // Seconds since start
}
```

---

### InterviewStatusResponse

```typescript
{
  session_id: string,                 // UUID
  questions_asked: number,            // Total questions count
  is_concluded: boolean,              // Interview concluded flag
  time_elapsed: number,               // Seconds since start
  start_time: string | null           // ISO 8601 timestamp or null
}
```

---

### TranscriptionSegment

```typescript
{
  text: string,                       // Transcribed text
  start: string,                      // Start time in seconds (as string)
  end: string,                        // End time in seconds (as string)
  is_final: boolean                   // Final or interim result
}
```

---

### TranscriptionMessage

```typescript
{
  type: "ready" | "transcript" | "complete",
  segment?: TranscriptionSegment      // Present when type="transcript"
}
```

---

### ErrorResponse

```typescript
{
  detail: string                      // Error message
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Request succeeded |
| 400 | Bad Request | Invalid file format, missing required fields |
| 404 | Not Found | Session not found |
| 422 | Unprocessable Entity | Validation error (Pydantic) |
| 500 | Internal Server Error | LLM API failure, document processing error, TTS failure |

---

### Error Response Format

All errors follow a consistent JSON structure:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Examples**:

**Session Not Found**:
```json
{
  "detail": "Session not found"
}
```

**Invalid File Format**:
```json
{
  "detail": "Only PDF and DOCX files are supported"
}
```

**Document Processing Error**:
```json
{
  "detail": "Failed to extract text from document: File is corrupted"
}
```

**LLM API Failure**:
```json
{
  "detail": "Failed to generate question: API quota exceeded"
}
```

---

### Client-Side Error Handling

**Example (axios)**:
```javascript
try {
  const response = await axios.post('/upload-documents', formData);
  console.log(response.data);
} catch (error) {
  if (error.response) {
    // Server responded with error status
    console.error('Error:', error.response.data.detail);
    displayErrorToUser(error.response.data.detail);
  } else if (error.request) {
    // No response received (network error)
    console.error('Network error:', error.message);
    displayErrorToUser('Network error. Please check your connection.');
  } else {
    // Request setup error
    console.error('Error:', error.message);
  }
}
```

**Example (fetch)**:
```javascript
const response = await fetch('/upload-documents', {
  method: 'POST',
  body: formData
});

if (!response.ok) {
  const errorData = await response.json();
  console.error('Error:', errorData.detail);
  throw new Error(errorData.detail);
}

const data = await response.json();
```

---

## Rate Limiting

### Current Implementation

**No rate limiting implemented** in base version.

### Recommended Rate Limits (Production)

| Endpoint | Limit | Window | Purpose |
|----------|-------|--------|---------|
| /upload-documents | 5 requests | 1 minute | Prevent abuse of document processing |
| /start-interview | 10 requests | 1 minute | Prevent session creation spam |
| /stream-answer | 60 requests | 1 minute | Normal interview pace (1 per second max) |
| /ws/transcribe | 10 connections | 1 minute per IP | Prevent WebSocket abuse |

### Implementation Example

**Using slowapi**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/upload-documents")
@limiter.limit("5/minute")
async def upload_documents(request: Request):
    ...
```

**Rate Limit Response** (429 Too Many Requests):
```json
{
  "detail": "Rate limit exceeded: 5 per 1 minute"
}
```

---

## Interactive API Documentation

### Swagger UI

**URL**: `http://localhost:8000/docs`

**Features**:
- Interactive API testing
- Request/response examples
- Schema documentation
- "Try it out" functionality

**Example Usage**:
1. Navigate to http://localhost:8000/docs
2. Click on an endpoint (e.g., `POST /upload-documents`)
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. View response

---

### ReDoc

**URL**: `http://localhost:8000/redoc`

**Features**:
- Clean, three-panel layout
- Search functionality
- Downloadable OpenAPI spec
- Code samples

---

## WebSocket Testing Tools

### wscat (CLI)

**Install**:
```bash
npm install -g wscat
```

**Connect**:
```bash
wscat -c ws://localhost:8000/ws/transcribe/test-session-id
```

**Send JSON**:
```
> {"type": "stop_recording"}
```

---

### Browser DevTools

**Network Tab**:
1. Open DevTools (F12)
2. Go to Network tab
3. Filter by "WS" (WebSocket)
4. Click on connection
5. View "Messages" tab

---

## SSE Testing Tools

### curl

**Basic Test**:
```bash
curl -N http://localhost:8000/stream-answer \
  ?session_id=test \
  &answer=test
```

**With Headers**:
```bash
curl -N -H "Accept: text/event-stream" \
  http://localhost:8000/stream-answer?session_id=test&answer=test
```

---

## Summary

The xquizit API provides:

- **REST endpoints** for document upload, interview initialization, and status checks
- **WebSocket protocol** for real-time audio transcription
- **Server-Sent Events** for streaming interview responses
- **Consistent error handling** with descriptive messages
- **Type-safe data models** using Pydantic
- **Interactive documentation** via Swagger UI and ReDoc

For production use, implement:
- Authentication and authorization
- Rate limiting
- Request validation
- Monitoring and logging
- CORS restrictions
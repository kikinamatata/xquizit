// API configuration

// Local development (default)
export const API_BASE_URL = 'http://localhost:56158';
export const WS_BASE_URL = 'ws://localhost:56158';

// TryCloudflare (for external access)
// After starting TryCloudflare tunnels, replace the URLs below with the random URL
// shown in the "Backend Tunnel" window, then uncomment and comment out localhost above:
//
//export const API_BASE_URL = 'https://infants-fonts-savannah-offering.trycloudflare.com';
//export const WS_BASE_URL = 'wss://infants-fonts-savannah-offering.trycloudflare.com';
  //
// Note: TryCloudflare URLs change on each restart, so you'll need to update this
// each time you start the tunnels.

export const API_ENDPOINTS = {
  UPLOAD_DOCUMENTS: '/upload-documents',
  START_INTERVIEW: '/start-interview',
  TRANSCRIBE_AUDIO: '/transcribe-audio', // Legacy endpoint (kept for backward compatibility)
  SUBMIT_ANSWER: '/submit-answer',
  STREAM_ANSWER: '/stream-answer',
  INTERVIEW_STATUS: '/interview-status',
  WS_TRANSCRIBE: '/ws/transcribe', // WebSocket endpoint for real-time transcription
};

export const INTERVIEW_DURATION_MS = 45 * 60 * 1000; // 45 minutes in milliseconds
export const INTERVIEW_DURATION_SECONDS = 45 * 60; // 45 minutes in seconds

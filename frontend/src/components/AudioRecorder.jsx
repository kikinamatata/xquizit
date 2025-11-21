import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import { API_BASE_URL, WS_BASE_URL } from '../config';
import './AudioRecorder.css';

const AudioRecorder = forwardRef(({ sessionId, onTranscriptionComplete, onLiveTranscriptUpdate, disabled }, ref) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isProcessingFinal, setIsProcessingFinal] = useState(false);
  const [error, setError] = useState(null);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [showTranscriptEditor, setShowTranscriptEditor] = useState(false);
  const [editableTranscript, setEditableTranscript] = useState('');
  const [finalTranscript, setFinalTranscript] = useState([]);      // Array of completed segments
  const [lastInterimSegment, setLastInterimSegment] = useState(null);  // Current interim preview

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const websocketRef = useRef(null);
  const finalTranscriptRef = useRef('');
  const isRecordingRef = useRef(false); // Ref to track recording state for callbacks
  const stopRecordingTimestampRef = useRef(null); // Track when stop button is pressed

  // Update live transcript display whenever finalTranscript or lastInterimSegment changes
  useEffect(() => {
    const finalText = finalTranscript.map(s => s.text).join(' ');
    const displayText = lastInterimSegment
      ? finalText + ' ' + lastInterimSegment.text
      : finalText;
    setLiveTranscript(displayText.trim());
  }, [finalTranscript, lastInterimSegment]);

  // Emit live transcript updates to parent
  useEffect(() => {
    if (onLiveTranscriptUpdate) {
      onLiveTranscriptUpdate(liveTranscript);
    }
  }, [liveTranscript, onLiveTranscriptUpdate]);

  // Expose imperative methods to parent via ref
  useImperativeHandle(ref, () => ({
    startRecording: () => startRecording(),
    stopRecording: () => stopRecording(),
    isRecording: () => isRecording,
    isProcessingFinal: () => isProcessingFinal,
  }));

  // Cleanup WebSocket and audio context on unmount
  useEffect(() => {
    return () => {
      // Cleanup audio processing
      if (mediaRecorderRef.current) {
        if (mediaRecorderRef.current.stop) {
          mediaRecorderRef.current.stop();
        }
      }
      // Cleanup microphone stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      // Cleanup WebSocket
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    };
  }, []);

  const startRecording = async () => {
    // Prevent starting if already recording or transcribing
    if (isRecording || isTranscribing) {
      console.warn('Already recording or transcribing');
      return;
    }

    // Clean up any previous audio context before starting new one
    if (mediaRecorderRef.current?.stop) {
      console.log('Cleaning up previous audio context before starting new recording');
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        console.warn('Error cleaning up previous audio context:', e);
      }
      mediaRecorderRef.current = null;
    }

    try {
      setError(null);
      setLiveTranscript('');
      finalTranscriptRef.current = '';
      setEditableTranscript('');
      setShowTranscriptEditor(false);  // Don't show editor (auto-submit mode)
      setFinalTranscript([]);
      setLastInterimSegment(null);

      // Get microphone access with 16kHz mono for PCM conversion
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,  // 16kHz
          channelCount: 1,    // Mono
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // Initialize WebSocket connection
      const wsUrl = `${WS_BASE_URL}/ws/transcribe/${sessionId}`;
      console.log('Connecting to WebSocket:', wsUrl);

      const ws = new WebSocket(wsUrl);
      websocketRef.current = ws;

      // Add 10-second connection timeout
      let connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          console.error('WebSocket connection timeout');
          ws.close();
          setError('Connection timeout. Please check if the server is running.');
          // Cleanup on timeout
          if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
          }
        }
      }, 10000);

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('✅ WebSocket connected, readyState:', ws.readyState, 'waiting for ready signal...');
        // Don't set isTranscribing yet - wait for "ready" message
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);

          // Handle ready message
          if (data.type === 'ready') {
            console.log('✅ Transcription service ready - backend is listening for audio');
            setIsTranscribing(true);
            return;
          }

          // Handle transcript messages
          if (data.type === 'transcript') {
            const segment = data.segment;
            const isFinal = segment?.is_final || false;
            const text = segment?.text || '';
            const start = segment?.start || "0.000";

            if (isFinal) {
              // Final segment - apply timestamp-based deduplication
              setFinalTranscript(prev => {
                // First segment - always add
                if (prev.length === 0) {
                  console.log(`✅ Added first segment: "${text}"`);
                  return [segment];
                }

                // Timestamp-based deduplication: only add if start >= last end
                const lastSeg = prev[prev.length - 1];
                const currentStart = parseFloat(start);
                const lastEnd = parseFloat(lastSeg.end);

                if (currentStart >= lastEnd) {
                  console.log(`✅ Added segment (${start} >= ${lastSeg.end}): "${text}"`);
                  return [...prev, segment];
                } else {
                  console.log(`⏭️  Skipped duplicate segment (${start} < ${lastSeg.end}): "${text}"`);
                  return prev;  // Skip overlapping segment
                }
              });

              // Clear interim segment when final arrives
              setLastInterimSegment(null);
            } else {
              // Interim segment - update preview only (don't add to final transcript)
              console.log(`🔄 Interim segment: "${text}"`);
              setLastInterimSegment(segment);
            }
          }

          // Handle complete message (backend finished final flush)
          if (data.type === 'complete') {
            console.log('✅ Backend sent "complete" - all segments received, auto-submitting...');

            // Close WebSocket
            if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
              websocketRef.current.close();
            }

            // Use setState callback to ensure we read the latest state
            // This is critical because React state updates are asynchronous
            setFinalTranscript(current => {
              const finalText = current.map(s => s.text).join(' ').trim();

              if (finalText) {
                console.log('🚀 Auto-submitting transcription:', finalText.substring(0, 50) + '...');

                // Add try-catch for error handling
                try {
                  onTranscriptionComplete(finalText, stopRecordingTimestampRef.current);
                } catch (error) {
                  console.error('❌ Error in onTranscriptionComplete:', error);
                  setError('Failed to submit transcription. Please try again.');
                }
              } else {
                console.error('❌ No speech detected in transcription');
                setError('No speech detected. Please try again.');
              }

              // Return empty array to clear state
              return [];
            });

            // Cleanup other states
            setLiveTranscript('');
            setEditableTranscript('');
            setIsTranscribing(false);
            setIsProcessingFinal(false);
            setLastInterimSegment(null);

            return; // Early return after processing
          }

          // Handle error messages
          if (data.type === 'error') {
            console.error('Transcription error:', data.error);
            setError(data.error || 'Transcription error occurred');

            stopRecording();
          }

        } catch (err) {
          console.error('Error parsing WebSocket message:', err, event.data);
        }
      };

      ws.onerror = (error) => {
        clearTimeout(connectionTimeout);
        console.error('WebSocket error:', error);
        setError('Connection error. Please check your network and try again.');
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket closed:', event.code, event.reason);
        setIsTranscribing(false);
        setIsProcessingFinal(false);

        // Check if close was unexpected
        if (event.code !== 1000 && event.code !== 1001 && isRecordingRef.current) {
          // Unexpected disconnect (not normal close or going away)
          console.error('❌ Unexpected WebSocket close:', event.code, event.reason || 'No reason provided');
          setError(`Connection lost during transcription (code: ${event.code}). Please try again.`);

          // Reset states
          setFinalTranscript([]);
          setLastInterimSegment(null);
          setLiveTranscript('');
          setEditableTranscript('');
          return;
        }

        // Normal close - finalize last interim segment if exists
        if (lastInterimSegment) {
          console.log('WebSocket closed - finalizing last interim segment:', lastInterimSegment.text);
          setFinalTranscript(prev => [...prev, { ...lastInterimSegment, is_final: true }]);
          setLastInterimSegment(null);
        }

        // Move to editable state with final transcript
        // Use a small timeout to ensure finalTranscript state has updated
        setTimeout(() => {
          setFinalTranscript(current => {
            const finalText = current.map(s => s.text).join(' ').trim();
            if (finalText) {
              setEditableTranscript(finalText);
            }
            return current;
          });
        }, 100);
      };

      // Create Web Audio context for PCM conversion
      const audioContext = new AudioContext({ sampleRate: 16000 });
      console.log('🎤 AudioContext created, state:', audioContext.state, 'sampleRate:', audioContext.sampleRate);

      // Resume audio context (browsers auto-suspend it)
      if (audioContext.state === 'suspended') {
        console.log('⚠️ AudioContext was suspended, resuming...');
        audioContext.resume().then(() => {
          console.log('✅ AudioContext resumed, state:', audioContext.state);
        });
      }

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1); // 4096 samples (~256ms chunks)
      console.log('🎧 ScriptProcessor created, buffer size: 4096 samples (~256ms)');

      let chunkCount = 0;
      processor.onaudioprocess = (e) => {
        chunkCount++;

        // Log every 10th callback to avoid spam
        if (chunkCount % 10 === 0) {
          console.log(`🔊 onaudioprocess callback #${chunkCount} - WS readyState: ${websocketRef.current?.readyState}, isRecording: ${isRecordingRef.current}`);
        }

        // Only send if WebSocket is open and recording (use ref to avoid closure issues)
        if (websocketRef.current?.readyState === WebSocket.OPEN && isRecordingRef.current) {
          const inputData = e.inputBuffer.getChannelData(0); // float32 [-1, 1]

          // Convert float32 to int16 PCM
          const int16Data = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            // Clamp to [-1, 1] and convert to int16 range
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Data[i] = s < 0 ? s * 32768 : s * 32767;
          }

          // Send raw PCM data as ArrayBuffer
          if (chunkCount % 10 === 0) {
            console.log(`📤 Sending audio chunk #${chunkCount} (${int16Data.buffer.byteLength} bytes)`);
          }
          websocketRef.current.send(int16Data.buffer);
        } else {
          // Log why we're NOT sending
          if (chunkCount % 10 === 0) {
            if (websocketRef.current?.readyState !== WebSocket.OPEN) {
              console.warn(`⚠️ NOT sending chunk #${chunkCount}: WebSocket not open (state: ${websocketRef.current?.readyState})`);
            } else if (!isRecordingRef.current) {
              console.warn(`⚠️ NOT sending chunk #${chunkCount}: isRecordingRef.current is false`);
            }
          }
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      // Store references for cleanup
      mediaRecorderRef.current = {
        audioContext,
        source,
        processor,
        stop: () => {
          processor.disconnect();
          source.disconnect();
          audioContext.close();
        }
      };

      // Update both state and ref (ref is used in callbacks to avoid closure issues)
      setIsRecording(true);
      isRecordingRef.current = true;
      console.log('✅ Recording started - isRecordingRef.current set to true');

    } catch (err) {
      console.error('Error accessing microphone:', err);
      if (err.name === 'NotAllowedError') {
        setError('Microphone access denied. Please allow microphone permissions.');
      } else if (err.name === 'NotFoundError') {
        setError('No microphone found. Please connect a microphone.');
      } else {
        setError('Error accessing microphone. Please try again.');
      }
    }
  };

  const stopRecording = () => {
    // Track timestamp for latency measurement
    stopRecordingTimestampRef.current = Date.now();
    console.log(`⏱️  [Stop Recording] Timestamp: ${new Date(stopRecordingTimestampRef.current).toISOString()}`);

    // Update both state and ref
    setIsRecording(false);
    isRecordingRef.current = false;
    setIsProcessingFinal(true);
    console.log('🛑 Recording stopped - isRecordingRef.current set to false');

    // Stop Web Audio processing
    if (mediaRecorderRef.current) {
      if (mediaRecorderRef.current.stop) {
        mediaRecorderRef.current.stop();
        console.log('Audio processing stopped');
      }
    }

    // Stop microphone stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
        console.log('Microphone track stopped');
      });
      streamRef.current = null;
    }

    // Send stop_recording signal to backend to trigger final flush
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      const stopMessage = {
        type: 'stop_recording',
        timestamp: stopRecordingTimestampRef.current,
      };
      websocketRef.current.send(JSON.stringify(stopMessage));
      console.log('📤 Sent stop_recording signal to backend');
    }

    // DON'T close WebSocket - wait for backend to send "complete" message
    // Just show processing indicator and wait for final segments
    console.log('🛑 Recording stopped - waiting for final segments from backend...');
  };

  // Handle transcript text changes (manual editing)
  const handleTranscriptChange = (e) => {
    setEditableTranscript(e.target.value);
  };

  // Submit the final transcript
  const handleSubmitTranscript = () => {
    const finalText = editableTranscript || liveTranscript;

    if (!finalText.trim()) {
      setError('Please record or type your answer before submitting.');
      return;
    }

    setShowTranscriptEditor(false);

    // Send the final transcript to the parent component
    onTranscriptionComplete(finalText.trim());

    // Reset states
    setLiveTranscript('');
    setEditableTranscript('');
    finalTranscriptRef.current = '';
  };

  // Cancel recording and transcript
  const handleCancelTranscript = () => {
    // Stop recording if active
    if (isRecording) {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        isRecordingRef.current = false;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }

    // Close WebSocket immediately (user is canceling, don't wait for "complete")
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.close();
      websocketRef.current = null;
    }

    // Reset all states
    setShowTranscriptEditor(false);
    setLiveTranscript('');
    setEditableTranscript('');
    finalTranscriptRef.current = '';
    setFinalTranscript([]);
    setLastInterimSegment(null);
    setError(null);
    setIsTranscribing(false);
    setIsProcessingFinal(false);
  };

  return (
    <div className="audio-recorder">
      {error && (
        <div className="audio-recorder-error">
          {error}
        </div>
      )}

      {/* Live transcription preview - shown only during recording */}
      {isRecording && (
        <div className="transcript-editor">
          <div className="transcript-header">
            <span>Your Answer</span>
            {isTranscribing && (
              <span className="transcribing-badge">
                <div className="pulse-indicator"></div>
                Transcribing in real-time...
              </span>
            )}
          </div>

          <div className="live-transcript-preview">
            <div className="preview-label">
              <span>Live Transcription</span>
              <div className="pulse-indicator-recording"></div>
            </div>
            <div className="preview-content">
              {liveTranscript || 'Speak now... your words will appear here'}
            </div>
          </div>
        </div>
      )}

      <div className="audio-recorder-controls">
        {!isRecording && !isTranscribing && (
          <button
            className="record-button"
            onClick={startRecording}
            disabled={disabled}
            aria-label="Start recording"
          >
            <div className="record-icon"></div>
            <span>Record Answer</span>
          </button>
        )}

        {isRecording && (
          <button
            className="stop-button"
            onClick={stopRecording}
            aria-label="Finish recording and submit"
          >
            <div className="stop-icon"></div>
            <span>I'm Done</span>
          </button>
        )}

        {isTranscribing && (
          <div className="transcribing-indicator">
            <div className="spinner"></div>
            <span>Processing...</span>
          </div>
        )}

        {isProcessingFinal && !isTranscribing && (
          <div className="processing-final-indicator">
            <div className="spinner"></div>
            <span>Processing final audio...</span>
          </div>
        )}
      </div>

      {isRecording && (
        <div className="recording-indicator">
          <div className="recording-pulse"></div>
          <span>Recording in progress... speak now</span>
        </div>
      )}
    </div>
  );
});

// Add display name for debugging
AudioRecorder.displayName = 'AudioRecorder';

export default AudioRecorder;

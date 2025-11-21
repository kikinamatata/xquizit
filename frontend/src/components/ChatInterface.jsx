import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL, API_ENDPOINTS, INTERVIEW_DURATION_SECONDS } from '../config';
import Timer from './Timer';
import InterviewerPanel from './InterviewerPanel';
import UserPanel from './UserPanel';
import AudioRecorder from './AudioRecorder';
import useAudioPlayback from '../hooks/useAudioPlayback';
import './ChatInterface.css';

const ChatInterface = ({ sessionId, firstQuestion, firstQuestionAudio, startTime }) => {
  const [isWaitingForQuestion, setIsWaitingForQuestion] = useState(false);
  const [interviewComplete, setInterviewComplete] = useState(false);
  const [currentQuestionNumber, setCurrentQuestionNumber] = useState(1);
  const [streamingMessage, setStreamingMessage] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  // Two-panel state
  const [currentTurn, setCurrentTurn] = useState('interviewer'); // 'interviewer' | 'user'
  const [isInterviewerSpeaking, setIsInterviewerSpeaking] = useState(false);
  const [interviewerMessages, setInterviewerMessages] = useState([]);
  const [userMessages, setUserMessages] = useState([]);
  const [currentUserTranscript, setCurrentUserTranscript] = useState('');

  // Processing answer state for UI feedback
  const [isProcessingAnswer, setIsProcessingAnswer] = useState(false);
  const [isProcessingTranscription, setIsProcessingTranscription] = useState(false);

  const messageIdCounter = useRef(1);
  const processingRef = useRef(false);  // Prevent race conditions
  const eventSourceRef = useRef(null);
  const streamingMessageIdRef = useRef(null);
  const stopRecordingTimestampRef = useRef(null);  // Track stop recording timestamp for latency
  const audioRecorderRef = useRef(null);
  const audioTimeoutRef = useRef(null);  // Track audio playback timeout
  const firstAudioQueuedRef = useRef(false);  // Prevent double-queuing in StrictMode

  // Audio playback completion callback
  const handleAudioPlaybackComplete = () => {
    // Clear any pending safety timeout
    if (audioTimeoutRef.current) {
      clearTimeout(audioTimeoutRef.current);
      audioTimeoutRef.current = null;
      console.log('🔕 Cleared safety timeout - audio completed naturally');
    }

    console.log('✅ Audio playback completed - switching turn to candidate');
    setIsInterviewerSpeaking(false);
    setCurrentTurn('user');

    // Auto-start recording with small delay
    setTimeout(() => {
      if (audioRecorderRef.current && !processingRef.current) {
        console.log('🎤 Auto-starting recording after audio completion');
        audioRecorderRef.current.startRecording();
      }
    }, 100);
  };

  // Audio playback hook with completion callback
  const {
    addAudioChunk,
    stopPlayback,
    reset: resetAudioPlayback,
    isPlaying,
    getQueueLength,
    startStreaming,
    endStreaming
  } = useAudioPlayback(handleAudioPlaybackComplete);

  // Initialize with first question in interviewer messages
  useEffect(() => {
    if (firstQuestion) {
      setInterviewerMessages([
        {
          id: messageIdCounter.current++,
          text: firstQuestion,
          timestamp: new Date(),
        },
      ]);
    }
  }, [firstQuestion]);

  // Play audio for first question
  useEffect(() => {
    if (firstQuestionAudio && firstQuestionAudio.length > 0 && !firstAudioQueuedRef.current) {
      // Mark as queued to prevent double-play in StrictMode
      firstAudioQueuedRef.current = true;

      // Queue audio immediately (no timeout needed - guard prevents double-queueing)
      console.log(`🔊 Queueing ${firstQuestionAudio.length} audio chunks for first question`);
      setIsInterviewerSpeaking(true); // Set speaking state when audio starts

      // Queue all audio chunks for playback
      firstQuestionAudio.forEach((audioChunk) => {
        addAudioChunk(audioChunk);  // Note: index parameter not needed by useAudioPlayback
      });
    }
  }, [firstQuestionAudio, addAudioChunk]);

  // Check interview time
  useEffect(() => {
    const checkTime = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      if (elapsed >= INTERVIEW_DURATION_SECONDS) {
        setInterviewComplete(true);
        clearInterval(checkTime);
      }
    }, 1000);

    return () => clearInterval(checkTime);
  }, [startTime]);

  // Cleanup EventSource and timeouts on unmount
  useEffect(() => {
    return () => {
      console.log('ChatInterface cleanup');

      // Clear audio timeout
      if (audioTimeoutRef.current) {
        clearTimeout(audioTimeoutRef.current);
        audioTimeoutRef.current = null;
      }

      // Close EventSource if still open
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }

      // Don't call stopPlayback here - let useAudioPlayback manage its own lifecycle
      // The hook has smart logic to protect active audio from being interrupted
    };
  }, []); // Empty deps - this effect sets up cleanup once and doesn't need updates

  const handleTranscriptionComplete = async (transcription, stopTimestamp = null) => {
    // Store stop recording timestamp for latency calculation
    if (stopTimestamp) {
      stopRecordingTimestampRef.current = stopTimestamp;
      console.log(`⏱️  [ChatInterface] Received stop timestamp: ${new Date(stopTimestamp).toISOString()}`);
    }

    // Reset transcription processing state
    setIsProcessingTranscription(false);

    // Prevent concurrent processing
    if (processingRef.current) {
      console.warn('Already processing an answer, ignoring new transcription');
      return;
    }

    processingRef.current = true;
    setIsProcessingAnswer(true);
    console.log('🔄 Processing answer started - UI feedback enabled');

    // Add candidate's answer to user messages
    const candidateMessage = {
      id: messageIdCounter.current++,
      text: transcription,
      timestamp: new Date(),
    };
    setUserMessages((prev) => [...prev, candidateMessage]);

    // Clear current user transcript
    setCurrentUserTranscript('');

    // Prepare for streaming response - switch to interviewer turn
    setCurrentTurn('interviewer');
    setIsWaitingForQuestion(true);
    setIsStreaming(true);
    setIsInterviewerSpeaking(true); // Set speaking state when response starts

    // Create a streaming message placeholder
    const streamingMsgId = messageIdCounter.current++;
    streamingMessageIdRef.current = streamingMsgId;

    const initialStreamingMessage = {
      id: streamingMsgId,
      text: '',
      timestamp: new Date(),
      isStreaming: true,
    };

    setStreamingMessage(initialStreamingMessage);

    // Close any existing EventSource before creating new one
    if (eventSourceRef.current) {
      console.log('Closing existing EventSource before creating new one');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    // Initiate streaming with EventSource
    try {
      // Build URL with query parameters
      const streamUrl = new URL(`${API_BASE_URL}${API_ENDPOINTS.STREAM_ANSWER}`);
      streamUrl.searchParams.append('session_id', sessionId);
      streamUrl.searchParams.append('answer', transcription);

      // Create EventSource connection
      const eventSource = new EventSource(streamUrl.toString());
      eventSourceRef.current = eventSource;

      let accumulatedText = '';
      let isConcluded = false;

      // Signal that audio streaming has started
      startStreaming();

      // Handle text_chunk events
      eventSource.addEventListener('text_chunk', (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.chunk) {
            accumulatedText += data.chunk;
            setStreamingMessage((prev) => ({
              ...prev,
              text: accumulatedText,
            }));
          }
        } catch (error) {
          console.error('Error parsing text_chunk:', error);
        }
      });

      // Handle audio_chunk events
      eventSource.addEventListener('audio_chunk', (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.audio) {
            addAudioChunk(data.audio);
          }
        } catch (error) {
          console.error('Error parsing audio_chunk:', error);
        }
      });

      // Handle metadata events
      eventSource.addEventListener('metadata', (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.is_concluded !== undefined) {
            isConcluded = data.is_concluded;
          }
        } catch (error) {
          console.error('Error parsing metadata:', error);
        }
      });

      // Handle done event
      eventSource.addEventListener('done', (event) => {
        console.log('📨 SSE stream completed - text and audio chunks finished arriving');

        // Calculate stop-to-question latency
        if (stopRecordingTimestampRef.current) {
          const questionReceivedTimestamp = Date.now();
          const totalLatencyMs = questionReceivedTimestamp - stopRecordingTimestampRef.current;
          console.log(`⏱️  [Question Received] Timestamp: ${new Date(questionReceivedTimestamp).toISOString()}`);
          console.log(`⏱️  [STOP-TO-QUESTION TOTAL LATENCY]: ${(totalLatencyMs / 1000).toFixed(3)}s (${totalLatencyMs}ms)`);

          // Reset timestamp for next question
          stopRecordingTimestampRef.current = null;
        }

        // Finalize the streaming message
        const finalMessage = {
          id: streamingMsgId,
          text: accumulatedText,
          timestamp: new Date(),
        };

        // Add to interviewer messages and clear streaming state
        setInterviewerMessages((prev) => [...prev, finalMessage]);
        setStreamingMessage(null);
        setIsStreaming(false);
        setIsWaitingForQuestion(false);

        // Update interview state
        if (isConcluded) {
          setInterviewComplete(true);
        } else {
          setCurrentQuestionNumber((prev) => prev + 1);
        }

        // Close EventSource
        eventSource.close();
        eventSourceRef.current = null;
        processingRef.current = false;
        setIsProcessingAnswer(false);

        // CRITICAL: Do NOT switch turns here - keep isInterviewerSpeaking=true
        // Let handleAudioPlaybackComplete callback handle turn switching
        console.log('✅ Stream bookkeeping complete');

        // Wait 100ms for final chunks to arrive, then end streaming
        setTimeout(() => {
          const queueLength = getQueueLength ? getQueueLength() : 0;
          const audioIsPlaying = isPlaying ? isPlaying() : false;

          console.log(`🔊 Audio status: queue=${queueLength}, playing=${audioIsPlaying}`);

          if (queueLength === 0 && !audioIsPlaying) {
            // No audio received - switch immediately
            console.warn('⚠️ No audio chunks received - switching turn immediately');
            handleAudioPlaybackComplete();
          } else {
            // Audio exists - end streaming mode, let natural completion handle turn switch
            console.log('📡 Ending streaming mode - completion will trigger naturally');
            endStreaming();

            // Set 30s safety timeout as fallback
            audioTimeoutRef.current = setTimeout(() => {
              console.warn('⚠️ Audio safety timeout (30s) - forcing turn switch');
              handleAudioPlaybackComplete();
            }, 30000);
          }
        }, 100); // 100ms delay for final chunks
      });

      // Fallback: Handle generic messages (in case named events don't work)
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Check if this is a fallback (named handlers should have already processed it)
          console.log('[Fallback Handler] Received generic message:', data.type || 'unknown');

          switch(data.type) {
            case 'text_chunk':
              if (data.chunk || data.content) {
                accumulatedText += (data.chunk || data.content);
                setStreamingMessage((prev) => ({
                  ...prev,
                  text: accumulatedText,
                }));
              }
              break;

            case 'audio_chunk':
              if (data.audio) {
                addAudioChunk(data.audio);
              }
              break;

            case 'metadata':
              if (data.is_concluded !== undefined) {
                isConcluded = data.is_concluded;
              }
              break;

            case 'done':
              console.log('[Fallback] 📨 SSE stream completed - text and audio chunks finished arriving');

              // Calculate stop-to-question latency
              if (stopRecordingTimestampRef.current) {
                const questionReceivedTimestamp = Date.now();
                const totalLatencyMs = questionReceivedTimestamp - stopRecordingTimestampRef.current;
                console.log(`⏱️  [Question Received - Fallback] Timestamp: ${new Date(questionReceivedTimestamp).toISOString()}`);
                console.log(`⏱️  [STOP-TO-QUESTION TOTAL LATENCY]: ${(totalLatencyMs / 1000).toFixed(3)}s (${totalLatencyMs}ms)`);

                // Reset timestamp for next question
                stopRecordingTimestampRef.current = null;
              }

              // Finalize the streaming message
              const finalMessage = {
                id: streamingMsgId,
                text: accumulatedText,
                timestamp: new Date(),
              };

              // Add to interviewer messages and clear streaming state
              setInterviewerMessages((prev) => [...prev, finalMessage]);
              setStreamingMessage(null);
              setIsStreaming(false);
              setIsWaitingForQuestion(false);

              // Update interview state
              if (isConcluded) {
                setInterviewComplete(true);
              } else {
                setCurrentQuestionNumber((prev) => prev + 1);
              }

              // Close EventSource
              eventSource.close();
              eventSourceRef.current = null;
              processingRef.current = false;
              setIsProcessingAnswer(false);

              // CRITICAL: Do NOT switch turns here - keep isInterviewerSpeaking=true
              // Let handleAudioPlaybackComplete callback handle turn switching
              console.log('[Fallback] ✅ Stream bookkeeping complete');

              // Wait 100ms for final chunks to arrive, then end streaming
              setTimeout(() => {
                const queueLength = getQueueLength ? getQueueLength() : 0;
                const audioIsPlaying = isPlaying ? isPlaying() : false;

                console.log(`[Fallback] 🔊 Audio status: queue=${queueLength}, playing=${audioIsPlaying}`);

                if (queueLength === 0 && !audioIsPlaying) {
                  // No audio received - switch immediately
                  console.warn('[Fallback] ⚠️ No audio chunks received - switching turn immediately');
                  handleAudioPlaybackComplete();
                } else {
                  // Audio exists - end streaming mode, let natural completion handle turn switch
                  console.log('[Fallback] 📡 Ending streaming mode - completion will trigger naturally');
                  endStreaming();

                  // Set 30s safety timeout as fallback
                  audioTimeoutRef.current = setTimeout(() => {
                    console.warn('[Fallback] ⚠️ Audio safety timeout (30s) - forcing turn switch');
                    handleAudioPlaybackComplete();
                  }, 30000);
                }
              }, 100); // 100ms delay for final chunks

              break;

            default:
              console.warn('[Fallback] Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('[Fallback] Error parsing generic message:', error);
        }
      };

      // Handle errors
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);

        // Close the connection
        eventSource.close();
        eventSourceRef.current = null;

        // Clear any pending audio timeout
        if (audioTimeoutRef.current) {
          clearTimeout(audioTimeoutRef.current);
          audioTimeoutRef.current = null;
        }

        // If we have accumulated text, save it
        if (accumulatedText) {
          const finalMessage = {
            id: streamingMsgId,
            text: accumulatedText,
            timestamp: new Date(),
          };
          setInterviewerMessages((prev) => [...prev, finalMessage]);
        } else {
          // Show error message if no text was received
          const errorMessage = {
            id: streamingMsgId,
            text: 'There was an error receiving the response. Please try again.',
            timestamp: new Date(),
          };
          setInterviewerMessages((prev) => [...prev, errorMessage]);
        }

        // Clean up streaming state
        setStreamingMessage(null);
        setIsStreaming(false);
        setIsWaitingForQuestion(false);
        processingRef.current = false;
        setIsProcessingAnswer(false);
        console.log('✅ Processing answer completed (error recovery)');
        stopPlayback();

        // Switch turn back to user to prevent stuck state
        console.log('⚠️ Network error - switching turn back to user');
        setIsInterviewerSpeaking(false);
        setCurrentTurn('user');

        // Try to start recording after a delay (allows UI to stabilize)
        setTimeout(() => {
          if (audioRecorderRef.current && !processingRef.current) {
            console.log('🎤 Attempting to start recording after network error recovery');
            audioRecorderRef.current.startRecording();
          }
        }, 500); // 500ms delay to allow UI to update
      };
    } catch (err) {
      console.error('Error initiating stream:', err);

      // Add error message
      const errorMessage = {
        id: messageIdCounter.current++,
        text: 'There was an error processing your answer. Please try again.',
        timestamp: new Date(),
      };
      setInterviewerMessages((prev) => [...prev, errorMessage]);

      // Clean up state
      setStreamingMessage(null);
      setIsStreaming(false);
      setIsWaitingForQuestion(false);
      processingRef.current = false;
      setIsProcessingAnswer(false);
      console.log('✅ Processing answer completed (exception caught)');
      stopPlayback();
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h1>Screening Interview</h1>
        <div className="interview-progress">
          Question {currentQuestionNumber}
        </div>
      </div>

      <Timer startTime={startTime} totalDuration={INTERVIEW_DURATION_SECONDS} />

      {!interviewComplete ? (
        <div className="interview-panels">
          <InterviewerPanel
            messages={interviewerMessages}
            streamingText={streamingMessage?.text || ''}
            isStreaming={isStreaming}
            isSpeaking={isInterviewerSpeaking}
            isActive={currentTurn === 'interviewer'}
          />

          <UserPanel
            messages={userMessages}
            liveTranscript={currentUserTranscript}
            isRecording={currentTurn === 'user'}
            isActive={currentTurn === 'user'}
            onDoneClick={() => {
              if (audioRecorderRef.current) {
                console.log('🛑 User clicked "I\'m Done" - stopping recording');
                setIsProcessingTranscription(true);
                audioRecorderRef.current.stopRecording();
              }
            }}
            disabled={processingRef.current}
            isProcessingAnswer={isProcessingAnswer}
            isProcessingTranscription={isProcessingTranscription}
          />
        </div>
      ) : (
        <div className="interview-complete-container">
          <div className="interview-complete">
            <div className="complete-icon">✓</div>
            <h2>Interview Complete</h2>
            <p>Thank you for completing the screening interview. Your responses have been recorded.</p>
          </div>
        </div>
      )}

      {/* Hidden AudioRecorder - controlled programmatically via ref */}
      {!interviewComplete && (
        <AudioRecorder
          ref={audioRecorderRef}
          sessionId={sessionId}
          onTranscriptionComplete={handleTranscriptionComplete}
          onLiveTranscriptUpdate={setCurrentUserTranscript}
          disabled={isWaitingForQuestion || interviewComplete}
        />
      )}
    </div>
  );
};

export default ChatInterface;

import React, { useEffect, useRef } from 'react';
import SpeakingIndicator from './SpeakingIndicator';
import './UserPanel.css';

/**
 * UserPanel Component
 *
 * Displays the candidate's side of the interview interface.
 * Shows all candidate responses with scrollable history and live transcription.
 *
 * @param {Array} messages - Array of candidate messages {text, timestamp}
 * @param {string} liveTranscript - Current live transcription text
 * @param {boolean} isRecording - Whether user is currently speaking/recording
 * @param {boolean} isActive - Whether this panel is active (user's turn)
 * @param {Function} onDoneClick - Callback when "I'm Done" button is clicked
 * @param {boolean} disabled - Whether controls are disabled
 * @param {boolean} isProcessingAnswer - Whether answer is being processed
 * @param {boolean} isProcessingTranscription - Whether transcription is being processed
 */
const UserPanel = ({
  messages = [],
  liveTranscript = '',
  isRecording = false,
  isActive = true,
  onDoneClick,
  disabled = false,
  isProcessingAnswer = false,
  isProcessingTranscription = false
}) => {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, liveTranscript]);

  return (
    <div
      className={`user-panel ${!isActive ? 'panel-inactive' : ''}`}
      role="region"
      aria-label="Candidate panel"
      aria-busy={isRecording}
    >
      {/* Avatar Section */}
      <div className="panel-header">
        <div className="avatar-container">
          <div className="avatar avatar--user">
            <SpeakingIndicator isActive={isRecording} variant="glow" />
            <span className="avatar-initial">C</span>
          </div>
          <div className="panel-info">
            <h3 className="panel-title">You (Candidate)</h3>
            {isRecording && (
              <span className="status-indicator status-indicator--recording">
                Recording...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Messages Section */}
      <div className="panel-messages">
        {messages.length === 0 && !liveTranscript && (
          <div className="empty-state">
            <p>Your responses will appear here...</p>
          </div>
        )}

        {/* Previous Messages */}
        {messages.map((message, index) => (
          <div key={`user-msg-${index}`} className="message-bubble message-bubble--user">
            <div className="message-content">{message.text}</div>
            {message.timestamp && (
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            )}
          </div>
        ))}

        {/* Live Transcript Area */}
        {isRecording && (
          <div className="live-transcript-container">
            <div className="live-transcript-header">
              <span className="recording-dot"></span>
              Live Transcription
            </div>
            <div
              className="live-transcript"
              aria-live="polite"
              aria-atomic="false"
            >
              {liveTranscript || 'Speak now... your words will appear here'}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Action Button */}
      {isRecording && (
        <div className="panel-actions">
          <button
            className={`done-button ${(isProcessingAnswer || isProcessingTranscription) ? 'done-button--processing' : ''}`}
            onClick={onDoneClick}
            disabled={disabled || !liveTranscript || isProcessingAnswer || isProcessingTranscription}
            aria-label="Finish recording and submit response"
          >
            {isProcessingTranscription ? (
              <>
                <span className="spinner-small"></span>
                Processing transcription...
              </>
            ) : isProcessingAnswer ? (
              <>
                <span className="spinner-small"></span>
                Processing answer...
              </>
            ) : (
              <>
                <span className="done-button-icon">✓</span>
                I'm Done
              </>
            )}
          </button>
          <p className="action-hint">
            {isProcessingTranscription ? 'Finalizing your response...' : isProcessingAnswer ? 'Please wait...' : 'Click when you\'ve finished answering'}
          </p>
        </div>
      )}
    </div>
  );
};

export default UserPanel;

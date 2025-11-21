import React, { useEffect, useRef } from 'react';
import SpeakingIndicator from './SpeakingIndicator';
import './InterviewerPanel.css';

/**
 * InterviewerPanel Component
 *
 * Displays the interviewer's side of the interview interface.
 * Shows all interviewer questions with scrollable history.
 *
 * @param {Array} messages - Array of interviewer messages {text, timestamp}
 * @param {string} streamingText - Current streaming message text (if any)
 * @param {boolean} isStreaming - Whether text is currently streaming
 * @param {boolean} isSpeaking - Whether interviewer is speaking (streaming OR audio playing)
 * @param {boolean} isActive - Whether this panel is active (interviewer's turn)
 */
const InterviewerPanel = ({
  messages = [],
  streamingText = '',
  isStreaming = false,
  isSpeaking = false,
  isActive = true
}) => {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  return (
    <div
      className={`interviewer-panel ${!isActive ? 'panel-inactive' : ''}`}
      role="region"
      aria-label="Interviewer panel"
      aria-busy={isStreaming}
    >
      {/* Avatar Section */}
      <div className="panel-header">
        <div className="avatar-container">
          <div className="avatar avatar--interviewer">
            <SpeakingIndicator isActive={isSpeaking} variant="ring" />
            <span className="avatar-initial">I</span>
          </div>
          <div className="panel-info">
            <h3 className="panel-title">Interviewer</h3>
            {isSpeaking && (
              <span className="status-indicator status-indicator--speaking">
                Speaking...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Messages Section */}
      <div className="panel-messages">
        {messages.length === 0 && !streamingText && (
          <div className="empty-state">
            <p>Waiting for interview to begin...</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={`interviewer-msg-${index}`} className="message-bubble message-bubble--interviewer">
            <div className="message-content">{message.text}</div>
            {message.timestamp && (
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            )}
          </div>
        ))}

        {streamingText && (
          <div className="message-bubble message-bubble--interviewer message-bubble--streaming">
            <div className="message-content">
              {streamingText}
              {isStreaming && <span className="streaming-cursor">|</span>}
            </div>
            <div className="streaming-badge">
              <span className="streaming-dot"></span>
              Streaming...
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default InterviewerPanel;

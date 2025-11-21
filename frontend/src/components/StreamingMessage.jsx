import { useState, useEffect, useRef } from 'react';
import './StreamingMessage.css';

/**
 * StreamingMessage Component
 *
 * Displays a message that updates in real-time as text chunks arrive.
 * Shows a typing cursor while streaming is active.
 *
 * @param {string} sender - Message sender ('interviewer' or 'candidate')
 * @param {string} text - Current message text (may be incomplete)
 * @param {Date} timestamp - Message timestamp
 * @param {boolean} isStreaming - Whether the message is currently streaming
 * @param {boolean} showCursor - Whether to show typing cursor
 */
const StreamingMessage = ({ sender, text, timestamp, isStreaming = false, showCursor = false }) => {
  const isInterviewer = sender === 'interviewer';
  const [displayedText, setDisplayedText] = useState('');
  const previousTextRef = useRef('');

  // Update displayed text when text prop changes
  useEffect(() => {
    // Only update if text has actually changed
    if (text !== previousTextRef.current) {
      setDisplayedText(text);
      previousTextRef.current = text;
    }
  }, [text]);

  const formatTimestamp = (date) => {
    if (!date) return '';
    const d = new Date(date);
    return d.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div
      className={`message streaming-message ${isInterviewer ? 'message-interviewer' : 'message-candidate'} ${isStreaming ? 'is-streaming' : ''}`}
    >
      <div className="message-bubble">
        <div className="message-header">
          <span className="message-sender">
            {isInterviewer ? 'Interviewer' : 'You'}
          </span>
          {timestamp && !isStreaming && (
            <span className="message-timestamp">{formatTimestamp(timestamp)}</span>
          )}
        </div>
        <div className="message-text">
          {displayedText}
          {showCursor && isStreaming && (
            <span className="streaming-cursor" aria-hidden="true"></span>
          )}
        </div>
        {isStreaming && (
          <div className="streaming-indicator-badge">
            <span className="streaming-dot"></span>
            <span className="streaming-label">Streaming...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default StreamingMessage;

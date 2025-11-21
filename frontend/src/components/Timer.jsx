import { useState, useEffect } from 'react';
import './Timer.css';

const Timer = ({ startTime, totalDuration = 1800 }) => {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!startTime) return;

    const interval = setInterval(() => {
      const now = Date.now();
      const elapsed = Math.floor((now - startTime) / 1000);

      // Stop timer at max duration
      if (elapsed >= totalDuration) {
        setElapsedSeconds(totalDuration);
        clearInterval(interval);
      } else {
        setElapsedSeconds(elapsed);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime, totalDuration]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const progress = (elapsedSeconds / totalDuration) * 100;
  const isWarning = elapsedSeconds >= totalDuration * 0.8; // Warning at 80% (24 minutes)
  const isExpired = elapsedSeconds >= totalDuration;

  return (
    <div className={`timer ${isWarning ? 'timer-warning' : ''} ${isExpired ? 'timer-expired' : ''}`}>
      <div className="timer-content">
        <span className="timer-label">Time Elapsed:</span>
        <span className="timer-display">{formatTime(elapsedSeconds)}</span>
        <span className="timer-total">/ {formatTime(totalDuration)}</span>
      </div>
      <div className="timer-progress">
        <div
          className="timer-progress-bar"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
    </div>
  );
};

export default Timer;

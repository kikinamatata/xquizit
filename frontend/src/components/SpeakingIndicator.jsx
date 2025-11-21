import React from 'react';
import './SpeakingIndicator.css';

/**
 * SpeakingIndicator Component
 *
 * Displays an animated pulsing indicator to show when someone is speaking.
 * Supports two visual styles:
 * - 'ring': Expanding concentric rings (interviewer)
 * - 'glow': Pulsing glow effect (user/candidate)
 *
 * @param {boolean} isActive - Whether the indicator should be animated
 * @param {string} variant - 'ring' or 'glow' animation style
 * @param {string} color - CSS color for the indicator (default: blue for ring, green for glow)
 */
const SpeakingIndicator = ({ isActive = false, variant = 'ring', color }) => {
  if (!isActive) return null;

  const defaultColor = variant === 'ring' ? '#4a90e2' : '#28a745';
  const indicatorColor = color || defaultColor;

  if (variant === 'ring') {
    return (
      <div className="speaking-indicator speaking-indicator--ring">
        <div
          className="pulse-ring pulse-ring--1"
          style={{ borderColor: indicatorColor }}
        />
        <div
          className="pulse-ring pulse-ring--2"
          style={{ borderColor: indicatorColor }}
        />
        <div
          className="pulse-ring pulse-ring--3"
          style={{ borderColor: indicatorColor }}
        />
      </div>
    );
  }

  if (variant === 'glow') {
    return (
      <div
        className="speaking-indicator speaking-indicator--glow"
        style={{
          '--glow-color': indicatorColor,
          '--glow-color-alpha': `${indicatorColor}4D` // 30% opacity
        }}
      />
    );
  }

  return null;
};

export default SpeakingIndicator;

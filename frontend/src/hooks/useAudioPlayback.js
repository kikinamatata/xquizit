import { useRef, useCallback, useEffect } from 'react';

/**
 * Custom hook for handling streaming audio playback
 *
 * Features:
 * - Maintains a queue of audio chunks
 * - Converts base64 audio to playable Blob/Audio objects
 * - Implements seamless auto-play
 * - Handles memory cleanup (revoke object URLs)
 * - Supports playback state tracking
 * - Triggers callback when all audio playback completes
 *
 * @param {Function} onPlaybackComplete - Optional callback when all chunks finish playing
 */
const useAudioPlayback = (onPlaybackComplete) => {
  const audioQueueRef = useRef([]);
  const currentAudioRef = useRef(null);
  const isPlayingRef = useRef(false);
  const objectUrlsRef = useRef([]);
  const onPlaybackCompleteRef = useRef(onPlaybackComplete);
  const consecutiveErrorsRef = useRef(0);
  const isCommittedRef = useRef(false);  // Track if playback has been committed
  const isStreamingRef = useRef(false);  // Track if streaming is active (chunks arriving incrementally)

  // Keep callback ref up to date
  useEffect(() => {
    onPlaybackCompleteRef.current = onPlaybackComplete;
  }, [onPlaybackComplete]);

  /**
   * Convert base64 string to audio blob
   * @param {string} base64Audio - Base64 encoded audio data
   * @returns {Blob} Audio blob
   */
  const base64ToBlob = useCallback((base64Audio) => {
    // Remove data URL prefix if present (e.g., "data:audio/mpeg;base64,")
    const base64Data = base64Audio.includes(',')
      ? base64Audio.split(',')[1]
      : base64Audio;

    try {
      // Decode base64 string to binary
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);

      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Create blob with audio/wav MIME type (WAV with PCM)
      return new Blob([bytes], { type: 'audio/wav' });
    } catch (error) {
      console.error('Error converting base64 to blob:', error);
      return null;
    }
  }, []);

  /**
   * Play the next audio chunk in the queue
   */
  const playNextChunk = useCallback(() => {
    // If already playing or queue is empty, return
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    // Get next chunk from queue
    const nextChunk = audioQueueRef.current.shift();
    const blob = base64ToBlob(nextChunk);

    if (!blob) {
      console.error('Failed to create blob from audio chunk');
      // Try next chunk if available
      if (audioQueueRef.current.length > 0) {
        playNextChunk();
      }
      return;
    }

    // Create object URL for the blob
    const objectUrl = URL.createObjectURL(blob);
    objectUrlsRef.current.push(objectUrl);

    // Create and configure audio element
    const audio = new Audio(objectUrl);
    currentAudioRef.current = audio;
    isPlayingRef.current = true;

    // When chunk finishes playing, play next chunk
    audio.onended = () => {
      consecutiveErrorsRef.current = 0; // Reset error counter on success
      isPlayingRef.current = false;

      // Revoke object URL to free memory
      URL.revokeObjectURL(objectUrl);
      objectUrlsRef.current = objectUrlsRef.current.filter(url => url !== objectUrl);

      // Play next chunk if available
      if (audioQueueRef.current.length > 0) {
        playNextChunk();
      } else {
        // Queue is empty - only trigger completion if streaming has ended
        // This prevents premature completion during inter-chunk gaps
        if (!isStreamingRef.current) {
          // Streaming ended or never started - safe to complete
          if (onPlaybackCompleteRef.current) {
            console.log('✅ Last chunk finished, streaming ended - triggering completion');
            onPlaybackCompleteRef.current();
          }
        } else {
          // Still streaming - more chunks may arrive, don't complete yet
          console.log('⏳ Chunk finished but streaming active - waiting for more chunks');
        }
      }
    };

    // Handle playback errors
    audio.onerror = (error) => {
      console.error('Audio playback error:', error);
      isPlayingRef.current = false;
      consecutiveErrorsRef.current++;

      // Revoke object URL
      URL.revokeObjectURL(objectUrl);
      objectUrlsRef.current = objectUrlsRef.current.filter(url => url !== objectUrl);

      // If too many consecutive errors, abort all audio
      if (consecutiveErrorsRef.current >= 3) {
        console.error('❌ Too many consecutive audio errors (3+) - aborting audio playback');
        audioQueueRef.current = []; // Clear queue
        consecutiveErrorsRef.current = 0;

        // Trigger completion callback
        if (onPlaybackCompleteRef.current) {
          onPlaybackCompleteRef.current();
        }
        return;
      }

      // Try next chunk if available
      if (audioQueueRef.current.length > 0) {
        playNextChunk();
      } else {
        // Last chunk failed - check if this is a genuine error or cleanup-induced
        console.warn('⚠️ Last audio chunk error - determining cause');

        // Only trigger completion if we were committed to playback (genuine error)
        // If not committed, this error is likely from StrictMode cleanup/unmount
        if (isCommittedRef.current) {
          console.warn('⚠️ Genuine audio error detected (playback was committed) - triggering completion');
          if (onPlaybackCompleteRef.current) {
            onPlaybackCompleteRef.current();
          }
        } else {
          console.warn('⚠️ Error likely from cleanup/unmount (not committed) - not triggering completion');
        }
      }
    };

    // Start playing
    audio.play().catch(error => {
      console.error('Error playing audio:', error);
      isPlayingRef.current = false;

      // Revoke object URL
      URL.revokeObjectURL(objectUrl);
      objectUrlsRef.current = objectUrlsRef.current.filter(url => url !== objectUrl);

      // Check if browser blocked autoplay
      if (error.name === 'NotAllowedError') {
        console.warn('⚠️ Audio autoplay blocked by browser - clearing queue and switching turn');

        // Clear entire audio queue
        audioQueueRef.current = [];

        // Trigger completion callback to switch turn
        if (onPlaybackCompleteRef.current) {
          onPlaybackCompleteRef.current();
        }
      }
    });
  }, [base64ToBlob]);

  /**
   * Add an audio chunk to the playback queue
   * @param {string} base64AudioChunk - Base64 encoded audio chunk
   */
  const addAudioChunk = useCallback((base64AudioChunk) => {
    if (!base64AudioChunk) {
      return;
    }

    // Mark playback as committed (prevents premature cleanup)
    isCommittedRef.current = true;

    // Add to queue
    audioQueueRef.current.push(base64AudioChunk);

    // If not currently playing, start playback
    if (!isPlayingRef.current) {
      playNextChunk();
    }
  }, [playNextChunk]);

  /**
   * Stop playback and clear the queue
   */
  const stopPlayback = useCallback(() => {
    // Stop current audio
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }

    // Clear queue
    audioQueueRef.current = [];
    isPlayingRef.current = false;

    // Revoke all object URLs
    objectUrlsRef.current.forEach(url => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
  }, []);

  /**
   * Reset the playback system (clear queue, stop playback)
   */
  const reset = useCallback(() => {
    stopPlayback();
    isCommittedRef.current = false;  // Reset commitment flag
    consecutiveErrorsRef.current = 0;  // Reset error counter
    isStreamingRef.current = false;  // Reset streaming flag
  }, [stopPlayback]);

  /**
   * Check if audio is currently playing or queued
   * @returns {boolean} True if audio is playing or chunks are queued
   */
  const isPlaying = useCallback(() => {
    // Consider "playing" if audio is actively playing OR if chunks are queued
    // This prevents premature turn switching during gaps between chunks
    return isPlayingRef.current || audioQueueRef.current.length > 0;
  }, []);

  /**
   * Get the number of chunks in the queue
   * @returns {number} Queue length
   */
  const getQueueLength = useCallback(() => {
    return audioQueueRef.current.length;
  }, []);

  /**
   * Signal that streaming has started (chunks will arrive incrementally)
   * Call this when SSE streaming begins to prevent premature completion
   */
  const startStreaming = useCallback(() => {
    isStreamingRef.current = true;
    console.log('📡 Audio streaming started - completion blocked until endStreaming()');
  }, []);

  /**
   * Signal that streaming has ended (no more chunks will arrive)
   * Call this when SSE done event fires
   */
  const endStreaming = useCallback(() => {
    isStreamingRef.current = false;
    console.log('📡 Audio streaming ended - completion allowed when queue empties');

    // Check if queue is already empty and trigger completion if so
    if (!isPlayingRef.current && audioQueueRef.current.length === 0) {
      if (onPlaybackCompleteRef.current) {
        console.log('✅ Streaming ended with empty queue - triggering completion');
        onPlaybackCompleteRef.current();
      }
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Only cleanup if playback wasn't committed OR if it finished (queue empty AND not playing)
      // This prevents StrictMode from interrupting active audio playback
      if (!isCommittedRef.current || (audioQueueRef.current.length === 0 && !isPlayingRef.current)) {
        console.log('🧹 Cleaning up audio playback (not committed or finished playing)');
        stopPlayback();
      } else {
        console.log('🔒 Audio playback committed and still active - skipping cleanup to prevent interruption');
      }
    };
  }, [stopPlayback]);

  return {
    addAudioChunk,
    stopPlayback,
    reset,
    isPlaying,
    getQueueLength,
    startStreaming,
    endStreaming,
  };
};

export default useAudioPlayback;

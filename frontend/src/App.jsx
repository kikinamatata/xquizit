import { useState } from 'react';
import UploadScreen from './components/UploadScreen';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  const [interviewData, setInterviewData] = useState(null);

  const handleInterviewStart = (data) => {
    setInterviewData(data);
  };

  return (
    <div className="app">
      {!interviewData ? (
        <UploadScreen onInterviewStart={handleInterviewStart} />
      ) : (
        <ChatInterface
          sessionId={interviewData.sessionId}
          firstQuestion={interviewData.firstQuestion}
          firstQuestionAudio={interviewData.audioChunks}
          startTime={interviewData.startTime}
        />
      )}
    </div>
  );
}

export default App;

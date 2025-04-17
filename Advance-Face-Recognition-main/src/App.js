// App.js
import React, { useState, useEffect, useRef } from 'react';
import * as faceapi from 'face-api.js';
import './App.css';

function App() {
  // State variables
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [captureVideo, setCaptureVideo] = useState(false);
  const [registeredFaces, setRegisteredFaces] = useState([]);
  const [personName, setPersonName] = useState("");
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Load Models to Start");
  const [statusType, setStatusType] = useState("info"); // info, success, error, warning

  // Refs
  const videoRef = useRef();
  const videoHeight = 480;
  const videoWidth = 640;
  const canvasRef = useRef();
  const recognitionInterval = useRef(null);

  // Load face-api models
  const loadModels = async () => {
    setStatusMessage("Loading models... Please wait");
    setStatusType("info");
    
    const MODEL_URL = process.env.PUBLIC_URL + '/models';
    
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      ]);
      
      setModelsLoaded(true);
      setStatusMessage("Models loaded successfully! You can start the camera now.");
      setStatusType("success");
    } catch (error) {
      console.error(error);
      setStatusMessage("Error loading models. Please refresh and try again.");
      setStatusType("error");
    }
  };

  // Start/stop video capture
  const startVideo = () => {
    setCaptureVideo(true);
    setStatusMessage("Camera starting...");
    setStatusType("info");
    
    navigator.mediaDevices
      .getUserMedia({ video: { width: videoWidth, height: videoHeight } })
      .then(stream => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
        setStatusMessage("Camera is active. You can register faces or start recognition.");
        setStatusType("success");
      })
      .catch(err => {
        console.error("Error accessing camera:", err);
        setStatusMessage("Error accessing camera. Please check permissions.");
        setStatusType("error");
        setCaptureVideo(false);
      });
  };

  const stopVideo = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      setCaptureVideo(false);
      setStatusMessage("Camera stopped");
      setStatusType("info");
    }
  };

  // Register a face
  const registerFace = async () => {
    if (!personName.trim()) {
      setStatusMessage("Please enter a name before registering a face");
      setStatusType("warning");
      return;
    }

    if (!captureVideo) {
      setStatusMessage("Please start the camera first");
      setStatusType("warning");
      return;
    }

    setStatusMessage("Registering face... Please look at the camera");
    setStatusType("info");

    try {
      // Get face description
      const detections = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detections) {
        setStatusMessage("No face detected. Please make sure your face is visible");
        setStatusType("error");
        return;
      }

      // Create new face entry
      const newFace = {
        name: personName,
        descriptor: Array.from(detections.descriptor),
        timestamp: new Date().toISOString()
      };

      // Add to registered faces
      setRegisteredFaces(prev => [...prev, newFace]);
      setPersonName("");
      setStatusMessage(`Successfully registered ${personName}`);
      setStatusType("success");

    } catch (error) {
      console.error("Error registering face:", error);
      setStatusMessage("Error registering face. Please try again");
      setStatusType("error");
    }
  };

  // Start face recognition
  const startFaceRecognition = () => {
    if (registeredFaces.length === 0) {
      setStatusMessage("Please register at least one face before starting recognition");
      setStatusType("warning");
      return;
    }

    if (!captureVideo) {
      setStatusMessage("Please start the camera first");
      setStatusType("warning");
      return;
    }

    setIsRecognizing(true);
    setStatusMessage("Face recognition is active");
    setStatusType("success");

    // Create face matcher from registered faces
    const labeledDescriptors = registeredFaces.map(
      face => new faceapi.LabeledFaceDescriptors(
        face.name, 
        [new Float32Array(face.descriptor)]
      )
    );

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

    // Start recognition interval
    recognitionInterval.current = setInterval(async () => {
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const displaySize = { width: videoWidth, height: videoHeight };
        
        // Match canvas size to video
        faceapi.matchDimensions(canvas, displaySize);
        
        // Detect faces
        const detections = await faceapi
          .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceExpressions()
          .withFaceDescriptors();
        
        // Resize detections to match canvas size
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        
        // Clear the canvas
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw face detections
        resizedDetections.forEach(detection => {
          const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
          
          // Draw box
          const box = detection.detection.box;
          const drawBox = new faceapi.draw.DrawBox(box, { 
            label: `${bestMatch.label} (${Math.round(bestMatch.distance * 100) / 100})`,
            boxColor: bestMatch.label !== 'unknown' ? 'green' : 'red',
            lineWidth: 2
          });
          
          drawBox.draw(canvas);
          
          // Draw face landmarks
          faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
          
          // Draw face expressions
          faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
        });
      }
    }, 100);
  };

  // Stop face recognition
  const stopFaceRecognition = () => {
    if (recognitionInterval.current) {
      clearInterval(recognitionInterval.current);
      recognitionInterval.current = null;
      
      // Clear the canvas
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      }
      
      setIsRecognizing(false);
      setStatusMessage("Face recognition stopped");
      setStatusType("info");
    }
  };

  // Reset everything
  const resetSystem = () => {
    stopFaceRecognition();
    stopVideo();
    setRegisteredFaces([]);
    setPersonName("");
    setStatusMessage("System reset. You can start again");
    setStatusType("info");
  };

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopFaceRecognition();
      stopVideo();
    };
  }, []);

  return (
    <div className="app-container">
      <div className="header">
        <h1>🚀 Advanced Face Recognition System</h1>
        <p className={`status-message ${statusType}`}>{statusMessage}</p>
      </div>

      <div className="main-content">
        <div className="video-container">
          <div className="video-wrapper">
            {captureVideo && (
              <>
                <video 
                  ref={videoRef} 
                  height={videoHeight} 
                  width={videoWidth} 
                  autoPlay 
                  playsInline 
                  muted
                  style={{ transform: 'scaleX(-1)' }} // Add this line to remove mirror effect
                ></video>
                <canvas ref={canvasRef} className="face-canvas" />
              </>
            )}
            {!captureVideo && (
              <div className="no-video">
                <div className="camera-icon">📷</div>
                <p>Camera not active</p>
              </div>
            )}
          </div>
        </div>

        <div className="controls-panel">
          <div className="control-section">
            <h2>Setup</h2>
            <div className="button-group">
              <button 
                onClick={loadModels} 
                disabled={modelsLoaded}
                className={modelsLoaded ? "btn disabled" : "btn primary"}
              >
                {modelsLoaded ? "Models Loaded ✓" : "1. Load Models"}
              </button>

              <button 
                onClick={captureVideo ? stopVideo : startVideo} 
                disabled={!modelsLoaded}
                className={captureVideo ? "btn danger" : "btn primary"}
              >
                {captureVideo ? "2. Stop Camera" : "2. Start Camera"}
              </button>
            </div>
          </div>

          <div className="control-section">
            <h2>Face Registration</h2>
            <div className="input-group">
              <input
                type="text"
                placeholder="Enter person's name"
                value={personName}
                onChange={(e) => setPersonName(e.target.value)}
                disabled={!captureVideo}
              />
              <button 
                onClick={registerFace} 
                disabled={!captureVideo || !personName.trim()}
                className="btn success"
              >
                3. Register Face
              </button>
            </div>
          </div>

          <div className="control-section">
            <h2>Recognition Control</h2>
            <div className="button-group">
              <button 
                onClick={isRecognizing ? stopFaceRecognition : startFaceRecognition} 
                disabled={!captureVideo}
                className={isRecognizing ? "btn warning" : "btn primary"}
              >
                {isRecognizing ? "4. Stop Recognition" : "4. Start Recognition"}
              </button>

              <button 
                onClick={resetSystem} 
                className="btn danger"
              >
                Reset Everything
              </button>
            </div>
          </div>

          <div className="registered-faces">
            <h2>Registered People ({registeredFaces.length})</h2>
            {registeredFaces.length === 0 ? (
              <p className="no-faces">No faces registered yet</p>
            ) : (
              <ul className="face-list">
                {registeredFaces.map((face, index) => (
                  <li key={index} className="face-item">
                    <div className="face-avatar">{face.name.charAt(0).toUpperCase()}</div>
                    <div className="face-info">
                      <span className="face-name">{face.name}</span>
                      <span className="face-time">
                        {new Date(face.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="footer">
        <p>Advanced Face Recognition System | Created by Ankur Jha, Rohit Singh, Kumar Anubhav, Ankur Pathak</p>
      </div>
    </div>
  );
}

export default App;
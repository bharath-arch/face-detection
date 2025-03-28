import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const imageRef = useRef(null);

  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [captureMode, setCaptureMode] = useState({
    detection: false,
    landmarks: false,
    expressions: false,
    ageGender: false,
    FaceMatch: false
  });
  const [uploadedImage, setUploadedImage] = useState(null);
  const [recognitionResults, setRecognitionResults] = useState(null);

  const loadModels = () => {
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
      faceapi.nets.faceExpressionNet.loadFromUri('/models'),
      faceapi.nets.ageGenderNet.loadFromUri('/models')
    ]).then(() => {
      console.log("Models loaded successfully");
      setModelsLoaded(true);
    }).catch((err) => {
      console.error("Error loading models:", err);
    });
  };

  const startWebcam = () => {
    navigator.mediaDevices.getUserMedia({
      video: { width: 940, height: 720 }
    }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    }).catch((err) => {
      console.error("Error accessing webcam:", err);
    });
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          setUploadedImage(img);
          // Reset recognition results when a new image is uploaded
          setRecognitionResults(null);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const toggleFeature = (feature) => {
    setCaptureMode(prev => ({
      ...prev,
      [feature]: !prev[feature]
    }));
  };

  const compareDescriptors = async (imageDescriptor, videoDescriptors) => {
    const threshold = 0.6; // Adjust this value to control matching sensitivity
    return videoDescriptors.map((descriptor, index) => {
      const distance = faceapi.euclideanDistance(imageDescriptor, descriptor);
      return {
        index,
        distance,
        isMatch: distance <= threshold
      };
    });
  };

  const detectFaces = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video || !modelsLoaded) return;

    canvas.width = video.videoWidth || 940;
    canvas.height = video.videoHeight || 720;
    
    const displaySize = { width: canvas.width, height: canvas.height };
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const detectAndDraw = async () => {
      try {
        // Detect faces in the video
        const detections = await faceapi
          .detectAllFaces(
            video, 
            new faceapi.TinyFaceDetectorOptions({ 
              inputSize: 416, 
              scoreThreshold: 0.5 
            })
          )
          .withFaceLandmarks()
          .withFaceDescriptors()
          .withFaceExpressions()
          .withAgeAndGender();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Conditional drawing based on selected modes
        if (captureMode.detection) {
          faceapi.draw.drawDetections(canvas, resizedDetections);
        }

        if (captureMode.landmarks) {
          faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        }

        if (captureMode.expressions) {
          faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
        }

        if (captureMode.ageGender) {
          resizedDetections.forEach(detection => {
            const box = detection.detection.box;
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText(
              `Age: ${Math.round(detection.age)} | ${detection.gender}`, 
              box.x, 
              box.y - 10
            );
          });
        }

        // Face recognition logic
        if (captureMode.FaceMatch && uploadedImage) {
          // Detect face in the uploaded image
          const uploadedImageDetections = await faceapi
            .detectAllFaces(
              uploadedImage, 
              new faceapi.TinyFaceDetectorOptions({ 
                inputSize: 416, 
                scoreThreshold: 0.5 
              })
            )
            .withFaceLandmarks()
            .withFaceDescriptors();

          if (uploadedImageDetections.length > 0 && detections.length > 0) {
            // Get the first face descriptors
            const uploadedImageDescriptor = uploadedImageDetections[0].descriptor;
            const videoDescriptors = detections.map(det => det.descriptor);

            // Compare descriptors
            const matchResults = await compareDescriptors(
              uploadedImageDescriptor, 
              videoDescriptors
            );

            setRecognitionResults(matchResults);

            // Highlight matching faces
            matchResults.forEach(result => {
              if (result.isMatch) {
                const matchedDetection = resizedDetections[result.index];
                const box = matchedDetection.detection.box;
                
                // Draw a green border for matched faces
                ctx.strokeStyle = 'green';
                ctx.lineWidth = 4;
                ctx.strokeRect(box.x, box.y, box.width, box.height);
              }
            });
          }
        }
      } catch (err) {
        console.error("Face detection error:", err);
      }
    };

    detectAndDraw();
  };

  useEffect(() => {
    loadModels();
    startWebcam();
  }, []);

  useEffect(() => {
    if (Object.values(captureMode).some(mode => mode)) {
      const intervalId = setInterval(detectFaces, 500);
      return () => clearInterval(intervalId);
    }
  }, [captureMode, modelsLoaded, uploadedImage]);

  return (
    <>
      <div className="bg-gray-900 min-h-screen text-white p-6">
        <div className="container mx-auto max-w-6xl">
          <h1 className="text-3xl font-bold text-center mb-6 text-indigo-400">
            Face Detection & Recognition App
          </h1>

          {/* Image Upload */}
          <div className="flex justify-center mb-6">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              className="text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 
                       file:rounded-full file:border-0 
                       file:text-sm file:font-semibold
                       file:bg-indigo-500 file:text-white
                       hover:file:bg-indigo-600"
            />
          </div>

          {/* Feature Toggle Buttons */}
          <div className="flex flex-wrap justify-center gap-4 mb-6">
            {Object.keys(captureMode).map((feature) => (
              <button
                key={feature}
                onClick={() => toggleFeature(feature)}
                className={`px-4 py-2 rounded-lg transition-colors duration-300 
                          ${
                            captureMode[feature]
                              ? "bg-green-600 hover:bg-green-700"
                              : "bg-gray-700 hover:bg-gray-600"
                          }`}
              >
                {captureMode[feature] ? "Disable" : "Show"}{" "}
                {feature.charAt(0).toUpperCase() + feature.slice(1)}
              </button>
            ))}
          </div>

          {/* Recognition Results */}
          {recognitionResults && (
            <div className="bg-gray-800 p-4 rounded-lg mb-6 text-center">
              <strong className="text-lg text-indigo-400 block mb-2">
                Recognition Results
              </strong>
              {recognitionResults.map((result, index) => (
                <div
                  key={index}
                  className={`p-2 rounded mb-2 ${
                    result.isMatch
                      ? "bg-green-900 text-green-300"
                      : "bg-red-900 text-red-300"
                  }`}
                >
                  Face {index + 1}:{" "}
                  {result.isMatch ? "Match Found" : "No Match"}
                  {result.isMatch &&
                    ` (Distance: ${result.distance.toFixed(4)})`}
                </div>
              ))}
            </div>
          )}

          {/* Media Display */}
          <div className="flex justify-center items-center space-x-6">
            <div className="relative">
              <video
                ref={videoRef}
                width="940"
                height="720"
                autoPlay
                muted
                playsInline
                className="rounded-lg shadow-xl border-4 border-gray-700"
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 z-10 pointer-events-none"
              />
            </div>

            {uploadedImage && (
              <div className="text-center">
                <strong className="block mb-2 text-indigo-400">
                  Reference Image
                </strong>
                <div className="relative inline-block">
                  <img
                    ref={imageRef}
                    src={uploadedImage.src}
                    alt="Reference"
                    className="max-w-xs max-h-72 rounded-lg shadow-xl border-4 border-gray-700 object-contain"
                  />
                  <button
                    onClick={() => {
                      setUploadedImage(null);
                      setRecognitionResults(null);
                      fileInputRef.current.value = null;
                    }}
                    className="absolute top-2 right-2 bg-red-600 text-white p-2 rounded-full 
                             hover:bg-red-700 transition-colors"
                  >
                    ✕
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default App;

 // <div className="container" style={{ 
    //   position: 'relative', 
    //   width: "100dvw", 
    //   height: 900,
    //   textAlign: 'center',
    //   display: 'flex',
    //   flexDirection: 'column',
    //   alignItems: 'center',
    //   justifyContent: 'center',
    // }}>
    //   {/* Image Upload */}
    //   <div style={{ 
    //     display: 'flex', 
    //     justifyContent: 'center', 
    //     marginBottom: '20px' 
    //   }}>
    //     <input 
    //       type="file" 
    //       ref={fileInputRef}
    //       onChange={handleImageUpload}
    //       accept="image/*"
    //     />
    //   </div>

    //   {/* Feature Toggle Buttons */}
    //   <div style={{ 
    //     display: 'flex', 
    //     justifyContent: 'center', 
    //     marginBottom: '20px' 
    //   }}>
    //     <button 
    //       onClick={() => toggleFeature('detection')}
    //       style={{ 
    //         margin: '0 10px', 
    //         backgroundColor: captureMode.detection ? 'green' : 'gray',
    //         color: 'white',
    //         padding: '10px',
    //         border: 'none',
    //         borderRadius: '5px'
    //       }}
    //     >
    //       {captureMode.detection ? 'Disable' : 'Show'} Face Detection
    //     </button>

    //     <button 
    //       onClick={() => toggleFeature('landmarks')}
    //       style={{ 
    //         margin: '0 10px', 
    //         backgroundColor: captureMode.landmarks ? 'green' : 'gray',
    //         color: 'white',
    //         padding: '10px',
    //         border: 'none',
    //         borderRadius: '5px'
    //       }}
    //     >
    //       {captureMode.landmarks ? 'Disable' : 'Show'} Facial Landmarks
    //     </button>

    //     <button 
    //       onClick={() => toggleFeature('expressions')}
    //       style={{ 
    //         margin: '0 10px', 
    //         backgroundColor: captureMode.expressions ? 'green' : 'gray',
    //         color: 'white',
    //         padding: '10px',
    //         border: 'none',
    //         borderRadius: '5px'
    //       }}
    //     >
    //       {captureMode.expressions ? 'Disable' : 'Show'} Face Expressions
    //     </button>

    //     <button 
    //       onClick={() => toggleFeature('ageGender')}
    //       style={{ 
    //         margin: '0 10px', 
    //         backgroundColor: captureMode.ageGender ? 'green' : 'gray',
    //         color: 'white',
    //         padding: '10px',
    //         border: 'none',
    //         borderRadius: '5px'
    //       }}
    //     >
    //       {captureMode.ageGender ? 'Disable' : 'Show'} Age & Gender
    //     </button>

    //     <button 
    //       onClick={() => toggleFeature('recognition')}
    //       style={{ 
    //         margin: '0 10px', 
    //         backgroundColor: captureMode.recognition ? 'green' : 'gray',
    //         color: 'white',
    //         padding: '10px',
    //         border: 'none',
    //         borderRadius: '5px'
    //       }}
    //     >
    //       {captureMode.recognition ? 'Disable' : 'Show'} Face Recognition
    //     </button>
    //   </div>

    //   {/* Recognition Results */}
    //   {recognitionResults && (
    //     <div style={{ 
    //       marginBottom: '20px', 
    //       padding: '10px', 
    //       // backgroundColor: '#f0f0f0',
    //       borderRadius: '5px'
    //     }}>
    //       <strong>Recognition Results:</strong>
    //       {recognitionResults.map((result, index) => (
    //         <div key={index}>
    //           Face {index + 1}: {result.isMatch ? 'Match Found' : 'No Match'}
    //           {result.isMatch && ` (Distance: ${result.distance.toFixed(4)})`}
    //         </div>
    //       ))}
    //     </div>
    //   )}

    //   {/* Media Display */}
    //   <div style={{ display: 'flex', alignItems: 'center' }}>
    //     <video 
    //       ref={videoRef} 
    //       width="940" 
    //       height="720" 
    //       autoPlay 
    //       muted 
    //       playsInline
    //     />

    //     {uploadedImage && (
    //       <>
          
    //       <div style={{ marginLeft: '20px' }}>
    //         <strong>Reference Image:</strong>
    //         <img 
    //           ref={imageRef}
    //           src={uploadedImage.src}
    //           alt="Reference"
    //           style={{ 
    //             maxWidth: '300px',
    //             maxHeight: '300px',
    //             objectFit: 'contain',
    //             border: '2px solid #ddd'
    //           }}
    //         />
    //       </div>
    //       <button onClick={() => {setUploadedImage(null)}}>
    //         delete
    //       </button>
    //       </>
    //     )}
    //   </div>

    //   <canvas 
    //     ref={canvasRef} 
    //     style={{ 
    //       position: 'absolute', 
    //       top: 150, 
    //       left: 400, 
    //       zIndex: 10 
    //     }}
    //   />
    // </div>
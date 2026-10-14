import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  FlipHorizontal,
  Eye,
  EyeOff,
  Video,
  AlertCircle,
  Activity,
  Layers,
  Sparkles,
  Maximize2,
  ZoomIn,
  Split,
  ShieldAlert,
} from 'lucide-react';
import {
  LandmarksStatus,
  PerformanceMetrics,
  LandmarkPoint,
  BoundingBox,
  RecognitionStateCode,
  QualityCoachMetrics,
  CameraViewMode,
} from '../types';

interface CameraPanelProps {
  onFrameCaptured: (imageDataUrl: string, timestampMs: number) => void;
  isStreaming: boolean;
  setIsStreaming: (streaming: boolean) => void;
  landmarksStatus: LandmarksStatus | null;
  leftLandmarks: LandmarkPoint[];
  rightLandmarks: LandmarkPoint[];
  leftBox: BoundingBox | null;
  rightBox: BoundingBox | null;
  quality: QualityCoachMetrics | null;
  stateCode: RecognitionStateCode;
  candidateWord: string | null;
  performance: PerformanceMetrics | null;
  targetFps: number;
  mirrorCamera: boolean;
  onToggleMirror: () => void;
}

const HAND_CONNECTIONS = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  // Index
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  // Middle
  [9, 10],
  [10, 11],
  [11, 12],
  // Ring
  [13, 14],
  [14, 15],
  [15, 16],
  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  // Palm knuckles
  [5, 9],
  [9, 13],
  [13, 17],
];

export const CameraPanel: React.FC<CameraPanelProps> = ({
  onFrameCaptured,
  isStreaming,
  setIsStreaming,
  landmarksStatus,
  leftLandmarks,
  rightLandmarks,
  leftBox,
  rightBox,
  quality,
  stateCode,
  candidateWord,
  performance,
  targetFps,
  mirrorCamera,
  onToggleMirror,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const focusCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<CameraViewMode>('full');

  // Smoothed bounding box for Hand Focus inset
  const smoothBoxRef = useRef<BoundingBox | null>(null);

  // Enumerate video devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devs.filter((d) => d.kind === 'videoinput');
        setDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId);
        }
      } catch (err) {
        console.warn('Could not enumerate video devices:', err);
      }
    };
    getDevices();
  }, [selectedDeviceId]);

  // Start Camera Stream
  const startCamera = async () => {
    setPermissionError(null);
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
      setIsStreaming(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraActive(false);
      setIsStreaming(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermissionError('Camera access was denied. Please allow camera permissions in your browser address bar.');
      } else {
        setPermissionError(`Failed to access camera: ${err.message || 'Unknown error'}`);
      }
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsStreaming(false);
  };

  // In-flight transmission lock to prevent socket buffer backlog
  const inFlightRef = useRef<boolean>(false);

  // Release in-flight lock whenever server delivers frame results
  useEffect(() => {
    inFlightRef.current = false;
  }, [leftLandmarks, rightLandmarks, landmarksStatus]);

  // Frame Capture Loop (Optimized for real-time responsiveness)
  useEffect(() => {
    if (!cameraActive || !isStreaming) {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const intervalMs = Math.round(1000 / Math.max(10, Math.min(25, targetFps)));

    intervalRef.current = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      // Skip frame if previous frame is still being processed by backend
      if (inFlightRef.current) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      // High-speed 320x240 frame (preserves all hand landmarks with 3x faster encoding)
      const targetW = 320;
      const targetH = Math.round((height / width) * 320);

      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, targetW, targetH);
      inFlightRef.current = true;
      const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
      onFrameCaptured(dataUrl, Date.now());

      // Safety timeout to reset in-flight flag if network delays
      window.setTimeout(() => {
        inFlightRef.current = false;
      }, 120);
    }, intervalMs);

    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [cameraActive, isStreaming, targetFps, onFrameCaptured]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Draw 21-point Hand Landmarks Skeleton Overlay
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !cameraActive || !showOverlay) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const mapX = (x: number) => (mirrorCamera ? (1.0 - x) * w : x * w);
    const mapY = (y: number) => y * h;

    // Helper to draw single hand skeleton
    const renderHandSkeleton = (
      points: LandmarkPoint[],
      box: BoundingBox | null,
      label: string,
      color: string,
      glow: string
    ) => {
      if (!points || points.length < 21) return;
      const hasValid = points.some((p) => p.valid);
      if (!hasValid) return;

      // Draw bones
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.shadowColor = glow;
      ctx.shadowBlur = 6;

      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = points[startIdx];
        const p2 = points[endIdx];
        if (p1?.valid && p2?.valid) {
          ctx.beginPath();
          ctx.moveTo(mapX(p1.x), mapY(p1.y));
          ctx.lineTo(mapX(p2.x), mapY(p2.y));
          ctx.stroke();
        }
      }

      // Draw 21 joint nodes
      ctx.shadowBlur = 0;
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (!p.valid) {
          // Visually distinguish missing / unreliable joints
          ctx.strokeStyle = '#f43f5e';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(mapX(p.x), mapY(p.y), 3, 0, 2 * Math.PI);
          ctx.stroke();
          continue;
        }

        const cx = mapX(p.x);
        const cy = mapY(p.y);

        // Fingertip highlights (indices 4, 8, 12, 16, 20)
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const isWrist = i === 0;

        ctx.fillStyle = isTip ? '#ffffff' : color;
        ctx.beginPath();
        ctx.arc(cx, cy, isWrist ? 5.5 : isTip ? 4.5 : 3.5, 0, 2 * Math.PI);
        ctx.fill();

        if (isTip) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // Draw bounding box if present
      if (box) {
        const bx = mirrorCamera ? (1.0 - box.xmax) * w : box.xmin * w;
        const by = box.ymin * h;
        const bw = box.width * w;
        const bh = box.height * h;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);

        // Handedness label tag
        ctx.fillStyle = color;
        ctx.fillRect(bx, Math.max(0, by - 20), 80, 20);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(label, bx + 6, Math.max(14, by - 6));
      }
    };

    // Draw Right Hand (Teal) and Left Hand (Indigo)
    renderHandSkeleton(rightLandmarks, rightBox, 'Right Hand', '#14b8a6', '#0d9488');
    renderHandSkeleton(leftLandmarks, leftBox, 'Left Hand', '#6366f1', '#4f46e5');
  }, [cameraActive, showOverlay, leftLandmarks, rightLandmarks, leftBox, rightBox, mirrorCamera]);

  // Hand Focus Inset Viewport Render
  useEffect(() => {
    if (viewMode === 'full' || !cameraActive) return;

    const canvas = focusCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.readyState < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const targetBox = rightBox || leftBox;
    if (!targetBox) {
      // Clear empty state
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No hand detected in focus frame', canvas.width / 2, canvas.height / 2);
      return;
    }

    // Smooth bounding box coordinates using EMA filter
    if (!smoothBoxRef.current) {
      smoothBoxRef.current = { ...targetBox };
    } else {
      const alpha = 0.25;
      smoothBoxRef.current = {
        xmin: smoothBoxRef.current.xmin * (1 - alpha) + targetBox.xmin * alpha,
        ymin: smoothBoxRef.current.ymin * (1 - alpha) + targetBox.ymin * alpha,
        xmax: smoothBoxRef.current.xmax * (1 - alpha) + targetBox.xmax * alpha,
        ymax: smoothBoxRef.current.ymax * (1 - alpha) + targetBox.ymax * alpha,
        width: smoothBoxRef.current.width * (1 - alpha) + targetBox.width * alpha,
        height: smoothBoxRef.current.height * (1 - alpha) + targetBox.height * alpha,
      };
    }

    const sb = smoothBoxRef.current;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    const sx = Math.max(0, sb.xmin * vw);
    const sy = Math.max(0, sb.ymin * vh);
    const sw = Math.min(vw - sx, sb.width * vw);
    const sh = Math.min(vh - sy, sb.height * vh);

    const cw = canvas.width;
    const ch = canvas.height;

    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    if (mirrorCamera) {
      ctx.translate(cw, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch);
    ctx.restore();

    // Draw hand skeleton on close-up crop
    const pts = rightBox ? rightLandmarks : leftLandmarks;
    if (pts && pts.length >= 21) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#14b8a6';
      for (const [s, e] of HAND_CONNECTIONS) {
        const p1 = pts[s];
        const p2 = pts[e];
        if (p1?.valid && p2?.valid) {
          const x1 = ((p1.x - sb.xmin) / sb.width) * cw;
          const y1 = ((p1.y - sb.ymin) / sb.height) * ch;
          const x2 = ((p2.x - sb.xmin) / sb.width) * cw;
          const y2 = ((p2.y - sb.ymin) / sb.height) * ch;

          const mx1 = mirrorCamera ? cw - x1 : x1;
          const mx2 = mirrorCamera ? cw - x2 : x2;

          ctx.beginPath();
          ctx.moveTo(mx1, y1);
          ctx.lineTo(mx2, y2);
          ctx.stroke();
        }
      }

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        if (!p.valid) continue;
        const x = ((p.x - sb.xmin) / sb.width) * cw;
        const y = ((p.y - sb.ymin) / sb.height) * ch;
        const mx = mirrorCamera ? cw - x : x;

        ctx.fillStyle = [4, 8, 12, 16, 20].includes(i) ? '#ffffff' : '#14b8a6';
        ctx.beginPath();
        ctx.arc(mx, y, i === 0 ? 6 : 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }, [viewMode, cameraActive, leftBox, rightBox, leftLandmarks, rightLandmarks, mirrorCamera]);

  // Distinct Recognition State Banner
  const renderStateBanner = () => {
    switch (stateCode) {
      case 'STATE_A_INACTIVE':
        return (
          <div className="bg-slate-900/80 backdrop-blur-xs text-slate-300 text-xs px-3 py-1.5 rounded-lg border border-slate-700/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span>Camera is off</span>
          </div>
        );
      case 'STATE_B_NO_HAND':
        return (
          <div className="bg-slate-900/80 backdrop-blur-xs text-amber-300 text-xs px-3 py-1.5 rounded-lg border border-amber-800/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            <span>No hand detected — place hand in center frame</span>
          </div>
        );
      case 'STATE_C_COLLECTING':
        return (
          <div className="bg-teal-950/80 backdrop-blur-xs text-teal-300 text-xs px-3 py-1.5 rounded-lg border border-teal-700/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
            <span>Hand detected — collecting sign motion...</span>
          </div>
        );
      case 'STATE_D_CANDIDATE':
        return (
          <div className="bg-indigo-950/80 backdrop-blur-xs text-indigo-300 text-xs px-3 py-1.5 rounded-lg border border-indigo-700/60 flex items-center space-x-2">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
            <span>
              Recognizing sign: <strong>{candidateWord?.toUpperCase()}</strong>
            </span>
          </div>
        );
      case 'STATE_E_COMMITTED':
        return (
          <div className="bg-emerald-950/80 backdrop-blur-xs text-emerald-300 text-xs px-3 py-1.5 rounded-lg border border-emerald-600/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Word recognized and committed to transcript!</span>
          </div>
        );
      case 'STATE_F_UNCERTAIN':
        return (
          <div className="bg-rose-950/80 backdrop-blur-xs text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-rose-800/60 flex items-center space-x-2">
            <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
            <span>Sign uncertain — improve hand visibility or try again</span>
          </div>
        );
      case 'STATE_G_MODEL_UNAVAILABLE':
        return (
          <div className="bg-rose-950/80 backdrop-blur-xs text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-rose-800/60 flex items-center space-x-2">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            <span>Recognition model unavailable (checkpoint missing)</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Panel Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-5 h-5 text-teal-600" />
          <h2 className="text-base font-semibold text-slate-800">Live Camera Workspace</h2>
        </div>

        {/* View Mode & Overlay Controls */}
        <div className="flex items-center space-x-1.5">
          <div className="bg-slate-100 p-0.5 rounded-lg flex items-center text-xs">
            <button
              onClick={() => setViewMode('full')}
              className={`px-2 py-1 rounded-md transition-colors ${
                viewMode === 'full' ? 'bg-white text-slate-800 shadow-xs font-semibold' : 'text-slate-600'
              }`}
              title="Full Camera View"
            >
              Full
            </button>
            <button
              onClick={() => setViewMode('focus')}
              className={`px-2 py-1 rounded-md transition-colors ${
                viewMode === 'focus' ? 'bg-white text-slate-800 shadow-xs font-semibold' : 'text-slate-600'
              }`}
              title="Hand Focus Zoom"
            >
              Focus
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-2 py-1 rounded-md transition-colors ${
                viewMode === 'split' ? 'bg-white text-slate-800 shadow-xs font-semibold' : 'text-slate-600'
              }`}
              title="Split View"
            >
              Split
            </button>
          </div>

          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`p-1.5 rounded-lg border transition-colors ${
              showOverlay ? 'bg-teal-50 border-teal-200 text-teal-700' : 'bg-white border-slate-200 text-slate-400'
            }`}
            title={showOverlay ? 'Hide 21-pt Skeleton' : 'Show 21-pt Skeleton'}
          >
            {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Viewport Grid */}
      <div
        className={`grid gap-3 ${
          viewMode === 'split' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
        }`}
      >
        {/* Primary Camera Viewport */}
        {viewMode !== 'focus' && (
          <div className="relative aspect-[4/3] bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center border border-slate-800 shadow-inner group">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${mirrorCamera ? '-scale-x-100' : ''}`}
            />
            <canvas ref={captureCanvasRef} className="hidden" />

            {/* Skeleton Overlay Canvas */}
            <canvas
              ref={overlayCanvasRef}
              width={640}
              height={480}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            {/* Camera Inactive State */}
            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/90 text-slate-400 p-6 text-center space-y-3">
                <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center text-slate-300 ring-4 ring-slate-800/60">
                  <Video className="w-7 h-7" />
                </div>
                <div>
                  <p className="font-semibold text-slate-200 text-sm">Camera is Inactive</p>
                  <p className="text-xs text-slate-400 mt-1 max-w-xs">
                    Click "Start Camera" to begin real-time sign detection, hand tracking, and recognition.
                  </p>
                </div>
                <button
                  onClick={startCamera}
                  className="mt-2 inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition-colors"
                >
                  <Camera className="w-4 h-4" />
                  <span>Start Camera</span>
                </button>
              </div>
            )}

            {/* In-Frame Status & Framing Guide */}
            {cameraActive && (
              <div className="absolute inset-0 pointer-events-none p-3 flex flex-col justify-between">
                <div className="flex justify-between items-start gap-2">
                  {renderStateBanner()}
                  {performance && (
                    <div className="bg-slate-900/75 backdrop-blur-xs text-slate-300 text-[11px] px-2.5 py-1 rounded-md border border-slate-700/60 font-mono">
                      {performance.fps} FPS | {performance.total_latency_ms}ms
                    </div>
                  )}
                </div>

                {/* Posture Framing Area Box */}
                <div className="self-center w-3/4 h-3/4 border-2 border-dashed border-teal-400/25 rounded-2xl flex items-center justify-center">
                  <span className="text-[10px] text-teal-300/40 font-mono uppercase tracking-widest">
                    Sign Framing Guide
                  </span>
                </div>

                {/* Bottom Modality Status Chips */}
                <div className="flex items-center space-x-2">
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                      landmarksStatus?.left_hand ? 'bg-indigo-500/80 text-white' : 'bg-slate-800/80 text-slate-400'
                    }`}
                  >
                    L-Hand: {landmarksStatus?.left_hand ? 'Tracking' : 'None'}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                      landmarksStatus?.right_hand ? 'bg-teal-500/80 text-white' : 'bg-slate-800/80 text-slate-400'
                    }`}
                  >
                    R-Hand: {landmarksStatus?.right_hand ? 'Tracking' : 'None'}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Hand Focus Close-Up Viewport */}
        {(viewMode === 'focus' || viewMode === 'split') && (
          <div className="relative aspect-[4/3] bg-slate-950 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 shadow-inner">
            <canvas ref={focusCanvasRef} width={480} height={360} className="w-full h-full object-contain" />
            <div className="absolute top-2 left-2 bg-slate-900/80 backdrop-blur-xs text-slate-300 text-[10px] px-2 py-1 rounded font-mono border border-slate-700/60 flex items-center space-x-1.5">
              <ZoomIn className="w-3.5 h-3.5 text-teal-400" />
              <span>Hand Focus Close-Up</span>
            </div>
          </div>
        )}
      </div>

      {/* Permission Error Banner */}
      {permissionError && (
        <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Camera Access Error</p>
            <p className="mt-0.5">{permissionError}</p>
          </div>
        </div>
      )}

      {/* Bottom Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-slate-100">
        <div className="flex items-center space-x-2">
          {!cameraActive ? (
            <button
              onClick={startCamera}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition-colors"
            >
              <Camera className="w-4 h-4" />
              <span>Start Camera</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold shadow-xs transition-colors"
            >
              <CameraOff className="w-4 h-4" />
              <span>Stop Camera</span>
            </button>
          )}

          {cameraActive && (
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={`inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl border text-xs font-semibold transition-colors ${
                isStreaming
                  ? 'bg-amber-50 border-amber-200 text-amber-700 hover:bg-amber-100'
                  : 'bg-teal-50 border-teal-200 text-teal-700 hover:bg-teal-100'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>{isStreaming ? 'Pause Transmission' : 'Resume'}</span>
            </button>
          )}
        </div>

        {/* Mirror and Device Selection */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onToggleMirror}
            className={`p-2 rounded-xl border text-xs font-semibold transition-colors flex items-center space-x-1.5 ${
              mirrorCamera ? 'bg-teal-50 border-teal-200 text-teal-700' : 'bg-white border-slate-200 text-slate-600'
            }`}
            title="Toggle Mirror Camera"
          >
            <FlipHorizontal className="w-4 h-4" />
            <span className="hidden sm:inline">Mirror</span>
          </button>

          {devices.length > 1 && (
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              disabled={cameraActive}
              className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1.5 text-slate-700 focus:outline-none focus:ring-1 focus:ring-teal-500"
            >
              {devices.map((d, idx) => (
                <option key={d.deviceId || idx} value={d.deviceId}>
                  {d.label || `Camera ${idx + 1}`}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>
    </div>
  );
};

import React, { useEffect, useRef, useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipBack,
  SkipForward,
  X,
  Clock,
  Activity,
  Layers,
  Sparkles,
  Info,
} from 'lucide-react';
import { ReplayFrame, LandmarkPoint } from '../types';

interface GestureReplayProps {
  frames: ReplayFrame[];
  onClose: () => void;
  candidateWord?: string | null;
  confidence?: number;
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

export const GestureReplay: React.FC<GestureReplayProps> = ({
  frames,
  onClose,
  candidateWord,
  confidence = 0.0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0); // 0.5x or 1.0x

  const totalFrames = frames.length;

  // Playback timer
  useEffect(() => {
    if (!isPlaying || totalFrames === 0) return;

    const interval = window.setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= totalFrames - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, Math.round(50 / playbackSpeed));

    return () => clearInterval(interval);
  }, [isPlaying, totalFrames, playbackSpeed]);

  // Draw current frame skeleton on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || totalFrames === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#0f172a'; // slate-900
    ctx.fillRect(0, 0, width, height);

    // Grid guide lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    const currentFrame = frames[currentIndex];
    if (!currentFrame) return;

    const drawHand = (points: LandmarkPoint[], color: string, glow: string) => {
      if (!points || points.length < 21) return;
      const validPoints = points.filter((p) => p.valid);
      if (validPoints.length === 0) return;

      // Draw bones
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.shadowColor = glow;
      ctx.shadowBlur = 4;

      for (const [s, e] of HAND_CONNECTIONS) {
        const p1 = points[s];
        const p2 = points[e];
        if (p1?.valid && p2?.valid) {
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      }

      // Draw joints
      ctx.shadowBlur = 0;
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (!p.valid) continue;
        const x = p.x * width;
        const y = p.y * height;

        ctx.fillStyle = i === 4 || i === 8 || i === 12 || i === 16 || i === 20 ? '#ffffff' : color;
        ctx.beginPath();
        ctx.arc(x, y, i === 0 ? 5 : 3.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    };

    // Draw hands (Right hand in Teal, Left hand in Indigo)
    drawHand(currentFrame.right_landmarks, '#14b8a6', '#0d9488');
    drawHand(currentFrame.left_landmarks, '#6366f1', '#4f46e5');
  }, [currentIndex, frames, totalFrames]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Clock className="w-5 h-5 text-teal-600" />
          <h3 className="text-sm font-semibold text-slate-800">Gesture Replay & Visual Timeline</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
          title="Close Replay"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {totalFrames === 0 ? (
        <div className="py-8 text-center text-slate-400 text-xs">
          No buffered landmark frames available to replay yet. Perform a sign gesture to populate the
          buffer.
        </div>
      ) : (
        <>
          {/* Canvas Viewport */}
          <div className="relative aspect-[4/3] bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
            <canvas ref={canvasRef} width={480} height={360} className="w-full h-full object-contain" />

            {/* Frame metadata overlay */}
            <div className="absolute top-2 left-2 bg-slate-950/80 backdrop-blur-xs text-slate-300 text-[10px] px-2 py-1 rounded font-mono border border-slate-700/60">
              Frame {currentIndex + 1} / {totalFrames}
            </div>

            {candidateWord && (
              <div className="absolute top-2 right-2 bg-teal-950/80 backdrop-blur-xs text-teal-300 text-[11px] px-2.5 py-1 rounded font-semibold border border-teal-700/60 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                <span>Candidate: {candidateWord.toUpperCase()}</span>
                {confidence > 0 && <span className="font-mono text-teal-400">({Math.round(confidence * 100)}%)</span>}
              </div>
            )}
          </div>

          {/* Timeline Slider */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>Start (0ms)</span>
              <span>
                Frame {currentIndex + 1} of {totalFrames}
              </span>
              <span>End ({totalFrames * 33}ms)</span>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={currentIndex}
              onChange={(e) => {
                setCurrentIndex(Number(e.target.value));
                setIsPlaying(false);
              }}
              className="w-full accent-teal-600 cursor-pointer"
            />
          </div>

          {/* Transport Controls Bar */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentIndex(0)}
                className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors"
                title="Restart"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
                className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors"
                title="Step Backward"
              >
                <SkipBack className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold flex items-center space-x-1.5 transition-colors shadow-xs"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
              </button>
              <button
                onClick={() => setCurrentIndex((prev) => Math.min(totalFrames - 1, prev + 1))}
                className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors"
                title="Step Forward"
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>

            {/* Playback speed toggle */}
            <div className="flex items-center space-x-1 bg-slate-100 p-0.5 rounded-lg text-[11px] font-medium text-slate-600">
              <button
                onClick={() => setPlaybackSpeed(0.5)}
                className={`px-2 py-1 rounded-md transition-colors ${
                  playbackSpeed === 0.5 ? 'bg-white text-teal-700 shadow-xs font-semibold' : ''
                }`}
              >
                0.5x
              </button>
              <button
                onClick={() => setPlaybackSpeed(1.0)}
                className={`px-2 py-1 rounded-md transition-colors ${
                  playbackSpeed === 1.0 ? 'bg-white text-teal-700 shadow-xs font-semibold' : ''
                }`}
              >
                1.0x
              </button>
            </div>
          </div>

          {/* Research & Privacy Note */}
          <div className="flex items-start space-x-1.5 text-[11px] text-slate-400 bg-slate-50/50 p-2 rounded-lg border border-slate-100">
            <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
            <p>
              Replay utilizes recent in-memory landmark observations strictly on your device. Video frames
              are not recorded or uploaded, preserving privacy and adhering to BDSLW131 research ethics terms.
            </p>
          </div>
        </>
      )}
    </div>
  );
};

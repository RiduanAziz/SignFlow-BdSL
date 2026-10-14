import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { CameraPanel } from './components/CameraPanel';
import { LiveRecognitionPanel } from './components/LiveRecognitionPanel';
import { HandQualityCoach } from './components/HandQualityCoach';
import { PostureInspector } from './components/PostureInspector';
import { GestureReplay } from './components/GestureReplay';
import { RobustnessLab } from './components/RobustnessLab';
import { PracticeMode } from './components/PracticeMode';
import { TranscriptWorkspace } from './components/TranscriptWorkspace';
import { SpeechWorkspace } from './components/SpeechWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { HelpModal } from './components/HelpModal';
import {
  HealthStatus,
  ModelInfo,
  RecognitionMode,
  CommittedToken,
  LivePrediction,
  WordCandidate,
  LandmarksStatus,
  PerformanceMetrics,
  AppSettings,
  LandmarkPoint,
  BoundingBox,
  QualityCoachMetrics,
  RecognitionStateCode,
  ReplayFrame,
} from './types';
import {
  AlertCircle,
  RefreshCw,
  Radio,
  ExternalLink,
  ShieldCheck,
  Hand,
  Clock,
  FlaskConical,
  GraduationCap,
} from 'lucide-react';

const DEFAULT_SETTINGS: AppSettings = {
  confidenceThreshold: 0.28,
  debounceFrames: 2,
  pauseThresholdSec: 0.5,
  targetFps: 15,
  autoSpeak: false,
  ttsRate: 1.0,
  ttsPitch: 1.0,
  ttsVoice: '',
  mirrorCamera: true,
  showLandmarkGuide: true,
  wsUrl:
    typeof window !== 'undefined' && window.location.protocol === 'https:'
      ? `wss://${window.location.hostname}:8000/ws/live`
      : 'ws://127.0.0.1:8000/ws/live',
};

type AdvancedTab = 'coach_posture' | 'replay' | 'robustness' | 'practice';

export const App: React.FC = () => {
  // System & Model Status
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Recognition States
  const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>('guided');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [landmarksStatus, setLandmarksStatus] = useState<LandmarksStatus | null>(null);
  const [prediction, setPrediction] = useState<LivePrediction | null>(null);
  const [candidate, setCandidate] = useState<WordCandidate | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [lastProcessedFrameId, setLastProcessedFrameId] = useState<number>(0);

  // Advanced Visuals & Quality States
  const [leftLandmarks, setLeftLandmarks] = useState<LandmarkPoint[]>([]);
  const [rightLandmarks, setRightLandmarks] = useState<LandmarkPoint[]>([]);
  const [leftBox, setLeftBox] = useState<BoundingBox | null>(null);
  const [rightBox, setRightBox] = useState<BoundingBox | null>(null);
  const [quality, setQuality] = useState<QualityCoachMetrics | null>(null);
  const [stateCode, setStateCode] = useState<RecognitionStateCode>('STATE_A_INACTIVE');
  const [replayFrames, setReplayFrames] = useState<ReplayFrame[]>([]);
  const [activeTab, setActiveTab] = useState<AdvancedTab>('coach_posture');

  // Transcript Tokens
  const [tokens, setTokens] = useState<CommittedToken[]>(() => {
    try {
      const saved = localStorage.getItem('signflow_tokens');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [transcriptText, setTranscriptText] = useState<string>('');

  // Settings
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = localStorage.getItem('signflow_settings');
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isMountedRef = useRef<boolean>(true);
  const settingsRef = useRef<AppSettings>(settings);
  settingsRef.current = settings;

  // Persist tokens
  useEffect(() => {
    try {
      localStorage.setItem('signflow_tokens', JSON.stringify(tokens));
    } catch (e) {
      console.warn('Failed to save tokens to localStorage:', e);
    }
  }, [tokens]);

  // Persist settings
  useEffect(() => {
    try {
      localStorage.setItem('signflow_settings', JSON.stringify(settings));
    } catch (e) {
      console.warn('Failed to save settings:', e);
    }
  }, [settings]);

  // Send dynamic configuration update over open WebSocket without reconnecting
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          mode: recognitionMode,
          confidence_threshold: settings.confidenceThreshold,
          debounce_frames: settings.debounceFrames,
          pause_threshold_sec: settings.pauseThresholdSec,
        })
      );
    }
  }, [settings.confidenceThreshold, settings.debounceFrames, settings.pauseThresholdSec, recognitionMode]);

  // Fetch backend health and model info (environment aware)
  const fetchStatus = useCallback(async () => {
    try {
      const baseUrl =
        typeof window !== 'undefined' && window.location.origin && !window.location.origin.includes(':5173')
          ? window.location.origin
          : 'http://127.0.0.1:8000';
      const healthRes = await fetch(`${baseUrl}/health`);
      if (healthRes.ok) {
        const hData = await healthRes.json();
        setHealth(hData);
        setBackendError(null);
      } else {
        setHealth(null);
        setBackendError('Backend health check returned non-200');
      }

      const modelRes = await fetch(`${baseUrl}/model/info`);
      if (modelRes.ok) {
        const mData = await modelRes.json();
        setModelInfo(mData);
      }
    } catch (err: any) {
      setHealth(null);
      setBackendError(
        'Unable to reach FastAPI backend service. Please ensure the backend is running.'
      );
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Auto-speak newly committed token if enabled
  const speakWord = useCallback(
    (word: string) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
      const utterance = new SpeechSynthesisUtterance(word);
      utterance.rate = settingsRef.current.ttsRate;
      utterance.pitch = settingsRef.current.ttsPitch;
      if (settingsRef.current.ttsVoice) {
        const v = window.speechSynthesis.getVoices().find((x) => x.name === settingsRef.current.ttsVoice);
        if (v) utterance.voice = v;
      }
      window.speechSynthesis.speak(utterance);
    },
    []
  );

  // Connect to WebSocket (only dependent on wsUrl)
  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(settings.wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setBackendError(null);
        ws.send(
          JSON.stringify({
            type: 'config',
            mode: recognitionMode,
            confidence_threshold: settingsRef.current.confidenceThreshold,
            debounce_frames: settingsRef.current.debounceFrames,
            pause_threshold_sec: settingsRef.current.pauseThresholdSec,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          switch (msg.type) {
            case 'session_started':
              if (msg.model_ready) {
                fetchStatus();
              }
              break;

            case 'landmarks_status':
              setLandmarksStatus(msg);
              break;

            case 'hand_landmarks':
              if (msg.left_landmarks) setLeftLandmarks(msg.left_landmarks);
              if (msg.right_landmarks) setRightLandmarks(msg.right_landmarks);
              setLeftBox(msg.left_box || null);
              setRightBox(msg.right_box || null);
              if (msg.quality) setQuality(msg.quality);
              if (msg.state_code) setStateCode(msg.state_code);
              setLandmarksStatus({
                hands_detected: Boolean(msg.hands_detected),
                left_hand: Boolean(msg.left_hand),
                right_hand: Boolean(msg.right_hand),
                pose: Boolean(msg.pose),
                face: Boolean(msg.face),
                valid_points: Number(msg.valid_points || 0),
              });
              break;

            case 'frame_processed':
              if (typeof msg.frame_id === 'number') {
                setLastProcessedFrameId(msg.frame_id);
              }
              break;

            case 'live_prediction':
              setPrediction(msg);
              if (msg.state_code) setStateCode(msg.state_code);
              break;

            case 'word_candidate':
              setCandidate(msg);
              break;

            case 'word_committed':
              const newToken: CommittedToken = {
                id: msg.token_id || String(Date.now()),
                word: msg.word,
                bengali_word: msg.bengali_word || msg.word,
                english_gloss: msg.english_gloss || msg.word,
                confidence: msg.confidence || 0,
                timestamp: msg.timestamp || Date.now(),
                mode: msg.mode || 'guided',
              };
              setTokens((prev) => [...prev, newToken]);
              setStateCode('STATE_E_COMMITTED');
              if (settings.autoSpeak) {
                speakWord(msg.bengali_word || msg.word);
              }
              // Automatically fetch replay buffer for recently completed gesture
              ws.send(JSON.stringify({ type: 'request_replay' }));
              break;

            case 'replay_buffer':
              if (msg.frames) {
                setReplayFrames(msg.frames);
              }
              break;

            case 'performance':
              setPerformance(msg);
              break;

            case 'error':
              console.warn('Backend WebSocket error:', msg.message);
              break;

            default:
              break;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (isMountedRef.current) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (isMountedRef.current) {
              connectWebSocket();
            }
          }, 3000);
        }
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
        setWsConnected(false);
      };
    } catch (err) {
      console.error('WebSocket creation exception:', err);
      setWsConnected(false);
    }
  }, [settings.wsUrl, recognitionMode, fetchStatus, speakWord]);

  // Initial WS connection
  useEffect(() => {
    isMountedRef.current = true;
    connectWebSocket();
    return () => {
      isMountedRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWebSocket]);

  // Frame transmission
  const handleFrameCaptured = useCallback(
    (imageDataUrl: string, timestampMs: number, frameId?: number) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'frame',
            data: imageDataUrl,
            timestamp_ms: timestampMs,
            frame_id: frameId,
          })
        );
      }
    },
    []
  );

  // Manual actions
  const handleConfirmWord = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'confirm_word' }));
    }
  };

  const handleRejectCandidate = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reject_candidate' }));
    }
  };

  const handleModeChange = (newMode: RecognitionMode) => {
    setRecognitionMode(newMode);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'config', mode: newMode }));
    }
  };

  const handleRemoveToken = (id: string) => {
    setTokens((prev) => prev.filter((t) => t.id !== id));
  };

  const handleClearTranscript = () => {
    setTokens([]);
    setTranscriptText('');
  };

  const handleUndo = () => {
    setTokens((prev) => prev.slice(0, -1));
  };

  const handleNewSession = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
    setTokens([]);
    setTranscriptText('');
    setPrediction(null);
    setCandidate(null);
  };

  const requestReplayBuffer = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'request_replay' }));
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 selection:bg-teal-100 selection:text-teal-900">
      {/* Top Navigation Header */}
      <Header
        health={health}
        modelInfo={modelInfo}
        wsConnected={wsConnected}
        recognitionMode={recognitionMode}
        onModeChange={handleModeChange}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenHelp={() => setIsHelpOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Connection Error Banner */}
        {backendError && (
          <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between shadow-xs">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
              <div>
                <p className="font-bold text-sm">Backend Service Offline</p>
                <p className="mt-0.5 text-rose-700">{backendError}</p>
              </div>
            </div>
            <button
              onClick={fetchStatus}
              className="px-3 py-1.5 rounded-xl bg-white border border-rose-200 text-rose-700 font-semibold hover:bg-rose-50 transition-colors flex items-center space-x-1"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Checkpoint Missing Advisory */}
        {health && !health.trained_model_loaded && (
          <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start space-x-3 shadow-xs">
            <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-bold text-sm">Trained Model Checkpoint Not Detected</p>
              <p className="leading-relaxed">
                The backend is currently running without a trained model checkpoint. Expected{' '}
                <code>checkpoints/bdslw131/robust_holistic_fusion_best.pt</code>. In accordance with thesis
                integrity requirements, predictions are paused to avoid invented words or random
                fallbacks.
              </p>
            </div>
          </div>
        )}

        {/* Primary Workspace: Live Camera (Left) + Live Recognition (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Panel: Camera, 21-pt Skeleton, Hand Focus Inset */}
          <div className="lg:col-span-7">
            <CameraPanel
              onFrameCaptured={handleFrameCaptured}
              lastProcessedFrameId={lastProcessedFrameId}
              isStreaming={isStreaming}
              setIsStreaming={setIsStreaming}
              landmarksStatus={landmarksStatus}
              leftLandmarks={leftLandmarks}
              rightLandmarks={rightLandmarks}
              leftBox={leftBox}
              rightBox={rightBox}
              quality={quality}
              stateCode={stateCode}
              candidateWord={candidate?.word || prediction?.predicted_label || null}
              performance={performance}
              targetFps={settings.targetFps}
              mirrorCamera={settings.mirrorCamera}
              onToggleMirror={() =>
                setSettings((s) => ({ ...s, mirrorCamera: !s.mirrorCamera }))
              }
            />
          </div>

          {/* Right Panel: Recognition Card, Candidate Distribution, Confirm Actions */}
          <div className="lg:col-span-5">
            <LiveRecognitionPanel
              prediction={prediction}
              candidate={candidate}
              mode={recognitionMode}
              stateCode={stateCode}
              onConfirmWord={handleConfirmWord}
              onRejectCandidate={handleRejectCandidate}
              confidenceThreshold={settings.confidenceThreshold}
              lastCommittedWord={tokens.length > 0 ? tokens[tokens.length - 1].word : null}
            />
          </div>
        </div>

        {/* Advanced Features Tab Strip */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-slate-200 pb-2 overflow-x-auto">
            <button
              onClick={() => setActiveTab('coach_posture')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'coach_posture'
                  ? 'bg-teal-600 text-white shadow-xs'
                  : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>Hand Quality Coach & Posture</span>
            </button>

            <button
              onClick={() => {
                setActiveTab('replay');
                requestReplayBuffer();
              }}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'replay'
                  ? 'bg-teal-600 text-white shadow-xs'
                  : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              <Clock className="w-4 h-4" />
              <span>Gesture Replay Timeline</span>
            </button>

            <button
              onClick={() => setActiveTab('robustness')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'robustness'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              <FlaskConical className="w-4 h-4" />
              <span>Robustness Lab (Thesis)</span>
            </button>

            <button
              onClick={() => setActiveTab('practice')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'practice'
                  ? 'bg-teal-600 text-white shadow-xs'
                  : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              <GraduationCap className="w-4 h-4" />
              <span>Guided Practice Mode</span>
            </button>
          </div>

          {/* Active Tab Content Area */}
          <div>
            {activeTab === 'coach_posture' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <HandQualityCoach
                  quality={quality}
                  handsDetected={Boolean(landmarksStatus?.hands_detected)}
                  stateCode={stateCode}
                />
                <PostureInspector
                  leftLandmarks={leftLandmarks}
                  rightLandmarks={rightLandmarks}
                  currentWord={candidate?.word || prediction?.predicted_label || null}
                />
              </div>
            )}

            {activeTab === 'replay' && (
              <GestureReplay
                frames={replayFrames}
                onClose={() => setActiveTab('coach_posture')}
                candidateWord={candidate?.word || prediction?.predicted_label || null}
                confidence={candidate?.confidence ?? prediction?.confidence ?? 0}
              />
            )}

            {activeTab === 'robustness' && (
              <RobustnessLab
                validPoints={landmarksStatus?.valid_points ?? 0}
                totalPoints={553}
                stability={quality?.stability ?? 1.0}
                bufferCompleteness={quality?.buffer_completeness ?? 0.0}
                confidence={prediction?.confidence ?? 0}
                latencyMs={performance?.total_latency_ms ?? 0}
                recentFeatures={
                  replayFrames
                    .filter((f) => f.features && f.features.length > 0)
                    .map((f) => f.features as number[])
                }
              />
            )}

            {activeTab === 'practice' && (
              <PracticeMode
                currentPrediction={prediction?.predicted_label || null}
                confidence={prediction?.confidence ?? 0}
                qualityScore={quality?.score ?? 0}
              />
            )}
          </div>
        </div>

        {/* Full-Width Transcript Workspace */}
        <TranscriptWorkspace
          tokens={tokens}
          onRemoveToken={handleRemoveToken}
          onClearTranscript={handleClearTranscript}
          onUndo={handleUndo}
          transcriptText={transcriptText}
          setTranscriptText={setTranscriptText}
          onNewSession={handleNewSession}
        />

        {/* Full-Width Text-to-Speech (TTS) Workspace */}
        <SpeechWorkspace
          textToSpeak={transcriptText}
          autoSpeak={settings.autoSpeak}
          rate={settings.ttsRate}
          pitch={settings.ttsPitch}
          voiceName={settings.ttsVoice}
          onRateChange={(rate) => setSettings((s) => ({ ...s, ttsRate: rate }))}
          onPitchChange={(pitch) => setSettings((s) => ({ ...s, ttsPitch: pitch }))}
          onVoiceChange={(voice) => setSettings((s) => ({ ...s, ttsVoice: voice }))}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>
            <strong>SignFlow-BdSL</strong> — Robust Real-Time Bangla Sign Language Word Recognition (BDSLW131)
          </p>
          <div className="flex items-center space-x-3 text-slate-400">
            <span>BDSLW131 Benchmark (131 Classes)</span>
            <span>•</span>
            <span>Local Inference Only</span>
          </div>
        </div>
      </footer>

      {/* Modals */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={(newSettings) => setSettings((s) => ({ ...s, ...newSettings }))}
      />

      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
};

export default App;

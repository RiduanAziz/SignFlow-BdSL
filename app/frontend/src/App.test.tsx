import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { Header } from './components/Header';
import { LiveRecognitionPanel } from './components/LiveRecognitionPanel';
import { HandQualityCoach } from './components/HandQualityCoach';
import { PostureInspector } from './components/PostureInspector';
import { GestureReplay } from './components/GestureReplay';
import { RobustnessLab } from './components/RobustnessLab';
import { PracticeMode } from './components/PracticeMode';
import { TranscriptWorkspace } from './components/TranscriptWorkspace';
import { SpeechWorkspace } from './components/SpeechWorkspace';
import { SettingsModal } from './components/SettingsModal';
import {
  CommittedToken,
  LivePrediction,
  WordCandidate,
  AppSettings,
  LandmarkPoint,
  ReplayFrame,
} from './types';

describe('SignFlow Frontend Suite', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (window as any).SpeechSynthesisUtterance = class {
      text: string;
      rate: number = 1.0;
      pitch: number = 1.0;
      voice: any = null;
      constructor(text: string) {
        this.text = text;
      }
    };
    window.speechSynthesis = {
      speak: vi.fn(),
      cancel: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      getVoices: vi.fn().mockReturnValue([]),
      onvoiceschanged: null,
      paused: false,
      pending: false,
      speaking: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    } as any;
  });

  it('renders Header with service statuses and modes', () => {
    render(
      <Header
        health={{
          status: 'ok',
          process_alive: true,
          backend_initialized: true,
          mediapipe_available: true,
          trained_model_loaded: true,
          inference_ready: true,
          checkpoint_path: 'checkpoints/bdslw131/robust_holistic_fusion_best.pt',
          error: null,
          mediapipe_error: null,
        }}
        modelInfo={{
          name: 'robust_holistic_fusion',
          architecture: 'RobustHolisticFusionClassifier',
          status: 'ready',
          checkpoint_path: 'checkpoints/bdslw131/robust_holistic_fusion_best.pt',
          num_classes: 131,
          input_dim: 4977,
          sequence_length: 64,
          device: 'cpu',
          feature_representation: 'holistic',
          class_names_sample: ['অংশগ্রহণ', 'অনুরোধ'],
          total_classes: 131,
          error: null,
        }}
        wsConnected={true}
        recognitionMode="guided"
        onModeChange={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHelp={vi.fn()}
      />
    );

    expect(screen.getByText('SignFlow-BdSL')).toBeDefined();
    expect(screen.getByText('Mode A: Guided')).toBeDefined();
    expect(screen.getByText('Holistic 553pts')).toBeDefined();
    expect(screen.getByText('Live WS')).toBeDefined();
  });

  it('renders LiveRecognitionPanel with candidate and responds to confirm/reject actions', () => {
    const handleConfirm = vi.fn();
    const handleReject = vi.fn();

    const candidate: WordCandidate = {
      word: 'DRINK',
      confidence: 0.88,
      stability_count: 5,
      target_count: 5,
      progress: 1.0,
    };

    const prediction: LivePrediction = {
      predicted_label: 'DRINK',
      confidence: 0.88,
      meets_threshold: true,
      top_k: [
        { class_index: 1, label: 'DRINK', confidence: 0.88 },
        { class_index: 0, label: 'BOOK', confidence: 0.08 },
      ],
      latency_ms: 18.5,
      state: 'CANDIDATE',
    };

    render(
      <LiveRecognitionPanel
        prediction={prediction}
        candidate={candidate}
        mode="guided"
        stateCode="STATE_D_CANDIDATE"
        onConfirmWord={handleConfirm}
        onRejectCandidate={handleReject}
        confidenceThreshold={0.4}
      />
    );

    const drinkElements = screen.getAllByText(/DRINK/i);
    expect(drinkElements.length).toBeGreaterThan(0);
    expect(screen.getAllByText('88%').length).toBeGreaterThan(0);

    const confirmBtn = screen.getByRole('button', { name: /confirm word/i });
    fireEvent.click(confirmBtn);
    expect(handleConfirm).toHaveBeenCalledTimes(1);

    const rejectBtn = screen.getByRole('button', { name: /reject \/ reset/i });
    fireEvent.click(rejectBtn);
    expect(handleReject).toHaveBeenCalledTimes(1);
  });

  it('renders HandQualityCoach with real measurement scores and guidance', () => {
    render(
      <HandQualityCoach
        quality={{
          score: 85,
          level: 'Good',
          feedback: ['Hand position and lighting are good.'],
          num_hands: 1,
          hand_size: 0.12,
          brightness: 120.0,
          blur_score: 80.0,
          stability: 0.95,
          buffer_completeness: 1.0,
        }}
        handsDetected={true}
        stateCode="STATE_D_CANDIDATE"
      />
    );

    expect(screen.getByText('Hand Quality Coach')).toBeDefined();
    expect(screen.getByText('Good')).toBeDefined();
    expect(screen.getByText('85 / 100')).toBeDefined();
    expect(screen.getByText(/Hand position and lighting are good/i)).toBeDefined();
  });

  it('renders PostureInspector with geometric finger joint states', () => {
    const dummyLandmarks: LandmarkPoint[] = Array.from({ length: 21 }, (_, i) => ({
      x: 0.5 + i * 0.01,
      y: 0.5 + i * 0.01,
      z: 0.0,
      valid: true,
    }));

    render(
      <PostureInspector
        leftLandmarks={dummyLandmarks}
        rightLandmarks={dummyLandmarks}
        currentWord="DRINK"
      />
    );

    expect(screen.getByText('Finger & Posture Inspector')).toBeDefined();
    expect(screen.getByText(/21 \/ 21 Landmarks/i)).toBeDefined();
    expect(screen.getByText('DRINK')).toBeDefined();
    expect(screen.getByText(/Thumb/i)).toBeDefined();
    expect(screen.getByText(/Index Finger/i)).toBeDefined();
  });

  it('renders GestureReplay with timeline controls', () => {
    const dummyPoints: LandmarkPoint[] = Array.from({ length: 21 }, () => ({
      x: 0.5,
      y: 0.5,
      z: 0.0,
      valid: true,
    }));

    const dummyFrames: ReplayFrame[] = Array.from({ length: 10 }, (_, i) => ({
      frame_idx: i,
      timestamp_ms: 1000 + i * 33,
      left_landmarks: dummyPoints,
      right_landmarks: dummyPoints,
      left_box: null,
      right_box: null,
      hands_detected: true,
      valid_points: 42,
      state_code: 'STATE_D_CANDIDATE',
      candidate: 'DRINK',
      confidence: 0.9,
    }));

    render(
      <GestureReplay
        frames={dummyFrames}
        onClose={vi.fn()}
        candidateWord="DRINK"
        confidence={0.9}
      />
    );

    expect(screen.getByText('Gesture Replay & Visual Timeline')).toBeDefined();
    expect(screen.getByText('Frame 1 / 10')).toBeDefined();
    expect(screen.getByText(/Candidate: DRINK/i)).toBeDefined();
  });

  it('renders RobustnessLab thesis diagnostic assistant', () => {
    render(
      <RobustnessLab
        validPoints={553}
        totalPoints={553}
        stability={0.9}
        bufferCompleteness={1.0}
        confidence={0.85}
        latencyMs={15}
      />
    );

    expect(screen.getByText('Robustness Lab')).toBeDefined();
    expect(screen.getByText(/Thesis Research Diagnostic Assistant/i)).toBeDefined();
    expect(screen.getByText(/Run Controlled Perturbation Test/i)).toBeDefined();
  });

  it('renders PracticeMode with target gloss and evaluate action', () => {
    render(
      <PracticeMode
        currentPrediction="DRINK"
        confidence={0.85}
        qualityScore={90}
      />
    );

    expect(screen.getByText('Guided Practice Mode')).toBeDefined();
    expect(screen.getByText('Evaluate Practice Sign')).toBeDefined();
  });

  it('renders TranscriptWorkspace with tokens, undo, and removal', () => {
    const handleRemove = vi.fn();
    const handleClear = vi.fn();
    const handleUndo = vi.fn();
    const handleSetText = vi.fn();
    const handleNewSession = vi.fn();

    const tokens: CommittedToken[] = [
      { id: '1', word: 'HELLO', confidence: 0.9, timestamp: 100, mode: 'guided' },
      { id: '2', word: 'WORLD', confidence: 0.85, timestamp: 200, mode: 'guided' },
    ];

    render(
      <TranscriptWorkspace
        tokens={tokens}
        onRemoveToken={handleRemove}
        onClearTranscript={handleClear}
        onUndo={handleUndo}
        transcriptText="HELLO WORLD"
        setTranscriptText={handleSetText}
        onNewSession={handleNewSession}
      />
    );

    expect(screen.getByText('HELLO')).toBeDefined();
    expect(screen.getByText('WORLD')).toBeDefined();

    const undoBtn = screen.getByRole('button', { name: /undo/i });
    fireEvent.click(undoBtn);
    expect(handleUndo).toHaveBeenCalledTimes(1);

    const deleteBtns = screen.getAllByTitle('Delete word');
    expect(deleteBtns.length).toBe(2);
    fireEvent.click(deleteBtns[0]);
    expect(handleRemove).toHaveBeenCalledWith('1');
  });

  it('renders SpeechWorkspace controls with Web Speech API', () => {
    render(
      <SpeechWorkspace
        textToSpeak="HELLO WORLD"
        autoSpeak={false}
        rate={1.0}
        pitch={1.0}
        voiceName=""
        onRateChange={vi.fn()}
        onPitchChange={vi.fn()}
        onVoiceChange={vi.fn()}
      />
    );

    expect(screen.getByText('Text-to-Speech (TTS)')).toBeDefined();
    const speakBtn = screen.getByRole('button', { name: /speak transcript/i });
    expect(speakBtn).toBeDefined();
    fireEvent.click(speakBtn);
  });

  it('renders SettingsModal and handles updates', () => {
    const handleUpdate = vi.fn();
    const settings: AppSettings = {
      confidenceThreshold: 0.45,
      debounceFrames: 5,
      pauseThresholdSec: 0.6,
      targetFps: 12,
      autoSpeak: false,
      ttsRate: 1.0,
      ttsPitch: 1.0,
      ttsVoice: '',
      mirrorCamera: true,
      showLandmarkGuide: true,
      wsUrl: 'ws://127.0.0.1:8000/ws/live',
    };

    render(
      <SettingsModal
        isOpen={true}
        onClose={vi.fn()}
        settings={settings}
        onUpdateSettings={handleUpdate}
      />
    );

    expect(screen.getByText('Recognition & App Settings')).toBeDefined();
    expect(screen.getByText('45%')).toBeDefined();
    expect(screen.getByText('5 frames')).toBeDefined();
  });
});

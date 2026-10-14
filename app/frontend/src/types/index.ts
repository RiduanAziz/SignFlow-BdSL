export type RecognitionMode = 'guided' | 'continuous';

export type RecognitionStateCode =
  | 'STATE_A_INACTIVE'
  | 'STATE_B_NO_HAND'
  | 'STATE_C_COLLECTING'
  | 'STATE_D_CANDIDATE'
  | 'STATE_E_COMMITTED'
  | 'STATE_F_UNCERTAIN'
  | 'STATE_G_MODEL_UNAVAILABLE';

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
  valid: boolean;
}

export interface BoundingBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  width: number;
  height: number;
}

export interface QualityCoachMetrics {
  score: number;
  level: 'Good' | 'Needs Improvement';
  feedback: string[];
  num_hands: number;
  hand_size: number;
  brightness: number;
  blur_score: number;
  stability: number;
  buffer_completeness: number;
  is_clipped?: boolean;
  is_centered?: boolean;
}

export interface HandLandmarksPayload {
  frame_id?: number;
  left_landmarks: LandmarkPoint[];
  right_landmarks: LandmarkPoint[];
  left_box: BoundingBox | null;
  right_box: BoundingBox | null;
  hands_detected: boolean;
  left_hand: boolean;
  right_hand: boolean;
  pose: boolean;
  face: boolean;
  valid_points: number;
  quality: QualityCoachMetrics;
  state_code: RecognitionStateCode;
  timestamp_ms: number;
}

export interface FrameProcessedAck {
  type: 'frame_processed';
  frame_id: number;
  timestamp_ms: number;
  processing_ms: number;
  success: boolean;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  status: string;
  checkpoint_path: string | null;
  num_classes: number;
  input_dim: number;
  sequence_length: number;
  device: string;
  feature_representation: string;
  class_names_sample: string[];
  total_classes: number;
  error: string | null;
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'error';
  process_alive: boolean;
  backend_initialized: boolean;
  websocket_available?: boolean;
  mediapipe_asset_present?: boolean;
  mediapipe_initialized?: boolean;
  mediapipe_available: boolean;
  trained_model_loaded: boolean;
  checkpoint_compatible?: boolean;
  inference_ready: boolean;
  checkpoint_path: string | null;
  error: string | null;
  mediapipe_error: string | null;
}

export interface CommittedToken {
  id: string;
  word: string;
  bengali_word?: string;
  english_gloss?: string;
  confidence: number;
  timestamp: number;
  mode: RecognitionMode;
  isEdited?: boolean;
}

export interface PerformanceMetrics {
  fps: number;
  extraction_ms: number;
  inference_ms: number;
  total_latency_ms: number;
  frame_count: number;
}

export interface LandmarksStatus {
  hands_detected: boolean;
  left_hand: boolean;
  right_hand: boolean;
  pose: boolean;
  face: boolean;
  valid_points: number;
}

export interface CandidatePrediction {
  class_index: number;
  label: string;
  bengali_label?: string;
  english_gloss?: string;
  transliteration?: string;
  confidence: number;
}

export interface LivePrediction {
  predicted_label: string;
  bengali_label?: string;
  english_gloss?: string;
  transliteration?: string;
  predicted_class?: number;
  confidence: number;
  meets_threshold: boolean;
  top_k: CandidatePrediction[];
  latency_ms: number;
  state: string;
  state_code?: RecognitionStateCode;
}

export interface WordCandidate {
  word: string | null;
  bengali_word?: string | null;
  english_gloss?: string | null;
  confidence: number;
  stability_count: number;
  target_count: number;
  progress: number;
}

export interface VocabularyItem {
  class_id: number;
  dataset_label: string;
  bengali_label: string;
  english_gloss: string;
  transliteration?: string;
  source?: string;
  notes?: string;
}

export interface AppSettings {
  confidenceThreshold: number;
  debounceFrames: number;
  pauseThresholdSec: number;
  targetFps: number;
  autoSpeak: boolean;
  ttsRate: number;
  ttsPitch: number;
  ttsVoice: string;
  mirrorCamera: boolean;
  showLandmarkGuide: boolean;
  wsUrl: string;
}

export type CameraViewMode = 'full' | 'focus' | 'split';

export interface ReplayFrame {
  frame_idx: number;
  timestamp_ms: number;
  left_landmarks: LandmarkPoint[];
  right_landmarks: LandmarkPoint[];
  left_box: BoundingBox | null;
  right_box: BoundingBox | null;
  hands_detected: boolean;
  valid_points: number;
  state_code: string;
  candidate: string | null;
  confidence: number;
  features?: number[];
}

export interface RobustnessExperimentResult {
  original_prediction: string;
  original_confidence: number;
  perturbed_prediction: string;
  perturbed_confidence: number;
  prediction_consistent: boolean;
  prediction_changed: boolean;
  perturbation_type: string;
  severity: number;
  processing_time_ms: number;
  clean_quality: {
    sequence_length: number;
    feature_dim: number;
    sparsity_ratio: number;
  };
  perturbed_quality: {
    sequence_length: number;
    feature_dim: number;
    sparsity_ratio: number;
  };
  note: string;
}

export type FingerState = 'extended' | 'flexed' | 'curled' | 'unknown';

export interface PostureData {
  thumb: FingerState;
  index: FingerState;
  middle: FingerState;
  ring: FingerState;
  pinky: FingerState;
  estimatedShape: string;
  hand: 'Left' | 'Right' | 'None';
}

export interface PracticeSessionItem {
  id: string;
  target: string;
  bengali_target?: string;
  predicted: string;
  bengali_predicted?: string;
  confidence: number;
  passed: boolean;
  timestamp: number;
}

export interface CalibrationData {
  typicalHandSize: number;
  preferredPosition: string;
  handRequirement: 'single' | 'both';
  calibrated: boolean;
}

import React, { useState } from 'react';
import {
  FlaskConical,
  Zap,
  CheckCircle,
  AlertTriangle,
  Activity,
  Layers,
  Sparkles,
  Info,
  Clock,
  ArrowRight,
} from 'lucide-react';
import { RobustnessExperimentResult, LandmarkPoint } from '../types';

interface RobustnessLabProps {
  validPoints: number;
  totalPoints?: number;
  stability: number;
  bufferCompleteness: number;
  confidence: number;
  latencyMs: number;
  recentFeatures?: number[][] | null;
}

export const RobustnessLab: React.FC<RobustnessLabProps> = ({
  validPoints,
  totalPoints = 553,
  stability,
  bufferCompleteness,
  confidence,
  latencyMs,
  recentFeatures,
}) => {
  const [perturbationType, setPerturbationType] = useState<string>('coordinate_jitter');
  const [severity, setSeverity] = useState<number>(0.25);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [result, setResult] = useState<RobustnessExperimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const missingPoints = Math.max(0, totalPoints - validPoints);
  const validPercentage = Math.round((validPoints / totalPoints) * 100);

  const perturbationOptions = [
    { id: 'coordinate_jitter', label: 'Coordinate Jitter (Spatial)', desc: 'Gaussian sensor jitter on coordinates' },
    { id: 'translation', label: 'Translation Shift (Spatial)', desc: 'Camera panning and displacement offset' },
    { id: 'scale', label: 'Scale Variation (Spatial)', desc: 'Distance changes and hand scaling' },
    { id: 'landmark_dropout', label: 'Landmark Dropout (Spatial)', desc: 'Missing or occluded hand joints' },
    { id: 'frame_drop', label: 'Frame Drop (Temporal)', desc: 'Camera shutter stutter and frame drops' },
    { id: 'frame_duplicate', label: 'Frame Duplication (Temporal)', desc: 'Processing latency / video stalls' },
    { id: 'sequence_truncate', label: 'Sequence Truncation (Temporal)', desc: 'Abrupt gesture cutoff' },
  ];

  const hasRealFeatures = Boolean(recentFeatures && recentFeatures.length >= 8);

  const runExperiment = async () => {
    if (!hasRealFeatures || !recentFeatures) {
      setError('A live gesture sequence with at least 8 frames is required. Perform a sign in the camera view first.');
      return;
    }

    setIsRunning(true);
    setError(null);

    try {
      const response = await fetch('/api/robustness/experiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sequence: recentFeatures,
          perturbation_type: perturbationType,
          severity,
          seed: 42,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(errData.detail || 'Experiment request failed');
      }

      const data: RobustnessExperimentResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to execute experiment');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FlaskConical className="w-5 h-5 text-indigo-600" />
          <div>
            <h3 className="text-sm font-semibold text-slate-800">Robustness Lab</h3>
            <p className="text-[11px] text-slate-400">Thesis Research Diagnostic Assistant</p>
          </div>
        </div>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
          Experimental
        </span>
      </div>

      {/* Live Diagnostic Indicators */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
        <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[10px] text-slate-400 uppercase font-medium">Valid Points</span>
          <p className="font-bold font-mono text-slate-800 mt-0.5">
            {validPercentage}% ({validPoints}/{totalPoints})
          </p>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[10px] text-slate-400 uppercase font-medium">Missing Points</span>
          <p className="font-bold font-mono text-amber-700 mt-0.5">{missingPoints}</p>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[10px] text-slate-400 uppercase font-medium">Window Completeness</span>
          <p className="font-bold font-mono text-teal-700 mt-0.5">
            {Math.round(bufferCompleteness * 100)}%
          </p>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[10px] text-slate-400 uppercase font-medium">Inference Latency</span>
          <p className="font-bold font-mono text-slate-800 mt-0.5">{latencyMs} ms</p>
        </div>
      </div>

      {/* Controlled Perturbation Setup */}
      <div className="p-3.5 rounded-xl bg-slate-50/70 border border-slate-200/80 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <label className="text-xs font-semibold text-slate-700">Perturbation Operator:</label>
          <select
            value={perturbationType}
            onChange={(e) => setPerturbationType(e.target.value)}
            className="text-xs bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            {perturbationOptions.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Severity Slider */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-600">
            <span>Noise Severity:</span>
            <span className="font-mono font-bold text-indigo-600">{Math.round(severity * 100)}%</span>
          </div>
          <input
            type="range"
            min={0.05}
            max={0.8}
            step={0.05}
            value={severity}
            onChange={(e) => setSeverity(Number(e.target.value))}
            className="w-full accent-indigo-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>Mild (5%)</span>
            <span>Moderate (40%)</span>
            <span>Severe (80%)</span>
          </div>
        </div>

        {!hasRealFeatures && (
          <div className="p-2.5 rounded-xl bg-amber-50 border border-amber-200/80 text-amber-800 text-xs flex items-start space-x-2">
            <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold block">Live Gesture Required:</span>
              <span>Perform a sign in the camera view to capture real landmark features before running perturbation testing.</span>
            </div>
          </div>
        )}

        {hasRealFeatures && (
          <div className="p-2 rounded-xl bg-emerald-50 border border-emerald-200/80 text-emerald-800 text-xs flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Live Gesture Captured ({recentFeatures?.length} frames ready)</span>
            </div>
            <span className="font-mono text-[11px] font-semibold text-emerald-700">1659 features/frame</span>
          </div>
        )}

        {/* Action Button */}
        <button
          onClick={runExperiment}
          disabled={isRunning || !hasRealFeatures}
          className="w-full py-2 px-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold shadow-xs flex items-center justify-center space-x-2 transition-colors"
        >
          {isRunning ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              <span>Simulating Perturbation...</span>
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              <span>Run Controlled Perturbation Test</span>
            </>
          )}
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs">
          {error}
        </div>
      )}

      {/* Experiment Results Comparison */}
      {result && (
        <div className="p-3.5 rounded-xl border border-indigo-100 bg-indigo-50/30 space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-700">Perturbation Test Outcome:</span>
            <span
              className={`text-[11px] font-semibold px-2 py-0.5 rounded-md border ${
                result.prediction_consistent
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                  : 'bg-amber-50 border-amber-200 text-amber-800'
              }`}
            >
              {result.prediction_consistent ? 'Prediction Consistent' : 'Prediction Changed'}
            </span>
          </div>

          {/* Clean vs Perturbed Cards */}
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-[10px] text-slate-400 uppercase font-medium">Clean Input</span>
              <p className="font-bold text-sm text-slate-800 mt-1">
                {result.original_prediction.toUpperCase()}
              </p>
              <p className="text-[11px] font-mono text-slate-500 mt-0.5">
                Conf: {Math.round(result.original_confidence * 100)}%
              </p>
            </div>

            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-[10px] text-indigo-500 uppercase font-medium">Perturbed Input</span>
              <p className="font-bold text-sm text-indigo-900 mt-1">
                {result.perturbed_prediction.toUpperCase()}
              </p>
              <p className="text-[11px] font-mono text-indigo-600 mt-0.5">
                Conf: {Math.round(result.perturbed_confidence * 100)}%
              </p>
            </div>
          </div>

          <div className="flex justify-between items-center text-[11px] text-slate-500 pt-1 border-t border-indigo-100/60">
            <span>
              Operator: <strong className="text-slate-700">{result.perturbation_type}</strong>
            </span>
            <span>
              Processing Time: <strong className="font-mono text-slate-700">{result.processing_time_ms} ms</strong>
            </span>
          </div>
        </div>
      )}

      {/* Critical Thesis Scientific Disclaimer */}
      <div className="flex items-start space-x-1.5 text-[11px] text-slate-400 bg-slate-50/50 p-2.5 rounded-lg border border-slate-100">
        <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
        <p>
          <strong>Scientific Notice:</strong> Live experiments report <em>prediction consistency</em> on
          unlabelled sequences, which does not claim or prove ground-truth correctness. True model
          accuracy and macro-F1 degradation under noise are evaluated on labelled offline benchmark datasets.
        </p>
      </div>
    </div>
  );
};

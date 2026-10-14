import React from 'react';
import {
  Sparkles,
  Check,
  X,
  Clock,
  Layers,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  ShieldAlert,
  HelpCircle,
} from 'lucide-react';
import { LivePrediction, WordCandidate, RecognitionMode, RecognitionStateCode } from '../types';

interface LiveRecognitionPanelProps {
  prediction: LivePrediction | null;
  candidate: WordCandidate | null;
  mode: RecognitionMode;
  stateCode: RecognitionStateCode;
  onConfirmWord: () => void;
  onRejectCandidate: () => void;
  confidenceThreshold: number;
  lastCommittedWord?: string | null;
}

export const LiveRecognitionPanel: React.FC<LiveRecognitionPanelProps> = ({
  prediction,
  candidate,
  mode,
  stateCode,
  onConfirmWord,
  onRejectCandidate,
  confidenceThreshold,
  lastCommittedWord,
}) => {
  const activeWord =
    candidate?.bengali_word ||
    candidate?.word ||
    (prediction?.meets_threshold ? (prediction.bengali_label || prediction.predicted_label) : null);
  const secondaryGloss =
    candidate?.english_gloss ||
    (prediction?.meets_threshold ? prediction.english_gloss : null);
  const confidence = candidate?.confidence ?? prediction?.confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);

  const stateConfigs: Record<
    RecognitionStateCode,
    { label: string; bg: string; text: string; dot: string }
  > = {
    STATE_A_INACTIVE: { label: 'Camera Off', bg: 'bg-slate-100', text: 'text-slate-600', dot: 'bg-slate-400' },
    STATE_B_NO_HAND: { label: 'No Hand', bg: 'bg-slate-100', text: 'text-slate-600', dot: 'bg-slate-400' },
    STATE_C_COLLECTING: { label: 'Collecting Sign', bg: 'bg-teal-50', text: 'text-teal-700', dot: 'bg-teal-500 animate-pulse' },
    STATE_D_CANDIDATE: { label: 'Candidate Active', bg: 'bg-indigo-50', text: 'text-indigo-700', dot: 'bg-indigo-500' },
    STATE_E_COMMITTED: { label: 'Word Confirmed', bg: 'bg-emerald-50', text: 'text-emerald-700', dot: 'bg-emerald-500' },
    STATE_F_UNCERTAIN: { label: 'Sign Uncertain', bg: 'bg-amber-50', text: 'text-amber-700', dot: 'bg-amber-500' },
    STATE_G_MODEL_UNAVAILABLE: { label: 'Model Unavailable', bg: 'bg-rose-50', text: 'text-rose-700', dot: 'bg-rose-500' },
  };

  const currentCfg = stateConfigs[stateCode] || stateConfigs.STATE_B_NO_HAND;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col justify-between space-y-4">
      {/* Header with Recognition State */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-indigo-600" />
          <h2 className="text-base font-semibold text-slate-800">Live Recognition</h2>
        </div>

        {/* State Badge */}
        <div
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${currentCfg.bg} ${currentCfg.text}`}
        >
          <span className={`w-2 h-2 rounded-full ${currentCfg.dot}`} />
          <span>{currentCfg.label}</span>
        </div>
      </div>

      {/* Main Status & Candidate Card */}
      <div className="bg-gradient-to-b from-slate-50 to-white rounded-xl border border-slate-200/90 p-5 flex flex-col items-center justify-center text-center space-y-3 relative overflow-hidden shadow-xs min-h-[175px]">
        {/* State G: Model Unavailable */}
        {stateCode === 'STATE_G_MODEL_UNAVAILABLE' ? (
          <div className="flex flex-col items-center justify-center text-rose-700 space-y-2 py-2">
            <ShieldAlert className="w-9 h-9 text-rose-500" />
            <p className="font-bold text-sm">Recognition Model Unavailable</p>
            <p className="text-xs text-rose-600 max-w-xs">
              Trained PyTorch checkpoint was not found. Recognition is disabled. Please verify
              checkpoint path.
            </p>
          </div>
        ) : stateCode === 'STATE_F_UNCERTAIN' ? (
          /* State F: Uncertain Sign */
          <div className="flex flex-col items-center justify-center text-amber-700 space-y-2 py-2">
            <AlertCircle className="w-9 h-9 text-amber-500" />
            <p className="font-bold text-sm">Sign Uncertain</p>
            <p className="text-xs text-amber-600 max-w-xs">
              Hand visibility or motion sequence did not exceed the required confidence threshold.
              Improve hand lighting or repeat the gesture.
            </p>
          </div>
        ) : activeWord ? (
          /* State D or E: Candidate or Confirmed */
          <>
            <div className="text-[11px] uppercase font-mono tracking-widest text-slate-400">
              {stateCode === 'STATE_E_COMMITTED' ? 'RECOGNIZED BANGLA SIGN' : 'PROVISIONAL CANDIDATE'}
            </div>
            <div
              className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${
                stateCode === 'STATE_E_COMMITTED' ? 'text-emerald-700' : 'text-slate-900'
              }`}
            >
              {activeWord}
            </div>
            {secondaryGloss && secondaryGloss !== activeWord && (
              <div className="text-xs font-medium text-slate-500 -mt-1">
                English Gloss: {secondaryGloss}
              </div>
            )}

            {/* Confidence & Stability Bar */}
            <div className="w-full max-w-xs space-y-1.5 pt-1">
              <div className="flex justify-between items-center text-xs font-medium">
                <span className="text-slate-500">Measured Confidence</span>
                <span
                  className={
                    confidencePct >= 70
                      ? 'text-emerald-600 font-bold'
                      : confidencePct >= 45
                      ? 'text-amber-600 font-bold'
                      : 'text-slate-500'
                  }
                >
                  {confidencePct}%
                </span>
              </div>

              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-200 ${
                    confidencePct >= 70
                      ? 'bg-emerald-500'
                      : confidencePct >= 45
                      ? 'bg-amber-500'
                      : 'bg-slate-400'
                  }`}
                  style={{ width: `${Math.min(100, confidencePct)}%` }}
                />
              </div>

              {mode === 'guided' && candidate && (
                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
                  <span>Stability filter</span>
                  <span className="font-mono">
                    {candidate.stability_count} / {candidate.target_count} frames
                  </span>
                </div>
              )}
            </div>
          </>
        ) : (
          /* State B or C: Waiting or Collecting */
          <div className="flex flex-col items-center justify-center text-slate-400 space-y-2 py-4">
            <Clock className="w-8 h-8 text-slate-300 stroke-1" />
            <p className="text-sm font-medium text-slate-500">
              {stateCode === 'STATE_C_COLLECTING'
                ? 'Collecting sign motion frames...'
                : 'Waiting for sign gesture...'}
            </p>
            <p className="text-xs text-slate-400 max-w-xs">
              Perform a Bangla sign (BDSLW131) in front of the camera. The system will detect your motion.
            </p>
          </div>
        )}
      </div>

      {/* Manual Actions: Confirm Word / Reject */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={onConfirmWord}
          disabled={!activeWord}
          className={`flex items-center justify-center space-x-1.5 py-2.5 px-3 rounded-xl text-xs font-semibold shadow-xs transition-all ${
            activeWord
              ? 'bg-emerald-600 hover:bg-emerald-700 text-white cursor-pointer active:scale-98'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed'
          }`}
          title="Manually confirm and commit word to transcript"
        >
          <Check className="w-4 h-4" />
          <span>Confirm Word</span>
        </button>

        <button
          onClick={onRejectCandidate}
          disabled={!activeWord}
          className={`flex items-center justify-center space-x-1.5 py-2.5 px-3 rounded-xl border text-xs font-semibold transition-all ${
            activeWord
              ? 'bg-white hover:bg-slate-50 text-slate-700 border-slate-200 cursor-pointer active:scale-98'
              : 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed'
          }`}
          title="Reject candidate and reset gesture buffer"
        >
          <X className="w-4 h-4" />
          <span>Reject / Reset</span>
        </button>
      </div>

      {/* Probability Distribution */}
      <div className="space-y-2 pt-1 border-t border-slate-100">
        <div className="flex items-center justify-between text-xs text-slate-600 font-medium">
          <span className="flex items-center space-x-1">
            <TrendingUp className="w-3.5 h-3.5 text-slate-400" />
            <span>Top Candidates Distribution</span>
          </span>
          <span className="text-[11px] text-slate-400 font-mono">Ranked</span>
        </div>

        {prediction?.top_k && prediction.top_k.length > 0 ? (
          <div className="space-y-1.5">
            {prediction.top_k.slice(0, 4).map((cand, idx) => {
              const pct = Math.round(cand.confidence * 100);
              const label = cand.bengali_label || cand.label;
              const sub = cand.english_gloss && cand.english_gloss !== label ? cand.english_gloss : null;
              return (
                <div key={idx} className="flex items-center text-xs space-x-2">
                  <div className="w-28 truncate font-medium text-slate-700">
                    <div>{label}</div>
                    {sub && <div className="text-[10px] text-slate-400 font-normal leading-tight">{sub}</div>}
                  </div>
                  <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        idx === 0 ? 'bg-teal-500' : 'bg-slate-300'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-8 text-right font-mono text-[11px] text-slate-500">
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-2 text-xs text-slate-400 italic">
            No active candidates
          </div>
        )}
      </div>

      {/* Mode Advisory Notice */}
      <div
        className={`p-3 rounded-xl text-xs flex items-start space-x-2 border ${
          mode === 'guided'
            ? 'bg-slate-50 border-slate-200 text-slate-600'
            : 'bg-amber-50/60 border-amber-200/80 text-amber-800'
        }`}
      >
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
        <div className="text-[11px] leading-relaxed">
          {mode === 'guided' ? (
            <>
              <strong>Guided Mode:</strong> Sign one word at a time, then pause. Confirmed words are
              safely committed to your transcript.
            </>
          ) : (
            <>
              <strong>Continuous Mode (Experimental):</strong> Heuristic temporal segmentation.
              Candidates stream as sliding windows complete.
            </>
          )}
        </div>
      </div>
    </div>
  );
};

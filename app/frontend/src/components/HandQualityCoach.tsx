import React from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  Sun,
  Maximize2,
  Move,
  Eye,
  Activity,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { QualityCoachMetrics, RecognitionStateCode } from '../types';

interface HandQualityCoachProps {
  quality: QualityCoachMetrics | null;
  handsDetected: boolean;
  stateCode: RecognitionStateCode;
}

export const HandQualityCoach: React.FC<HandQualityCoachProps> = ({
  quality,
  handsDetected,
  stateCode,
}) => {
  const score = quality?.score ?? (handsDetected ? 70 : 15);
  const isGood = score >= 70;
  const feedback = quality?.feedback || (
    handsDetected
      ? ['Hand position and lighting are good.']
      : ['No hand detected. Bring your hand into the camera view.']
  );

  const getScoreColor = (val: number) => {
    if (val >= 70) return 'text-emerald-600 bg-emerald-50 border-emerald-200';
    if (val >= 45) return 'text-amber-600 bg-amber-50 border-amber-200';
    return 'text-rose-600 bg-rose-50 border-rose-200';
  };

  const getProgressColor = (val: number) => {
    if (val >= 70) return 'bg-emerald-500';
    if (val >= 45) return 'bg-amber-500';
    return 'bg-rose-500';
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-teal-600" />
          <h3 className="text-sm font-semibold text-slate-800">Hand Quality Coach</h3>
        </div>
        <span
          className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${getScoreColor(
            score
          )}`}
        >
          {quality?.level || (isGood ? 'Good' : 'Needs Improvement')}
        </span>
      </div>

      {/* Main Score & Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between items-baseline text-xs">
          <span className="text-slate-500 font-medium">Input Quality Index</span>
          <span className="font-bold font-mono text-slate-800 text-sm">{score} / 100</span>
        </div>
        <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
          <div
            className={`h-2.5 rounded-full transition-all duration-300 ${getProgressColor(score)}`}
            style={{ width: `${score}%` }}
          />
        </div>
        <p className="text-[11px] text-slate-400 italic">
          Calculated from real spatial framing, lighting, clipping, and landmark stability.
        </p>
      </div>

      {/* Actionable Feedback Guidance */}
      <div className="space-y-1.5">
        <span className="text-xs font-medium text-slate-700">Posture & Framing Guidance:</span>
        <div className="space-y-1">
          {feedback.map((item, idx) => (
            <div
              key={idx}
              className={`text-xs px-2.5 py-1.5 rounded-lg flex items-start space-x-2 border ${
                item.includes('good') || item.includes('OK')
                  ? 'bg-emerald-50/70 border-emerald-100 text-emerald-800'
                  : 'bg-amber-50/70 border-amber-100 text-amber-800'
              }`}
            >
              {item.includes('good') || item.includes('OK') ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
              )}
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Measurable Diagnostic Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
        <div className="bg-slate-50 border border-slate-100 rounded-xl p-2.5 text-center">
          <div className="flex items-center justify-center space-x-1 text-slate-500 text-[11px]">
            <Eye className="w-3.5 h-3.5 text-teal-600" />
            <span>Hands</span>
          </div>
          <span className="font-bold text-slate-800 text-xs font-mono mt-1 block">
            {quality ? quality.num_hands : handsDetected ? 1 : 0} detected
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-100 rounded-xl p-2.5 text-center">
          <div className="flex items-center justify-center space-x-1 text-slate-500 text-[11px]">
            <Maximize2 className="w-3.5 h-3.5 text-indigo-600" />
            <span>Hand Area</span>
          </div>
          <span className="font-bold text-slate-800 text-xs font-mono mt-1 block">
            {quality ? `${Math.round(quality.hand_size * 100)}%` : '--'}
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-100 rounded-xl p-2.5 text-center">
          <div className="flex items-center justify-center space-x-1 text-slate-500 text-[11px]">
            <Sun className="w-3.5 h-3.5 text-amber-600" />
            <span>Brightness</span>
          </div>
          <span className="font-bold text-slate-800 text-xs font-mono mt-1 block">
            {quality ? Math.round(quality.brightness) : '--'}
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-100 rounded-xl p-2.5 text-center">
          <div className="flex items-center justify-center space-x-1 text-slate-500 text-[11px]">
            <Activity className="w-3.5 h-3.5 text-rose-600" />
            <span>Stability</span>
          </div>
          <span className="font-bold text-slate-800 text-xs font-mono mt-1 block">
            {quality ? `${Math.round(quality.stability * 100)}%` : '--'}
          </span>
        </div>
      </div>

      {/* Dynamic Motion Tip */}
      <div className="flex items-start space-x-2 text-[11px] text-teal-800 bg-teal-50/70 p-2.5 rounded-xl border border-teal-200/60">
        <Activity className="w-4 h-4 text-teal-600 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold block text-teal-900">Clear Hand Presentation:</span>
          <span>
            Bangla signs (যেমনঃ একটা, বই, ভাত, অপেক্ষা করো) require distinct hand shape and spatial positioning. Keep hands well-lit and centered in frame.
          </span>
        </div>
      </div>

      {/* Research Note */}
      <div className="flex items-start space-x-1.5 text-[11px] text-slate-400 bg-slate-50/50 p-2 rounded-lg border border-slate-100">
        <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
        <p>
          Hand quality score reflects tracking visibility and camera framing stability. It does not
          represent word prediction confidence or classification accuracy.
        </p>
      </div>
    </div>
  );
};

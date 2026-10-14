import React from 'react';
import { X, BookOpen, Layers, ShieldCheck, Cpu, GitBranch } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-lg w-full p-6 space-y-4 max-h-[90vh] overflow-y-auto animate-in zoom-in-95 duration-150 text-xs">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-indigo-600" />
            <h3 className="text-base font-bold text-slate-800">About SignFlow-BdSL & Research</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4 text-slate-600 leading-relaxed">
          <div>
            <h4 className="font-bold text-slate-900 text-sm">Research Title</h4>
            <p className="mt-1 text-slate-700 font-medium">
              <em>
                SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic
                Spatiotemporal Features
              </em>
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 font-bold text-slate-800">
              <Cpu className="w-4 h-4 text-teal-600" />
              <span>Feature Extraction Pipeline</span>
            </div>
            <ul className="list-disc list-inside space-y-1 text-slate-600 text-[11px]">
              <li>Canonical 553-point Holistic Landmark schema (Hands: 42, Pose: 33, Face: 478).</li>
              <li>Per-frame structured geometric normalization with explicit visibility mask preservation.</li>
              <li>Spatial coordinates and motion sequences with configurable temporal windowing.</li>
              <li>Trained models: Baseline & Robust Holistic Fusion Classifier (131 BDSLW131 classes).</li>
            </ul>
          </div>

          <div className="p-3 bg-indigo-50/60 border border-indigo-100 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 font-bold text-indigo-900">
              <Layers className="w-4 h-4 text-indigo-600" />
              <span>Operating Modes & Research Integrity</span>
            </div>
            <div className="space-y-1.5 text-[11px] text-indigo-950">
              <p>
                <strong>Mode A (Guided Word Accumulation):</strong> Primary operational mode. Evaluates
                isolated sign gestures using calibrated confidence thresholds and debounce stability.
                Allows accumulating verified tokens into persistent sentences and speech.
              </p>
              <p>
                <strong>Mode B (Continuous Sign Recognition - Experimental):</strong> Continuous sliding
                window prototype. Word boundaries and temporal segmentation are active as experimental
                heuristics. Genuine continuous signing requires continuous Bangla sign sentence recordings.
              </p>
            </div>
          </div>

          <div>
            <h4 className="font-bold text-slate-900">Academic Citation & Dataset Attribution</h4>
            <p className="mt-1 text-[11px] text-slate-500">
              Built upon the BDSLW131 (Bangla Sign Language Word Dataset) by Musaddique Ali Erfan,
              containing 29,930 verified image samples across 131 Bangla sign classes with deterministic
              class-stratified train/validation/test splits.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-3 border-t border-slate-100">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

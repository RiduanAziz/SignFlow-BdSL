import React, { useState, useEffect } from 'react';
import {
  GraduationCap,
  CheckCircle2,
  XCircle,
  Sparkles,
  Info,
  RotateCcw,
  BookOpen,
  Award,
} from 'lucide-react';
import { PracticeSessionItem } from '../types';

interface PracticeModeProps {
  currentPrediction: string | null;
  confidence: number;
  qualityScore: number;
}

interface VocabEntry {
  id: number;
  label: string;
  bengali: string;
  english: string;
}

export const PracticeMode: React.FC<PracticeModeProps> = ({
  currentPrediction,
  confidence,
  qualityScore,
}) => {
  const [vocabEntries, setVocabEntries] = useState<VocabEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<VocabEntry | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [vocabError, setVocabError] = useState<string | null>(null);
  const [sessionHistory, setSessionHistory] = useState<PracticeSessionItem[]>([]);
  const [lastOutcome, setLastOutcome] = useState<'pass' | 'try_again' | null>(null);

  // Load vocabulary from backend checkpoint
  useEffect(() => {
    const fetchVocabulary = async () => {
      setIsLoading(true);
      setVocabError(null);
      try {
        const url =
          typeof window !== 'undefined' && window.location.origin && !window.location.origin.includes(':5173')
            ? `${window.location.origin}/api/vocabulary`
            : 'http://127.0.0.1:8000/api/vocabulary';
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          if (data.available && data.items && data.items.length > 0) {
            const entries: VocabEntry[] = data.items.map((it: any, idx: number) => ({
              id: it.class_id ?? idx,
              label: it.dataset_label || it.bengali_label || `Class_${idx}`,
              bengali: it.bengali_label || it.dataset_label || `Class_${idx}`,
              english: it.english_gloss || it.dataset_label || `Class_${idx}`,
            }));
            setVocabEntries(entries);
            setSelectedEntry(entries[0]);
          } else if (data.available && data.vocabulary && data.vocabulary.length > 0) {
            const entries: VocabEntry[] = data.vocabulary.map((w: string, idx: number) => ({
              id: idx,
              label: w,
              bengali: w,
              english: w,
            }));
            setVocabEntries(entries);
            setSelectedEntry(entries[0]);
          } else {
            setVocabError('Model checkpoint is not loaded. Train or supply a checkpoint to practice.');
          }
        } else {
          setVocabError('Unable to load vocabulary from backend service.');
        }
      } catch (err) {
        setVocabError('Backend service unreachable. Start the backend server to load vocabulary.');
      } finally {
        setIsLoading(false);
      }
    };
    fetchVocabulary();
  }, []);

  const evaluateAttempt = () => {
    if (!currentPrediction || !selectedEntry) return;

    const predClean = currentPrediction.toLowerCase().trim();
    const isMatch =
      predClean === selectedEntry.label.toLowerCase().trim() ||
      predClean === selectedEntry.bengali.toLowerCase().trim() ||
      predClean === selectedEntry.english.toLowerCase().trim();
    const meetsConfidence = confidence >= 0.45;
    const passes = isMatch && meetsConfidence;

    setLastOutcome(passes ? 'pass' : 'try_again');

    const item: PracticeSessionItem = {
      id: Math.random().toString(36).substring(2, 9),
      target: selectedEntry.english,
      bengali_target: selectedEntry.bengali,
      predicted: currentPrediction,
      confidence,
      passed: passes,
      timestamp: Date.now(),
    };

    setSessionHistory((prev) => [item, ...prev].slice(0, 10));
  };

  const passCount = sessionHistory.filter((i) => i.passed).length;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <GraduationCap className="w-5 h-5 text-teal-600" />
          <h3 className="text-sm font-semibold text-slate-800">Guided Practice Mode</h3>
        </div>
        <div className="flex items-center space-x-1.5 text-xs text-slate-500 font-medium">
          <Award className="w-3.5 h-3.5 text-teal-600" />
          <span>
            {passCount} / {sessionHistory.length} Passed
          </span>
        </div>
      </div>

      {/* Target Gloss Selection */}
      <div className="p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-2">
        <label className="text-xs font-semibold text-slate-700 flex items-center justify-between">
          <span>Target Bangla Sign / Gloss (BDSLW131):</span>
          <span className="text-[11px] text-teal-700 font-normal">
            {vocabEntries.length > 0 ? `${vocabEntries.length} words in checkpoint` : 'Loading...'}
          </span>
        </label>

        {vocabEntries.length > 0 ? (
          <select
            value={selectedEntry?.id ?? ''}
            onChange={(e) => {
              const found = vocabEntries.find((v) => v.id === Number(e.target.value));
              if (found) setSelectedEntry(found);
              setLastOutcome(null);
            }}
            className="w-full text-xs font-medium bg-white border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-500 tracking-wide"
          >
            {vocabEntries.map((w) => (
              <option key={w.id} value={w.id}>
                {w.bengali} ({w.english})
              </option>
            ))}
          </select>
        ) : (
          <div className="text-xs text-slate-400 py-1">Loading model vocabulary...</div>
        )}

        {vocabError && (
          <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center space-x-2">
            <Info className="w-4 h-4 text-amber-600 shrink-0" />
            <span>{vocabError}</span>
          </div>
        )}

        <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
          <span>Required Modality: Holistic Hands + Pose</span>
          {vocabEntries.length > 0 ? (
            <span className="text-emerald-700 font-medium">Supported by Active Checkpoint</span>
          ) : (
            <span className="text-amber-700 font-medium">Checkpoint Unavailable</span>
          )}
        </div>
      </div>

      {/* Live Target vs Prediction Comparison */}
      <div className="grid grid-cols-2 gap-2 text-center text-xs">
        <div className="p-3 rounded-xl bg-teal-50/60 border border-teal-200/70">
          <span className="text-[10px] text-teal-600 uppercase font-semibold">Target Bangla Word</span>
          <p className="text-lg font-bold text-teal-900 mt-1 tracking-wide">
            {selectedEntry?.bengali || '---'}
          </p>
          {selectedEntry && (
            <p className="text-[11px] text-teal-700 font-medium">{selectedEntry.english}</p>
          )}
        </div>

        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
          <span className="text-[10px] text-slate-500 uppercase font-semibold">Live Prediction</span>
          <p className="text-lg font-bold text-slate-800 mt-1 tracking-wide">
            {currentPrediction || '---'}
          </p>
          <span className="text-[10px] text-slate-400 font-mono">
            {confidence > 0 ? `${Math.round(confidence * 100)}% Conf` : 'Awaiting sign'}
          </span>
        </div>
      </div>

      {/* Evaluation Button */}
      <button
        onClick={evaluateAttempt}
        disabled={!currentPrediction || vocabEntries.length === 0}
        className="w-full py-2.5 px-4 rounded-xl bg-teal-600 hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold shadow-xs flex items-center justify-center space-x-2 transition-colors"
      >
        <Sparkles className="w-4 h-4" />
        <span>Evaluate Practice Sign</span>
      </button>

      {/* Result Outcome Banner */}
      {lastOutcome && (
        <div
          className={`p-3 rounded-xl border flex items-center space-x-3 text-xs ${
            lastOutcome === 'pass'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
              : 'bg-amber-50 border-amber-200 text-amber-900'
          }`}
        >
          {lastOutcome === 'pass' ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
          ) : (
            <XCircle className="w-5 h-5 text-amber-600 shrink-0" />
          )}
          <div>
            <p className="font-semibold text-sm">
              {lastOutcome === 'pass' ? 'Pass! Sign Matched' : 'Try Again'}
            </p>
            <p className="text-[11px] mt-0.5 opacity-90">
              {lastOutcome === 'pass'
                ? `Predicted "${currentPrediction}" with ${Math.round(confidence * 100)}% confidence.`
                : `Target was "${selectedEntry?.bengali} (${selectedEntry?.english})", but model predicted "${currentPrediction || 'none'}" (${Math.round(
                    confidence * 100
                  )}% confidence).`}
            </p>
          </div>
        </div>
      )}

      {/* Practice Session History */}
      {sessionHistory.length > 0 && (
        <div className="space-y-1.5 pt-1">
          <span className="text-xs font-semibold text-slate-700">Recent Attempts:</span>
          <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
            {sessionHistory.map((item) => (
              <div
                key={item.id}
                className="text-xs p-2 rounded-lg bg-slate-50 border border-slate-100 flex items-center justify-between"
              >
                <div className="flex items-center space-x-2">
                  {item.passed ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  ) : (
                    <XCircle className="w-3.5 h-3.5 text-amber-600" />
                  )}
                  <span className="font-semibold text-slate-800">
                    {item.bengali_target ? `${item.bengali_target} (${item.target})` : item.target}
                  </span>
                  <span className="text-slate-400">→</span>
                  <span className="text-slate-600">{item.predicted}</span>
                </div>
                <span className="text-[11px] font-mono text-slate-400">
                  {Math.round(item.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational Notice */}
      <div className="flex items-start space-x-1.5 text-[11px] text-slate-400 bg-slate-50/50 p-2 rounded-lg border border-slate-100">
        <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
        <p>
          Practice Mode evaluates model output against target signs from the BDSLW131 vocabulary.
          A single successful classifier score does not equate to complete linguistic fluency or
          substitute for native BdSL instruction.
        </p>
      </div>
    </div>
  );
};

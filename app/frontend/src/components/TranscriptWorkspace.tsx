import React, { useState, useEffect } from 'react';
import {
  FileText,
  Copy,
  Download,
  RotateCcw,
  Trash2,
  Check,
  Plus,
  X,
  Sparkles,
  BookOpen,
  Volume2
} from 'lucide-react';
import { CommittedToken } from '../types';

interface TranscriptWorkspaceProps {
  tokens: CommittedToken[];
  onRemoveToken: (id: string) => void;
  onClearTranscript: () => void;
  onUndo: () => void;
  transcriptText: string;
  setTranscriptText: (text: string) => void;
  onNewSession: () => void;
}

export const TranscriptWorkspace: React.FC<TranscriptWorkspaceProps> = ({
  tokens,
  onRemoveToken,
  onClearTranscript,
  onUndo,
  transcriptText,
  setTranscriptText,
  onNewSession,
}) => {
  const [activeTab, setActiveTab] = useState<'gloss' | 'draft'>('gloss');
  const [copied, setCopied] = useState<boolean>(false);
  const [confirmClearOpen, setConfirmClearOpen] = useState<boolean>(false);

  // Compute formatted gloss draft assistance from tokens
  const generateEnglishDraft = (words: string[]): string => {
    if (words.length === 0) return '';
    const joined = words.join(' ').toLowerCase();
    // Capitalize first letter of sentences
    const capitalized = joined.replace(/(^\w|\.\s+\w)/g, (c) => c.toUpperCase());
    return capitalized.endsWith('.') ? capitalized : `${capitalized}.`;
  };

  const prevTokensLengthRef = React.useRef<number>(tokens.length);

  // Preserve user edits: append new tokens when they arrive without clobbering existing edits
  useEffect(() => {
    const prevLen = prevTokensLengthRef.current;
    if (tokens.length === 0) {
      if (prevLen > 0) {
        setTranscriptText('');
      }
    } else if (tokens.length > prevLen) {
      // Append newly committed word(s)
      const newTokens = tokens.slice(prevLen);
      const newWords = newTokens.map((t) => t.bengali_word || t.word).join(' ');
      const updated = transcriptText.trim() ? `${transcriptText.trim()} ${newWords}` : newWords;
      setTranscriptText(updated);
    }
    prevTokensLengthRef.current = tokens.length;
  }, [tokens, transcriptText, setTranscriptText]);

  // Copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(transcriptText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Copy failed:', err);
    }
  };

  // Download transcript as .txt
  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([transcriptText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `SignFlow_BdSL_Transcript_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Quick punctuation inserters
  const insertPunctuation = (punct: string) => {
    if (punct === ' ') {
      setTranscriptText(transcriptText + ' ');
    } else {
      const trimmed = transcriptText.trimEnd();
      setTranscriptText(trimmed + punct + ' ');
    }
  };

  const wordCount = transcriptText.trim() ? transcriptText.trim().split(/\s+/).length : 0;
  const charCount = transcriptText.length;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-6 flex flex-col space-y-4">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center space-x-2">
          <FileText className="w-5 h-5 text-indigo-600" />
          <h2 className="text-base font-semibold text-slate-800">Transcript Workspace</h2>
          <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-mono">
            {tokens.length} {tokens.length === 1 ? 'word' : 'words'}
          </span>
        </div>

        {/* View Tabs: Gloss Tokens vs English Draft */}
        <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
          <button
            onClick={() => setActiveTab('gloss')}
            className={`px-3 py-1 rounded-md font-medium transition-all ${
              activeTab === 'gloss'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Recognized Glosses
          </button>
          <button
            onClick={() => setActiveTab('draft')}
            className={`px-3 py-1 rounded-md font-medium transition-all flex items-center space-x-1 ${
              activeTab === 'draft'
                ? 'bg-white text-indigo-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            <span>English Draft</span>
          </button>
        </div>
      </div>

      {/* Gloss Chips View */}
      {activeTab === 'gloss' ? (
        <div className="space-y-3">
          <div className="min-h-[52px] p-3 bg-slate-50 border border-slate-200/80 rounded-xl flex flex-wrap gap-2 items-center">
            {tokens.length > 0 ? (
              tokens.map((token, index) => (
                <div
                  key={token.id || index}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-white border border-slate-200 shadow-2xs text-xs font-semibold text-slate-800 animate-in fade-in zoom-in-95 duration-150"
                >
                  <span className="tracking-wide text-indigo-900 font-bold">
                    {token.bengali_word || token.word}
                  </span>
                  {token.english_gloss && token.english_gloss !== (token.bengali_word || token.word) && (
                    <span className="text-[10px] text-slate-500 font-normal">
                      ({token.english_gloss})
                    </span>
                  )}
                  {token.confidence > 0 && (
                    <span className="text-[10px] text-slate-400 font-mono">
                      {Math.round(token.confidence * 100)}%
                    </span>
                  )}
                  <button
                    onClick={() => onRemoveToken(token.id)}
                    className="p-0.5 text-slate-400 hover:text-rose-600 rounded-full hover:bg-slate-100 transition-colors"
                    title="Delete word"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))
            ) : (
              <span className="text-xs text-slate-400 italic">
                Confirmed words will accumulate here as chips...
              </span>
            )}
          </div>
        </div>
      ) : (
        /* Formatted Gloss Draft View Notice */
        <div className="p-3 bg-indigo-50/60 border border-indigo-100 rounded-xl text-xs text-indigo-900 space-y-1">
          <div className="font-semibold flex items-center space-x-1.5">
            <Sparkles className="w-4 h-4 text-indigo-600" />
            <span>Formatted Gloss Draft (Assisted Formatting)</span>
          </div>
          <p className="text-[11px] text-indigo-700/80 leading-relaxed">
            Formatted Draft: <em>"{generateEnglishDraft(tokens.map((t) => t.english_gloss || t.word)) || 'No words accumulated yet.'}"</em>
            <br />
            (Note: Formats recognized glosses with sentence structure. BdSL signs do not have a 1:1 word-order syntax with English sentences.)
          </p>
        </div>
      )}

      {/* Editable Multi-Line Textarea */}
      <div className="space-y-2">
        <textarea
          value={transcriptText}
          onChange={(e) => setTranscriptText(e.target.value)}
          placeholder="Recognized words will accumulate here. You can also click to type, correct, or format the text directly..."
          rows={3}
          className="w-full p-3.5 text-sm bg-white border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all shadow-inner leading-relaxed resize-y"
        />

        {/* Counts & Punctuation Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          {/* Quick Punctuation Buttons */}
          <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
            <span className="text-[11px] font-medium text-slate-500 px-1.5">Insert:</span>
            {['.', ',', '?', '!', ' '].map((p, idx) => (
              <button
                key={idx}
                onClick={() => insertPunctuation(p)}
                className="w-6 h-6 rounded bg-white border border-slate-200 text-slate-700 font-mono text-xs hover:bg-slate-50 transition-colors flex items-center justify-center font-bold"
                title={`Insert ${p === ' ' ? 'Space' : p}`}
              >
                {p === ' ' ? '␣' : p}
              </button>
            ))}
          </div>

          <div className="text-slate-400 font-mono text-[11px]">
            {wordCount} words | {charCount} chars
          </div>
        </div>
      </div>

      {/* Transcript Action Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
        <div className="flex items-center space-x-2">
          <button
            onClick={onUndo}
            disabled={tokens.length === 0}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
              tokens.length > 0
                ? 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100 cursor-pointer'
                : 'bg-slate-50 border-slate-100 text-slate-300 cursor-not-allowed'
            }`}
            title="Undo most recent committed word"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Undo</span>
          </button>

          {!confirmClearOpen ? (
            <button
              onClick={() => setConfirmClearOpen(true)}
              disabled={tokens.length === 0 && !transcriptText}
              className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
                tokens.length > 0 || transcriptText
                  ? 'bg-slate-50 border-slate-200 text-rose-600 hover:bg-rose-50 cursor-pointer'
                  : 'bg-slate-50 border-slate-100 text-slate-300 cursor-not-allowed'
              }`}
              title="Clear transcript"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          ) : (
            <div className="flex items-center space-x-1 bg-rose-50 p-1 rounded-xl border border-rose-200 text-xs">
              <span className="text-rose-700 text-[11px] font-medium px-1">Clear all?</span>
              <button
                onClick={() => {
                  onClearTranscript();
                  setConfirmClearOpen(false);
                }}
                className="px-2 py-0.5 rounded bg-rose-600 text-white font-bold text-[11px] hover:bg-rose-700"
              >
                Yes
              </button>
              <button
                onClick={() => setConfirmClearOpen(false)}
                className="px-2 py-0.5 rounded bg-slate-200 text-slate-700 font-medium text-[11px] hover:bg-slate-300"
              >
                No
              </button>
            </div>
          )}

          <button
            onClick={onNewSession}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-all"
            title="Reset session and start new conversation"
          >
            <span>New Session</span>
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleCopy}
            disabled={!transcriptText}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
              transcriptText
                ? 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50 cursor-pointer'
                : 'bg-slate-50 border-slate-100 text-slate-300 cursor-not-allowed'
            }`}
            title="Copy transcript to clipboard"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>

          <button
            onClick={handleDownload}
            disabled={!transcriptText}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
              transcriptText
                ? 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50 cursor-pointer'
                : 'bg-slate-50 border-slate-100 text-slate-300 cursor-not-allowed'
            }`}
            title="Download transcript as text file"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export .txt</span>
          </button>
        </div>
      </div>
    </div>
  );
};

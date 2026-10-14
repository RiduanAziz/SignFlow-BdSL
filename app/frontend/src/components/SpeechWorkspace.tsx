import React, { useState, useEffect, useRef } from 'react';
import {
  Volume2,
  VolumeX,
  Play,
  Pause,
  Square,
  Sliders,
  AudioWaveform as Waveform,
  AlertCircle
} from 'lucide-react';

interface SpeechWorkspaceProps {
  textToSpeak: string;
  autoSpeak: boolean;
  rate: number;
  pitch: number;
  voiceName: string;
  onRateChange: (rate: number) => void;
  onPitchChange: (pitch: number) => void;
  onVoiceChange: (voice: string) => void;
}

export const SpeechWorkspace: React.FC<SpeechWorkspaceProps> = ({
  textToSpeak,
  autoSpeak,
  rate,
  pitch,
  voiceName,
  onRateChange,
  onPitchChange,
  onVoiceChange,
}) => {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [speechSupported, setSpeechSupported] = useState<boolean>(true);

  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Load available system voices
  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      setSpeechSupported(false);
      return;
    }

    const loadVoices = () => {
      const available = window.speechSynthesis.getVoices();
      setVoices(available);
      if (available.length > 0 && !voiceName) {
        // Prioritize Bengali (bn) voice if installed on system, fallback to English or first available
        const bnVoice = available.find((v) => v.lang.startsWith('bn'));
        const enVoice = available.find((v) => v.lang.startsWith('en')) || available[0];
        onVoiceChange((bnVoice || enVoice).name);
      }
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  // Speak handler
  const handleSpeak = () => {
    if (!speechSupported || !textToSpeak.trim()) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utteranceRef.current = utterance;

    const selectedVoice = voices.find((v) => v.name === voiceName);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }

    utterance.rate = rate;
    utterance.pitch = pitch;

    utterance.onstart = () => {
      setIsSpeaking(true);
      setIsPaused(false);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      setIsPaused(false);
    };

    utterance.onerror = (e) => {
      console.warn('Speech synthesis error:', e);
      setIsSpeaking(false);
      setIsPaused(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  // Pause / Resume
  const handleTogglePause = () => {
    if (!speechSupported) return;
    if (isSpeaking && !isPaused) {
      window.speechSynthesis.pause();
      setIsPaused(true);
    } else if (isPaused) {
      window.speechSynthesis.resume();
      setIsPaused(false);
    }
  };

  // Stop
  const handleStop = () => {
    if (!speechSupported) return;
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    setIsPaused(false);
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Volume2 className="w-5 h-5 text-teal-600" />
          <h2 className="text-base font-semibold text-slate-800">Text-to-Speech (TTS)</h2>
        </div>

        {/* Active Speaking Indicator */}
        {isSpeaking && (
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-700 text-xs font-semibold animate-pulse">
            <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping" />
            <span>Speaking...</span>
          </div>
        )}
      </div>

      {!speechSupported ? (
        <div className="p-3 bg-amber-50 border border-amber-200 text-amber-800 rounded-xl text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
          <span>Web Speech API is not supported by this browser.</span>
        </div>
      ) : (
        <>
          {/* Main TTS Action Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleSpeak}
              disabled={!textToSpeak.trim()}
              className={`inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold shadow-xs transition-all ${
                textToSpeak.trim()
                  ? 'bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 text-white cursor-pointer active:scale-98'
                  : 'bg-slate-100 text-slate-400 cursor-not-allowed'
              }`}
            >
              <Volume2 className="w-4 h-4" />
              <span>Speak Transcript</span>
            </button>

            {isSpeaking && (
              <button
                onClick={handleTogglePause}
                className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-all"
              >
                {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                <span>{isPaused ? 'Resume' : 'Pause'}</span>
              </button>
            )}

            {isSpeaking && (
              <button
                onClick={handleStop}
                className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl border border-rose-200 bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold transition-all"
              >
                <Square className="w-3.5 h-3.5" />
                <span>Stop</span>
              </button>
            )}
          </div>

          {/* Voice, Rate, and Pitch Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 border-t border-slate-100 text-xs">
            {/* Voice Dropdown */}
            <div className="space-y-1">
              <label className="text-slate-500 font-medium">Voice</label>
              <select
                value={voiceName}
                onChange={(e) => onVoiceChange(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-1.5 text-slate-800 text-xs focus:ring-1 focus:ring-teal-500 focus:outline-none"
              >
                {voices.map((v, idx) => (
                  <option key={idx} value={v.name}>
                    {v.name} ({v.lang})
                  </option>
                ))}
              </select>
            </div>

            {/* Rate Slider */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-500 font-medium">
                <span>Speed</span>
                <span className="font-mono">{rate.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={rate}
                onChange={(e) => onRateChange(parseFloat(e.target.value))}
                className="w-full accent-teal-600 cursor-pointer"
              />
            </div>

            {/* Pitch Slider */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-500 font-medium">
                <span>Pitch</span>
                <span className="font-mono">{pitch.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="1.5"
                step="0.1"
                value={pitch}
                onChange={(e) => onPitchChange(parseFloat(e.target.value))}
                className="w-full accent-teal-600 cursor-pointer"
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};

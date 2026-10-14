import React from 'react';
import { X, Sliders, Check, ShieldCheck, Sparkles } from 'lucide-react';
import { AppSettings } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdateSettings: (newSettings: Partial<AppSettings>) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-md w-full p-6 space-y-5 animate-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-teal-600" />
            <h3 className="text-base font-bold text-slate-800">Recognition & App Settings</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Settings Body */}
        <div className="space-y-4 text-xs">
          {/* Confidence Threshold */}
          <div className="space-y-1.5">
            <div className="flex justify-between font-medium text-slate-700">
              <span>Confidence Threshold</span>
              <span className="font-mono text-teal-600 font-bold">
                {Math.round(settings.confidenceThreshold * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.20"
              max="0.90"
              step="0.05"
              value={settings.confidenceThreshold}
              onChange={(e) =>
                onUpdateSettings({ confidenceThreshold: parseFloat(e.target.value) })
              }
              className="w-full accent-teal-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Minimum model probability required to register a recognition candidate.
            </p>
          </div>

          {/* Debounce / Stability Window */}
          <div className="space-y-1.5">
            <div className="flex justify-between font-medium text-slate-700">
              <span>Smoothing Window (Debounce Frames)</span>
              <span className="font-mono text-teal-600 font-bold">
                {settings.debounceFrames} frames
              </span>
            </div>
            <input
              type="range"
              min="2"
              max="10"
              step="1"
              value={settings.debounceFrames}
              onChange={(e) =>
                onUpdateSettings({ debounceFrames: parseInt(e.target.value, 10) })
              }
              className="w-full accent-teal-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Number of consecutive consistent frames required before auto-committing in Guided mode.
            </p>
          </div>

          {/* Neutral Pause Threshold */}
          <div className="space-y-1.5">
            <div className="flex justify-between font-medium text-slate-700">
              <span>Neutral Pause Duration</span>
              <span className="font-mono text-teal-600 font-bold">
                {settings.pauseThresholdSec.toFixed(1)}s
              </span>
            </div>
            <input
              type="range"
              min="0.3"
              max="2.0"
              step="0.1"
              value={settings.pauseThresholdSec}
              onChange={(e) =>
                onUpdateSettings({ pauseThresholdSec: parseFloat(e.target.value) })
              }
              className="w-full accent-teal-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Inactivity duration required to complete the gesture boundary and unlock the next sign.
            </p>
          </div>

          {/* Target FPS */}
          <div className="space-y-1.5">
            <label className="font-medium text-slate-700">Target Transmission Cadence</label>
            <div className="grid grid-cols-3 gap-2">
              {[10, 15, 20].map((fps) => (
                <button
                  key={fps}
                  onClick={() => onUpdateSettings({ targetFps: fps })}
                  className={`py-1.5 rounded-lg border font-medium transition-all ${
                    settings.targetFps === fps
                      ? 'bg-teal-50 border-teal-300 text-teal-800 font-bold'
                      : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  {fps} FPS
                </button>
              ))}
            </div>
          </div>

          {/* WebSocket Server Endpoint */}
          <div className="space-y-1.5">
            <label className="font-medium text-slate-700">WebSocket URL</label>
            <input
              type="text"
              value={settings.wsUrl}
              onChange={(e) => onUpdateSettings({ wsUrl: e.target.value })}
              className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 font-mono text-xs focus:ring-1 focus:ring-teal-500 focus:outline-none"
            />
          </div>

          {/* Privacy & Camera Toggles */}
          <div className="pt-2 border-t border-slate-100 space-y-2">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.autoSpeak}
                onChange={(e) => onUpdateSettings({ autoSpeak: e.target.checked })}
                className="rounded accent-teal-600"
              />
              <span className="text-slate-700 font-medium">Auto-Speak newly committed words</span>
            </label>

            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl text-[11px] text-slate-500 flex items-start space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>
                <strong>Privacy Guarantee:</strong> Video frames are processed strictly in local
                memory on your device/local server. No video frames are recorded or transmitted to
                external third parties.
              </span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white font-semibold text-xs shadow-xs transition-colors"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
};

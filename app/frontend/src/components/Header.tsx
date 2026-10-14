import React from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Settings,
  HelpCircle,
  Radio,
  Sliders,
  Sparkles,
  Camera,
  Layers
} from 'lucide-react';
import { HealthStatus, ModelInfo, RecognitionMode } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  modelInfo: ModelInfo | null;
  wsConnected: boolean;
  recognitionMode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
  onOpenSettings: () => void;
  onOpenHelp: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  modelInfo,
  wsConnected,
  recognitionMode,
  onModeChange,
  onOpenSettings,
  onOpenHelp,
}) => {
  const isModelReady = health?.trained_model_loaded ?? false;
  const isMediaPipeReady = health?.mediapipe_available ?? false;

  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-xs px-4 sm:px-6 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        {/* Brand & Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-indigo-600 flex items-center justify-center text-white shadow-sm ring-2 ring-teal-500/20">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-slate-900">SignFlow-BdSL</h1>
              <span className="px-2 py-0.5 text-xs font-semibold uppercase tracking-wider rounded-full bg-teal-50 text-teal-700 border border-teal-200/60">
                Live BdSL
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium hidden sm:block">
              Bangla Sign Language Word Recognition (BDSLW131)
            </p>
          </div>
        </div>

        {/* Live Status Indicators & Mode Control */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs">
          {/* Mode Switcher */}
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200">
            <button
              onClick={() => onModeChange('guided')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                recognitionMode === 'guided'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Primary: Guided Word Accumulation (Word-level isolated recognition)"
            >
              Mode A: Guided
            </button>
            <button
              onClick={() => onModeChange('continuous')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center space-x-1 ${
                recognitionMode === 'continuous'
                  ? 'bg-amber-100 text-amber-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Experimental: Continuous sign sliding window"
            >
              <span>Mode B: Continuous</span>
              <span className="text-[10px] bg-amber-200 text-amber-800 px-1 rounded uppercase">Exp</span>
            </button>
          </div>

          {/* Model Status Badge */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full border ${
              isModelReady
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : 'bg-amber-50 text-amber-700 border-amber-200'
            }`}
            title={modelInfo?.checkpoint_path || 'No model loaded'}
          >
            {isModelReady ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
            )}
            <span className="font-semibold">
              {isModelReady ? (modelInfo?.name || 'Model Ready') : 'Model Offline'}
            </span>
            {modelInfo?.total_classes ? (
              <span className="text-[10px] text-emerald-600/80">({modelInfo.total_classes} classes)</span>
            ) : null}
          </div>

          {/* MediaPipe Status Badge */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full border ${
              isMediaPipeReady
                ? 'bg-teal-50 text-teal-700 border-teal-200'
                : 'bg-slate-100 text-slate-600 border-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-teal-600" />
            <span className="font-medium">
              {isMediaPipeReady ? 'Holistic 553pts' : 'Extractor Offline'}
            </span>
          </div>

          {/* WebSocket Connection Badge */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full border ${
              wsConnected
                ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                : 'bg-rose-50 text-rose-700 border-rose-200'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                wsConnected ? 'bg-indigo-600 animate-pulse' : 'bg-rose-500'
              }`}
            />
            <span className="font-medium">{wsConnected ? 'Live WS' : 'Disconnected'}</span>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-1">
            <button
              onClick={onOpenSettings}
              className="p-1.5 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
              title="Recognition & Audio Settings"
            >
              <Settings className="w-4 h-4" />
            </button>
            <button
              onClick={onOpenHelp}
              className="p-1.5 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
              title="About & Research Information"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

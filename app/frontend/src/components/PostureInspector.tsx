import React, { useMemo } from 'react';
import { Hand, Eye, CheckCircle, XCircle, Info, ChevronRight } from 'lucide-react';
import { LandmarkPoint, FingerState } from '../types';

interface PostureInspectorProps {
  leftLandmarks: LandmarkPoint[];
  rightLandmarks: LandmarkPoint[];
  currentWord: string | null;
}

export const PostureInspector: React.FC<PostureInspectorProps> = ({
  leftLandmarks,
  rightLandmarks,
  currentWord,
}) => {
  // Select active hand (right hand prioritized, then left)
  const activeLandmarks = useMemo(() => {
    const rightValid = rightLandmarks.some((p) => p.valid);
    if (rightValid) return { points: rightLandmarks, label: 'Right Hand' };
    const leftValid = leftLandmarks.some((p) => p.valid);
    if (leftValid) return { points: leftLandmarks, label: 'Left Hand' };
    return { points: [], label: 'No Hand Detected' };
  }, [leftLandmarks, rightLandmarks]);

  // Geometric Finger State Estimation
  const fingerAnalysis = useMemo(() => {
    const pts = activeLandmarks.points;
    if (pts.length < 21) {
      return {
        thumb: 'unknown' as FingerState,
        index: 'unknown' as FingerState,
        middle: 'unknown' as FingerState,
        ring: 'unknown' as FingerState,
        pinky: 'unknown' as FingerState,
        derivedShape: 'None',
        wrist: { x: 0, y: 0, valid: false },
        fingertipCount: 0,
      };
    }

    const dist = (i1: number, i2: number) => {
      const p1 = pts[i1];
      const p2 = pts[i2];
      if (!p1.valid || !p2.valid) return 0;
      return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    };

    // Wrist
    const wrist = pts[0];

    // For index, middle, ring, pinky: compare tip-to-wrist distance vs pip-to-wrist distance
    const getFingerState = (mcp: number, pip: number, tip: number): FingerState => {
      if (!pts[tip].valid || !pts[mcp].valid) return 'unknown';
      const dTipWrist = dist(tip, 0);
      const dPipWrist = dist(pip, 0);
      const dMcpWrist = dist(mcp, 0);

      if (dTipWrist > dPipWrist * 1.15) return 'extended';
      if (dTipWrist < dMcpWrist * 1.1) return 'curled';
      return 'flexed';
    };

    const indexState = getFingerState(5, 6, 8);
    const middleState = getFingerState(9, 10, 12);
    const ringState = getFingerState(13, 14, 16);
    const pinkyState = getFingerState(17, 18, 20);

    // Thumb extension relative to index MCP
    const dThumbTipIndex = dist(4, 5);
    const dThumbMcpIndex = dist(2, 5);
    const thumbState: FingerState =
      pts[4].valid && pts[2].valid
        ? dThumbTipIndex > dThumbMcpIndex * 1.15
          ? 'extended'
          : 'flexed'
        : 'unknown';

    // Count extended fingertips
    const states = [thumbState, indexState, middleState, ringState, pinkyState];
    const extendedCount = states.filter((s) => s === 'extended').length;

    // Derived Geometric Handshape
    let derivedShape = 'Custom Configuration';
    if (extendedCount === 5) derivedShape = 'Open Palm (5 Fingers)';
    else if (extendedCount === 0 && states.every((s) => s === 'curled' || s === 'flexed'))
      derivedShape = 'Closed Fist';
    else if (indexState === 'extended' && middleState === 'curled' && ringState === 'curled')
      derivedShape = 'Point / Index Solo';
    else if (indexState === 'extended' && middleState === 'extended' && ringState === 'curled')
      derivedShape = 'V-Shape / Two Fingers';
    else if (thumbState === 'extended' && pinkyState === 'extended' && indexState === 'curled')
      derivedShape = 'Y-Shape / Horns';

    return {
      thumb: thumbState,
      index: indexState,
      middle: middleState,
      ring: ringState,
      pinky: pinkyState,
      derivedShape,
      wrist: { x: wrist.x, y: wrist.y, valid: wrist.valid },
      fingertipCount: extendedCount,
    };
  }, [activeLandmarks]);

  const renderBadge = (state: FingerState) => {
    switch (state) {
      case 'extended':
        return (
          <span className="text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
            Extended
          </span>
        );
      case 'curled':
        return (
          <span className="text-[11px] font-semibold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded-md border border-indigo-200">
            Curled
          </span>
        );
      case 'flexed':
        return (
          <span className="text-[11px] font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">
            Flexed
          </span>
        );
      default:
        return (
          <span className="text-[11px] text-slate-400 bg-slate-50 px-2 py-0.5 rounded-md border border-slate-200">
            Uncertain
          </span>
        );
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Hand className="w-5 h-5 text-teal-600" />
          <h3 className="text-sm font-semibold text-slate-800">Finger & Posture Inspector</h3>
        </div>
        <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-md bg-slate-100 text-slate-700">
          {activeLandmarks.label}
        </span>
      </div>

      {/* Distinction Banner: Geometric Points vs Derived Shape vs BdSL Sign */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
        <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200/70">
          <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider block">
            1. Geometrically Detected
          </span>
          <p className="font-semibold text-slate-800 mt-0.5">
            {activeLandmarks.points.filter((p) => p.valid).length} / 21 Landmarks
          </p>
          <p className="text-[11px] text-slate-500 mt-1">Raw spatial 3D joint coordinates</p>
        </div>

        <div className="p-2.5 rounded-xl bg-teal-50/70 border border-teal-200/70">
          <span className="text-[10px] text-teal-600 font-medium uppercase tracking-wider block">
            2. Estimated Handshape
          </span>
          <p className="font-semibold text-teal-900 mt-0.5">{fingerAnalysis.derivedShape}</p>
          <p className="text-[11px] text-teal-700 mt-1">Heuristic geometric approximation</p>
        </div>

        <div className="p-2.5 rounded-xl bg-indigo-50/70 border border-indigo-200/70">
          <span className="text-[10px] text-indigo-600 font-medium uppercase tracking-wider block">
            3. Recognized BdSL Sign
          </span>
          <p className="font-semibold text-indigo-900 mt-0.5">
            {currentWord ? currentWord.toUpperCase() : 'None Pending'}
          </p>
          <p className="text-[11px] text-indigo-700 mt-1">Learned holistic model output</p>
        </div>
      </div>

      {/* Individual Finger State Table */}
      <div className="border border-slate-100 rounded-xl overflow-hidden text-xs">
        <div className="bg-slate-50/80 px-3 py-2 font-medium text-slate-600 border-b border-slate-100 flex justify-between">
          <span>Finger Joint</span>
          <span>Geometric State</span>
        </div>
        <div className="divide-y divide-slate-100 bg-white">
          <div className="px-3 py-2 flex justify-between items-center">
            <span className="text-slate-700 font-medium">Thumb (CMC, MCP, IP, Tip)</span>
            {renderBadge(fingerAnalysis.thumb)}
          </div>
          <div className="px-3 py-2 flex justify-between items-center">
            <span className="text-slate-700 font-medium">Index Finger (MCP, PIP, DIP, Tip)</span>
            {renderBadge(fingerAnalysis.index)}
          </div>
          <div className="px-3 py-2 flex justify-between items-center">
            <span className="text-slate-700 font-medium">Middle Finger (MCP, PIP, DIP, Tip)</span>
            {renderBadge(fingerAnalysis.middle)}
          </div>
          <div className="px-3 py-2 flex justify-between items-center">
            <span className="text-slate-700 font-medium">Ring Finger (MCP, PIP, DIP, Tip)</span>
            {renderBadge(fingerAnalysis.ring)}
          </div>
          <div className="px-3 py-2 flex justify-between items-center">
            <span className="text-slate-700 font-medium">Little / Pinky Finger</span>
            {renderBadge(fingerAnalysis.pinky)}
          </div>
        </div>
      </div>

      {/* Scientific Clarification Footer */}
      <div className="flex items-start space-x-1.5 text-[11px] text-slate-400 bg-slate-50/50 p-2 rounded-lg border border-slate-100">
        <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
        <p>
          Geometric posture indicators are computed from joint ratios and are labelled as estimates.
          Static geometric handshapes are never mapped directly to BdSL vocabulary without evaluation
          by the trained holistic recognition model.
        </p>
      </div>
    </div>
  );
};

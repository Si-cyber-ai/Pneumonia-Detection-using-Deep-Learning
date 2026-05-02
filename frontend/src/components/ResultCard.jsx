import { motion } from 'framer-motion';

const CONFIG = {
  NORMAL: {
    color: 'green',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    text: 'text-emerald-400',
    glow: 'glow-green',
    ring: 'ring-emerald-500/20',
    icon: (
      <svg className="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  PNEUMONIA: {
    color: 'red',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    text: 'text-red-400',
    glow: 'glow-red',
    ring: 'ring-red-500/20',
    icon: (
      <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  },
};

const DECISION_COLORS = {
  'HIGH CONFIDENCE': { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  'UNCERTAIN':       { bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   text: 'text-amber-400',   dot: 'bg-amber-400' },
  'REVIEW REQUIRED': { bg: 'bg-red-500/10',     border: 'border-red-500/30',     text: 'text-red-400',     dot: 'bg-red-400' },
};

export default function ResultCard({ result }) {
  const diagnosis = result?.diagnosis?.toUpperCase() || 'NORMAL';
  const cfg = CONFIG[diagnosis] || CONFIG.NORMAL;
  const confidence = result?.model1?.confidence ?? 0;
  const pct = (confidence * 100).toFixed(1);
  const decision = result?.decision || 'HIGH CONFIDENCE';
  const dcfg = DECISION_COLORS[decision] || DECISION_COLORS['HIGH CONFIDENCE'];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`glass-card glass-card-hover p-6 ${cfg.glow}`}
      id="result-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-5">
        <div className={`w-2 h-2 rounded-full ${diagnosis === 'PNEUMONIA' ? 'bg-red-400' : 'bg-emerald-400'} animate-pulse`} />
        <span className="section-label">Diagnosis Result</span>
      </div>

      {/* Main result */}
      <div className="flex items-center gap-5">
        <div className={`w-16 h-16 rounded-2xl border ${cfg.border} ${cfg.bg} flex items-center justify-center shrink-0`}>
          {cfg.icon}
        </div>

        <div className="flex-1 min-w-0">
          <p className={`text-3xl font-bold tracking-tight ${cfg.text}`}>{diagnosis}</p>
          <p className="text-sm text-slate-400 mt-0.5">Primary Diagnosis</p>

          {/* Confidence bar */}
          <div className="mt-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">Confidence</span>
              <span className={`text-sm font-bold mono ${cfg.text}`}>{pct}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 1, ease: 'easeOut', delay: 0.3 }}
                className={`h-full rounded-full ${diagnosis === 'PNEUMONIA' ? 'bg-gradient-to-r from-red-600 to-red-400' : 'bg-gradient-to-r from-emerald-600 to-emerald-400'}`}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Decision status */}
      <div className="mt-5 pt-5 border-t border-slate-800">
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border ${dcfg.bg} ${dcfg.border}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${dcfg.dot} animate-pulse`} />
          <span className={`text-xs font-semibold tracking-widest uppercase ${dcfg.text}`}>{decision}</span>
        </div>
      </div>
    </motion.div>
  );
}

import { motion } from 'framer-motion';

function ModelRow({ name, label, confidence, delay }) {
  const isPneumonia = label?.toUpperCase() === 'PNEUMONIA';
  const pct = ((confidence ?? 0) * 100).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="flex items-center gap-4 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-colors"
      id={`model-row-${name.replace(/\s+/g, '-').toLowerCase()}`}
    >
      {/* Model name */}
      <div className="w-28 shrink-0">
        <p className="text-xs font-semibold text-slate-200 mono">{name}</p>
      </div>

      {/* Label badge */}
      <div className={`shrink-0 px-2.5 py-1 rounded-lg text-xs font-bold tracking-wider
        ${isPneumonia
          ? 'bg-red-500/15 text-red-400 border border-red-500/25'
          : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/25'
        }`}>
        {label?.toUpperCase() || '—'}
      </div>

      {/* Confidence bar */}
      <div className="flex-1 min-w-0">
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 1, ease: 'easeOut', delay: delay + 0.2 }}
            className={`h-full rounded-full ${isPneumonia ? 'bg-red-500' : 'bg-emerald-500'}`}
          />
        </div>
      </div>

      {/* Confidence value */}
      <div className="w-12 text-right shrink-0">
        <span className={`text-sm font-bold mono ${isPneumonia ? 'text-red-400' : 'text-emerald-400'}`}>
          {pct}%
        </span>
      </div>
    </motion.div>
  );
}

export default function ModelComparison({ result }) {
  const m1 = result?.model1;
  const m2 = result?.model2;

  const agree = m1?.label?.toUpperCase() === m2?.label?.toUpperCase();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="glass-card glass-card-hover p-6"
      id="model-comparison-card"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400" />
          <span className="section-label">Model Comparison</span>
        </div>

        {/* Agreement badge */}
        <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold border
          ${agree
            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
            : 'bg-orange-500/10 border-orange-500/30 text-orange-400'
          }`}>
          {agree ? (
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
          Agreement: {agree ? 'YES' : 'NO'}
        </div>
      </div>

      {/* Model rows */}
      <div className="flex flex-col gap-2.5">
        <ModelRow name="DenseNet121" label={m1?.label} confidence={m1?.confidence} delay={0.2} />
        <ModelRow name="ResNet50" label={m2?.label} confidence={m2?.confidence} delay={0.35} />
      </div>

      {/* Disagreement warning */}
      {!agree && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-4 flex items-start gap-2 p-3 rounded-xl bg-orange-500/8 border border-orange-500/20"
        >
          <svg className="w-4 h-4 text-orange-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="text-xs text-orange-300 leading-relaxed">
            Models disagree on diagnosis. Clinical review is strongly recommended before taking action.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}

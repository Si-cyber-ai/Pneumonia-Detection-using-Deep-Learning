import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { getDetailedExplanation } from '../services/api';

const LANGUAGES = [
  { code: 'English', label: 'English' },
  { code: 'Spanish', label: 'Spanish' },
  { code: 'French', label: 'French' },
  { code: 'German', label: 'Deutsch' },
  { code: 'Hindi', label: 'Hindi' },
  { code: 'Arabic', label: 'Arabic' },
  { code: 'Chinese (Simplified)', label: 'Chinese (Simplified)' },
];

function normalizeExplanation(text) {
  return text
    .replace(/\r/g, '')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/^[\-\u2022]\s+/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export default function DetailedExplanation({ result, onExplanationChange }) {
  const [language, setLanguage] = useState('English');
  const [explanation, setExplanation] = useState('');
  const [loading, setLoading] = useState(false);

  const diagnosis = result?.prediction || 'NORMAL';
  const confidence = result?.confidence || 0;
  const severity = result?.severity || 'NONE';

  useEffect(() => {
    let isCancelled = false;

    async function fetchExplanation() {
      if (!result) {
        return;
      }

      setLoading(true);
      setExplanation('');
      onExplanationChange?.('');

      const text = await getDetailedExplanation(diagnosis, confidence, severity, language);
      const normalizedText = normalizeExplanation(text);

      if (!isCancelled) {
        setExplanation(normalizedText);
        onExplanationChange?.(normalizedText);
        setLoading(false);
      }
    }

    fetchExplanation();

    return () => {
      isCancelled = true;
    };
  }, [result, diagnosis, confidence, severity, language, onExplanationChange]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
      className="mt-5 rounded-2xl border border-slate-200 bg-white p-6"
      id="detailed-explanation-card"
    >
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-500" />
          <span className="block text-xs font-bold uppercase tracking-wider text-slate-500">AI Detailed Explanation</span>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5">
          <svg className="h-4 w-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
          </svg>
          <select
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            className="cursor-pointer bg-transparent text-sm font-semibold text-slate-700 outline-none"
          >
            {LANGUAGES.map((item) => (
              <option key={item.code} value={item.code} className="bg-white text-slate-800">
                {item.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="min-h-[140px] rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        {loading ? (
          <div className="flex flex-col items-center justify-center gap-3 py-6">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
            <p className="text-xs text-slate-500">Generating AI explanation...</p>
          </div>
        ) : (
          <div className="space-y-4 text-slate-700">
            {(explanation || 'No explanation returned yet.')
              .split(/\n{2,}/)
              .map((paragraph) => paragraph.trim())
              .filter(Boolean)
              .map((paragraph, index) => (
                <p key={`${index}-${paragraph.slice(0, 24)}`} className="text-sm leading-7 text-slate-700">
                  {paragraph}
                </p>
              ))}
          </div>
        )}
      </div>

      <p className="mt-4 text-xs italic leading-relaxed text-slate-600">
        AI-generated text. Always consult a healthcare professional for an official diagnosis and treatment plan.
      </p>
    </motion.div>
  );
}

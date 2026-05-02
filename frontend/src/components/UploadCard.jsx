import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function UploadCard({ onAnalyze, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png'];

  const handleFile = useCallback((f) => {
    setError('');
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      setError('Please upload a JPG or PNG chest X-ray image.');
      return;
    }
    if (f.size > 20 * 1024 * 1024) {
      setError('File size must be under 20 MB.');
      return;
    }
    setFile(f);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files[0];
    handleFile(dropped);
  }, [handleFile]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    setDragActive(e.type === 'dragover');
  }, []);

  const handleChange = (e) => handleFile(e.target.files[0]);

  const handleAnalyze = () => {
    if (file && !isLoading) onAnalyze(file, preview);
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="glass-card glass-card-hover p-6 w-full">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
          <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-100">Upload Chest X-Ray</h2>
          <p className="text-xs text-slate-500">JPG / PNG · Max 20 MB</p>
        </div>
      </div>

      {/* Drop Zone */}
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            id="dropzone"
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 h-52 select-none
              ${dragActive
                ? 'border-blue-400 bg-blue-500/8 scale-[1.01]'
                : 'border-slate-700 bg-slate-900/40 hover:border-blue-500/50 hover:bg-blue-500/5'
              }`}
          >
            {dragActive && (
              <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
                <div className="scan-line" />
              </div>
            )}
            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-3 transition-colors duration-300
              ${dragActive ? 'bg-blue-500/20' : 'bg-slate-800'}`}>
              <svg className="w-7 h-7 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-300 mb-1">
              {dragActive ? 'Drop to upload' : 'Drag & drop X-ray here'}
            </p>
            <p className="text-xs text-slate-500">or <span className="text-blue-400 font-medium">browse files</span></p>
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              className="hidden"
              onChange={handleChange}
              id="xray-file-input"
            />
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-xl overflow-hidden border border-slate-700 bg-black"
          >
            <img
              src={preview}
              alt="X-ray preview"
              className="w-full object-contain max-h-64"
            />
            {/* Overlay info */}
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-3 flex items-center justify-between">
              <div>
                <p className="text-xs font-medium text-white truncate max-w-[200px]">{file?.name}</p>
                <p className="text-xs text-slate-400">{(file?.size / 1024).toFixed(1)} KB</p>
              </div>
              <button
                id="remove-image-btn"
                onClick={(e) => { e.stopPropagation(); handleReset(); }}
                className="w-7 h-7 rounded-lg bg-red-500/20 border border-red-500/30 flex items-center justify-center hover:bg-red-500/30 transition-colors"
                title="Remove image"
              >
                <svg className="w-3.5 h-3.5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="mt-3 flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20"
          >
            <svg className="w-4 h-4 text-red-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-xs text-red-400">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analyze Button */}
      <motion.button
        id="analyze-btn"
        onClick={handleAnalyze}
        disabled={!file || isLoading}
        whileTap={{ scale: 0.97 }}
        className={`mt-4 w-full h-11 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-300
          ${file && !isLoading
            ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-500 hover:to-cyan-500 glow-blue cursor-pointer'
            : 'bg-slate-800 text-slate-500 cursor-not-allowed'
          }`}
      >
        {isLoading ? (
          <>
            <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Analyze X-Ray
          </>
        )}
      </motion.button>
    </div>
  );
}

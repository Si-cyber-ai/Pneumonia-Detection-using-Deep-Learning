import { useRef, useState } from 'react';
import { motion } from 'framer-motion';

export default function GradCamViewer({ result, originalImage }) {
  const [viewMode, setViewMode] = useState('compare');
  const [activeModel, setActiveModel] = useState('DenseNet');
  const [opacity, setOpacity] = useState(0.5);
  const [isZoomed, setIsZoomed] = useState(false);
  const [splitPos, setSplitPos] = useState(50);
  const containerRef = useRef(null);

  const modelData = result?.models?.[activeModel];
  const gradcamBase64 = modelData?.heatmap;
  const gradcamSrc = gradcamBase64 ? `data:image/png;base64,${gradcamBase64}` : null;
  const hotspots = modelData?.hotspots || [];
  const topHotspot = hotspots[0];
  const focusScore = Number(result?.focus_score ?? 0);

  const handleMouseMove = (event) => {
    if (viewMode !== 'split' || !containerRef.current) {
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
    setSplitPos((x / rect.width) * 100);
  };

  const handleTouchMove = (event) => {
    if (viewMode !== 'split' || !containerRef.current) {
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(event.touches[0].clientX - rect.left, rect.width));
    setSplitPos((x / rect.width) * 100);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="relative overflow-hidden rounded-[28px] border border-[var(--color-border)] bg-white/92 p-6 shadow-[0_18px_42px_rgba(15,23,42,0.06)]"
    >
      <div className="pointer-events-none absolute -right-40 -top-40 h-96 w-96 rounded-full bg-[rgba(15,118,110,0.10)] blur-3xl" />

      <div className="relative z-10 mb-6 flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <div className="h-2.5 w-2.5 rounded-full bg-[var(--color-brand)] shadow-[0_0_12px_rgba(15,118,110,0.32)]" />
            <h2 className="font-display text-sm font-bold uppercase tracking-wider text-[var(--color-ink)]">Clinical Grad-CAM Analysis</h2>
          </div>
          <p className="text-xs text-[var(--color-copy)]">Class activation mapping using {activeModel}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2 rounded-full border border-[var(--color-border)] bg-[rgba(255,255,255,0.82)] p-1.5 shadow-sm">
          {[
            { id: 'compare', label: 'Compare' },
            { id: 'overlay', label: 'Overlay' },
            { id: 'split', label: 'Split Slider' },
            { id: 'heatmap', label: 'Heatmap Only' },
            { id: 'original', label: 'Original' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setViewMode(item.id)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-200 ${
                viewMode === item.id
                  ? 'bg-[var(--color-brand-deep)] text-white shadow-[0_10px_18px_rgba(15,61,60,0.18)]'
                  : 'text-[var(--color-copy)] hover:bg-slate-100 hover:text-[var(--color-ink)]'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative z-10 grid grid-cols-1 items-start gap-6 lg:grid-cols-4">
        <div className="flex flex-col gap-6 lg:col-span-1">
          <div className="panel-muted p-4">
            <p className="mb-3 font-label text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-copy-muted)]">Select Model</p>
            <div className="flex flex-col gap-2">
              {['DenseNet', 'ResNet'].map((model) => (
                <button
                  key={model}
                  onClick={() => setActiveModel(model)}
                  className={`flex flex-col rounded-lg border p-3 text-left transition-all duration-200 ${
                    activeModel === model
                      ? 'border-[rgba(15,118,110,0.28)] bg-[rgba(15,118,110,0.07)] shadow-[0_0_0_1px_rgba(15,118,110,0.06)]'
                      : 'border-[var(--color-border)] bg-white hover:bg-slate-50'
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between">
                    <span className={`font-display text-sm font-bold ${activeModel === model ? 'text-[var(--color-ink)]' : 'text-[var(--color-ink-soft)]'}`}>{model}</span>
                    {result?.models?.[model]?.label === 'PNEUMONIA'
                      ? <span className="h-2 w-2 rounded-full bg-red-500" title="Detected abnormalities" />
                      : <span className="h-2 w-2 rounded-full bg-emerald-500" title="Normal" />}
                  </div>
                  <span className="text-xs text-[var(--color-copy)]">Confidence: {(result?.models?.[model]?.confidence * 100 || 0).toFixed(1)}%</span>
                </button>
              ))}
            </div>
          </div>

          <div className="panel-muted p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="font-label text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-copy-muted)]">Overlay Opacity</span>
              <span className="text-xs font-bold text-[var(--color-brand)]">{Math.round(opacity * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={opacity}
              onChange={(event) => setOpacity(parseFloat(event.target.value))}
              disabled={viewMode !== 'overlay'}
              className={`w-full accent-[var(--color-brand)] transition-opacity ${viewMode !== 'overlay' ? 'cursor-not-allowed opacity-30' : 'cursor-grab opacity-100'}`}
            />
          </div>

          <div className="panel-muted p-4">
            <span className="mb-3 block font-label text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-copy-muted)]">Attention Intensity</span>
            <div className="mb-2 h-3 w-full rounded-full bg-gradient-to-r from-blue-600 via-emerald-500 via-yellow-400 to-red-600" />
            <div className="flex justify-between text-[10px] font-bold text-[var(--color-copy)]">
              <span>Low (Blue)</span>
              <span>Medium (Yellow)</span>
              <span>High (Red)</span>
            </div>
            <p className="mt-3 text-[10px] leading-relaxed text-[var(--color-copy-muted)]">
              Red regions indicate areas strongly contributing to the "{modelData?.label || 'Prediction'}" classification.
            </p>
            <div className="mt-4 border-t border-[var(--color-border)] pt-3">
              <p className="text-[11px] font-semibold text-[var(--color-ink)]">Focused attention level</p>
              <p className="mt-1 text-[11px] leading-5 text-[var(--color-copy)]">
                Current focus score is {focusScore.toFixed(3)}. Use the comparison cards below to review hotspot concentration without unused white space.
              </p>
            </div>
          </div>
        </div>

        <div className={`relative overflow-hidden rounded-[24px] border border-[var(--color-border)] bg-white/88 p-4 lg:col-span-3 ${viewMode === 'compare' ? '' : 'min-h-[400px]'}`}>
          {viewMode === 'compare' ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex h-full flex-col rounded-[20px] border border-[var(--color-border)] bg-[rgba(248,250,248,0.96)] p-3">
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <p className="font-label text-[11px] font-bold uppercase tracking-[0.2em] text-[var(--color-ink-soft)]">Grad-CAM Heatmap</p>
                    <p className="text-[11px] text-[var(--color-copy-muted)]">Overlayed on the X-ray</p>
                  </div>
                  <span className="rounded-full border border-[rgba(15,118,110,0.18)] bg-[rgba(15,118,110,0.10)] px-2 py-1 text-[10px] font-semibold text-[var(--color-brand)]">{activeModel}</span>
                </div>
                <div className={`relative aspect-square w-full overflow-hidden rounded-lg bg-black ${isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in'}`} onClick={() => setIsZoomed(!isZoomed)}>
                  <img
                    src={originalImage}
                    alt="Original X-ray"
                    className={`absolute inset-0 h-full w-full object-contain transition-transform duration-300 ${isZoomed ? 'scale-125' : 'scale-100'}`}
                  />
                  {gradcamSrc && (
                    <img
                      src={gradcamSrc}
                      alt="Grad-CAM heatmap"
                      className={`absolute inset-0 h-full w-full object-contain transition-transform duration-300 ${isZoomed ? 'scale-125' : 'scale-100'}`}
                      style={{
                        opacity,
                        mixBlendMode: 'screen',
                      }}
                    />
                  )}
                </div>
                <div className="mt-3 rounded-[18px] border border-[var(--color-border)] bg-white p-3">
                  <p className="font-label text-[11px] font-semibold uppercase tracking-wide text-[var(--color-copy-muted)]">Hotspot summary</p>
                  <p className="mt-2 text-sm leading-6 text-[var(--color-ink-soft)]">
                    {hotspots.length > 0
                      ? `${hotspots.length} concentrated region(s) detected for ${activeModel}.`
                      : `No concentrated hotspot region crossed the threshold for ${activeModel}.`}
                  </p>
                  {topHotspot && (
                    <p className="mt-2 text-sm leading-6 text-[var(--color-copy)]">
                      Largest region is centered near x {topHotspot.bbox.x}, y {topHotspot.bbox.y} with size {topHotspot.bbox.width} by {topHotspot.bbox.height} pixels.
                    </p>
                  )}
                </div>
              </div>

              <div className="flex h-full flex-col rounded-[20px] border border-[var(--color-border)] bg-[rgba(248,250,248,0.96)] p-3">
                <div className="mb-3">
                  <p className="font-label text-[11px] font-bold uppercase tracking-[0.2em] text-[var(--color-ink-soft)]">Original X-ray</p>
                  <p className="text-[11px] text-[var(--color-copy-muted)]">Grayscale clinical reference</p>
                </div>
                <div className={`relative aspect-square w-full overflow-hidden rounded-lg bg-black ${isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in'}`} onClick={() => setIsZoomed(!isZoomed)}>
                  <img
                    src={originalImage}
                    alt="Original X-ray"
                    className={`absolute inset-0 h-full w-full object-contain grayscale transition-transform duration-300 ${isZoomed ? 'scale-125' : 'scale-100'}`}
                  />
                </div>
                <div className="mt-3 rounded-[18px] border border-[var(--color-border)] bg-white p-3">
                  <p className="font-label text-[11px] font-semibold uppercase tracking-wide text-[var(--color-copy-muted)]">Reference details</p>
                  <p className="mt-2 text-sm leading-6 text-[var(--color-ink-soft)]">
                    Final prediction is {result?.prediction} with {((result?.confidence ?? 0) * 100).toFixed(1)}% confidence and severity {result?.severity}.
                  </p>
                  <p className="mt-2 text-sm leading-6 text-[var(--color-copy)]">
                    Compare the grayscale study against the heatmap to review whether highlighted regions align with visible parenchymal changes.
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div
              className="relative flex min-h-[400px] w-full items-center justify-center"
              ref={containerRef}
              onMouseMove={handleMouseMove}
              onTouchMove={handleTouchMove}
              onMouseLeave={() => viewMode === 'split' && setSplitPos(50)}
            >
              <div
                className={`relative h-[420px] w-full overflow-hidden rounded-lg bg-black sm:h-[520px] origin-center transition-transform duration-300 ease-out ${isZoomed ? 'scale-150 cursor-zoom-out' : 'scale-100 cursor-zoom-in'}`}
                onClick={() => setIsZoomed(!isZoomed)}
              >
                <img
                  src={originalImage}
                  alt="Original"
                  className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-300 ${viewMode === 'heatmap' ? 'opacity-0' : 'opacity-100'}`}
                />

                {gradcamSrc && (
                  <div
                    className="absolute inset-0 h-full w-full"
                    style={{
                      clipPath: viewMode === 'split' ? `inset(0 ${100 - splitPos}% 0 0)` : 'none',
                      opacity: viewMode === 'overlay' || viewMode === 'split' ? opacity : (viewMode === 'heatmap' ? 1 : 0),
                      transition: viewMode === 'split' ? 'none' : 'opacity 0.3s ease',
                      mixBlendMode: viewMode === 'heatmap' ? 'normal' : 'screen',
                    }}
                  >
                    <img src={gradcamSrc} alt="Grad-CAM overlay" className="h-full w-full object-contain" />
                  </div>
                )}

                {viewMode === 'split' && (
                  <div
                    className="absolute top-0 bottom-0 z-20 w-1 cursor-col-resize bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,1)]"
                    style={{ left: `${splitPos}%` }}
                  >
                    <div className="absolute left-1/2 top-1/2 flex h-8 w-6 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded border border-blue-400 bg-blue-600">
                      <div className="flex gap-0.5">
                        <div className="h-4 w-0.5 bg-white/60" />
                        <div className="h-4 w-0.5 bg-white/60" />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="absolute bottom-4 right-4 flex gap-2">
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    setIsZoomed(!isZoomed);
                  }}
                  className="tooltip-trigger flex h-10 w-10 items-center justify-center rounded-full border border-[var(--color-border-strong)] bg-white text-[var(--color-ink-soft)] hover:bg-slate-100 hover:text-[var(--color-ink)]"
                  title={isZoomed ? 'Zoom Out' : 'Zoom In'}
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    {isZoomed
                      ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
                      : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />}
                  </svg>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

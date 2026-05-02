import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predictPneumonia } from '../services/api';
import ClinicalReportPanel from '../components/ClinicalReportPanel';
import GradCamViewer from '../components/GradCamViewer';

const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
const MAX_FILE_SIZE = 20 * 1024 * 1024;

const WORKBENCH_FEATURES = [
  'Dual-model ensemble',
  'Live Grad-CAM review',
  'Portable clinical report',
];

const SESSION_GUIDE = [
  'Upload one frontal chest X-ray in JPG or PNG format.',
  'Run the real DenseNet121 and ResNet50 ensemble.',
  'Review prediction, quality, heatmaps, and exportable report.',
];

const PLATFORM_NOTES = [
  'Frontend and backend stay decoupled for local, GitHub, and Docker use.',
  'All outputs are generated from the live backend response for the uploaded image.',
  'The report flow remains export-ready without adding fake UI panels.',
];

function LoadingSpinner() {
  return (
    <div className="absolute inset-0 z-50 flex flex-col items-center justify-center rounded-[28px] bg-white/72 backdrop-blur-md">
      <div className="relative mb-4 h-16 w-16">
        <div className="absolute inset-0 rounded-full border-4 border-slate-200" />
        <div className="absolute inset-0 animate-spin rounded-full border-4 border-[var(--color-brand)] border-t-transparent" />
      </div>
      <p className="font-display text-sm font-bold tracking-wide text-[var(--color-ink)]">Analyzing scan...</p>
      <p className="mt-1 text-xs text-[var(--color-copy-muted)]">Running DenseNet121 and ResNet50 on the uploaded X-ray</p>
    </div>
  );
}

function FeaturePill({ children }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/70 bg-white/80 px-3 py-2 text-xs font-semibold text-[var(--color-ink-soft)] shadow-sm">
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-brand)]" />
      {children}
    </span>
  );
}

function ChecklistItem({ children }) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--color-brand-soft)] text-[var(--color-brand)]">
        <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <p className="text-sm leading-6 text-[var(--color-copy)]">{children}</p>
    </div>
  );
}

function UploadZone({ currentFile, onFileChange }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) {
      return;
    }

    if (!ACCEPTED_TYPES.includes(file.type)) {
      alert('Please upload a JPG or PNG chest X-ray image.');
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      alert('Please upload an image smaller than 20 MB.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      onFileChange({
        file,
        preview: event.target?.result,
      });
    };
    reader.readAsDataURL(file);
  }, [onFileChange]);

  const handleDrag = useCallback((event) => {
    event.preventDefault();
    setIsDragging(event.type === 'dragover');
  }, []);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }, [handleFile]);

  return (
    <div className="flex flex-col">
      <label className="mb-3 text-sm font-semibold text-[var(--color-ink-soft)]">Chest X-ray image</label>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        className={`relative flex h-80 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[28px] border transition-all duration-200 ${
          isDragging
            ? 'border-[var(--color-brand)] bg-[rgba(15,118,110,0.08)]'
            : currentFile
              ? 'border-[var(--color-border-strong)] bg-[rgba(255,255,255,0.96)]'
              : 'border-dashed border-[var(--color-border-strong)] bg-[rgba(255,255,255,0.84)] hover:border-[var(--color-brand)] hover:bg-[rgba(255,255,255,0.98)]'
        }`}
      >
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(15,118,110,0.08),_transparent_42%)]" />
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          className="hidden"
          onChange={(event) => handleFile(event.target.files?.[0])}
        />

        <AnimatePresence mode="wait">
          {currentFile ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0"
            >
              <img
                src={currentFile.preview}
                alt="Uploaded chest X-ray"
                className="h-full w-full object-contain bg-slate-950"
              />
              <div className="absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-slate-950/90 via-slate-900/45 to-transparent p-4">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-white">{currentFile.file.name}</p>
                  <p className="mt-1 text-xs text-slate-300">{(currentFile.file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onFileChange(null);
                  }}
                  className="rounded-full border border-white/20 bg-white/12 px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-red-600/80"
                >
                  Remove
                </button>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="relative z-10 flex max-w-md flex-col items-center px-6 text-center"
            >
              <div className={`mb-5 flex h-16 w-16 items-center justify-center rounded-[20px] border ${
                isDragging
                  ? 'border-[var(--color-brand)] bg-[rgba(15,118,110,0.12)] text-[var(--color-brand)]'
                  : 'border-[var(--color-border)] bg-white/92 text-[var(--color-ink-soft)]'
              }`}>
                <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </div>
              <p className="font-display text-xl font-bold tracking-tight text-[var(--color-ink)]">
                {isDragging ? 'Drop the X-ray here' : 'Upload a chest X-ray'}
              </p>
              <p className="mt-2 text-sm leading-6 text-[var(--color-copy)]">
                Drag and drop a frontal chest X-ray, or click to browse. The workspace accepts JPG and PNG images up to 20 MB.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const tones = {
    'HIGH CONFIDENCE': 'border-emerald-200 bg-emerald-50 text-emerald-700',
    'REVIEW REQUIRED': 'border-amber-200 bg-amber-50 text-amber-700',
  };

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] ${tones[status] || 'border-slate-200 bg-slate-50 text-slate-700'}`}>
      {status}
    </span>
  );
}

function QualityMetricsCard({ metrics }) {
  if (!metrics) {
    return null;
  }

  const blurScore = Number(metrics.blur_score ?? 0);
  const brightness = Number(metrics.brightness ?? 0);
  const qualityLabel = metrics.is_poor_quality ? 'Needs review' : 'Acceptable';
  const qualityTone = metrics.is_poor_quality
    ? 'border-amber-200 bg-amber-50 text-amber-800'
    : 'border-emerald-200 bg-emerald-50 text-emerald-800';

  let brightnessLabel = 'Normal exposure';
  if (brightness < 50) {
    brightnessLabel = 'Underexposed';
  } else if (brightness > 200) {
    brightnessLabel = 'Overexposed';
  }

  return (
    <div className="rounded-[28px] border border-[var(--color-border)] bg-white/92 p-6 shadow-[0_18px_42px_rgba(15,23,42,0.05)]">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="section-kicker">Image Quality Metrics</p>
          <p className="mt-2 text-sm text-[var(--color-copy)]">Measured from the uploaded scan before model interpretation.</p>
        </div>
        <span className={`rounded-full border px-3 py-1 text-xs font-bold ${qualityTone}`}>
          {qualityLabel}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="panel-muted p-4">
          <p className="section-kicker">Blur Score</p>
          <p className="mt-3 font-display text-3xl font-bold text-[var(--color-ink)]">{blurScore.toFixed(1)}</p>
          <p className="mt-2 text-xs leading-5 text-[var(--color-copy-muted)]">Higher values generally indicate a sharper image.</p>
        </div>
        <div className="panel-muted p-4">
          <p className="section-kicker">Brightness</p>
          <p className="mt-3 font-display text-3xl font-bold text-[var(--color-ink)]">{brightness.toFixed(1)}</p>
          <p className="mt-2 text-xs leading-5 text-[var(--color-copy-muted)]">{brightnessLabel}</p>
        </div>
        <div className="panel-muted p-4">
          <p className="section-kicker">Quality Flag</p>
          <p className="mt-3 font-display text-3xl font-bold text-[var(--color-ink)]">{metrics.is_poor_quality ? 'Review' : 'Pass'}</p>
          <p className="mt-2 text-xs leading-5 text-[var(--color-copy-muted)]">Computed from the real backend quality check.</p>
        </div>
      </div>
    </div>
  );
}

function ModelResultCard({ name, model }) {
  if (!model) {
    return null;
  }

  const isPneumonia = model.label === 'PNEUMONIA';

  return (
    <div className="panel-muted p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-display text-lg font-bold text-[var(--color-ink)]">{name}</p>
        <span className={`h-2.5 w-2.5 rounded-full ${isPneumonia ? 'bg-red-500' : 'bg-emerald-500'}`} />
      </div>
      <p className="mt-3 text-sm font-semibold tracking-[0.18em] text-[var(--color-copy-muted)]">{model.label}</p>
      <p className="mt-2 text-sm text-[var(--color-copy)]">Confidence {(Number(model.confidence ?? 0) * 100).toFixed(1)}%</p>
    </div>
  );
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [currentScan, setCurrentScan] = useState(null);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!currentScan || isLoading) {
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const prediction = await predictPneumonia(currentScan.file);
      setResult(prediction);
    } catch (error) {
      console.error(error);
      alert(error instanceof Error ? error.message : 'Analysis failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setCurrentScan(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 lg:px-8">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute left-[-8rem] top-16 h-64 w-64 rounded-full bg-[rgba(15,118,110,0.12)] blur-3xl" />
        <div className="absolute right-[-5rem] top-0 h-72 w-72 rounded-full bg-[rgba(251,146,60,0.12)] blur-3xl" />
        <div className="absolute bottom-[-8rem] left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-[rgba(15,23,42,0.06)] blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl">
        <div className="mb-6 grid gap-6 lg:grid-cols-[1.35fr_0.85fr]">
          <motion.section
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel-card subtle-grid p-8 sm:p-10"
          >
            <span className="eyebrow-pill">Clinical AI Workspace</span>
            <h1 className="font-display mt-6 max-w-3xl text-4xl font-bold tracking-tight text-[var(--color-ink)] sm:text-5xl">
              PneumoAI Screening Console
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--color-copy)]">
              A cleaner review workspace for dual-model pneumonia screening, Grad-CAM inspection, and report export.
              The stack stays real, lightweight, and ready for local runs, GitHub handoff, and Docker deployment.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              {WORKBENCH_FEATURES.map((item) => (
                <FeaturePill key={item}>{item}</FeaturePill>
              ))}
            </div>
          </motion.section>

          <motion.aside
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
            className="panel-card p-6"
          >
            <p className="section-kicker">Workspace Notes</p>
            <div className="mt-5 space-y-4">
              {PLATFORM_NOTES.map((item) => (
                <ChecklistItem key={item}>{item}</ChecklistItem>
              ))}
            </div>

            <div className="soft-divider my-6" />

            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
              <div className="panel-muted p-4">
                <p className="section-kicker">Models</p>
                <p className="mt-2 font-display text-lg font-bold text-[var(--color-ink)]">DenseNet121 + ResNet50</p>
              </div>
              <div className="panel-muted p-4">
                <p className="section-kicker">Narrative</p>
                <p className="mt-2 font-display text-lg font-bold text-[var(--color-ink)]">Groq explanation flow</p>
              </div>
              <div className="panel-muted p-4">
                <p className="section-kicker">Export</p>
                <p className="mt-2 font-display text-lg font-bold text-[var(--color-ink)]">Portable PDF report</p>
              </div>
            </div>
          </motion.aside>
        </div>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="panel-card relative overflow-hidden p-6 sm:p-8"
        >
          {isLoading && <LoadingSpinner />}

          <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
            <div>
              <p className="section-kicker">Study Intake</p>
              <h2 className="font-display mt-2 text-3xl font-bold tracking-tight text-[var(--color-ink)]">Upload and analyze a chest X-ray</h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--color-copy)]">
                The analysis pipeline keeps the workflow direct: upload the study, run the real backend inference, review the ensemble output, and export the report.
              </p>

              <div className="mt-6">
                <UploadZone currentFile={currentScan} onFileChange={setCurrentScan} />
              </div>

              <div className="mt-6 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={!currentScan || isLoading}
                  className={`rounded-full px-7 py-3 text-sm font-semibold transition-all duration-200 ${
                    currentScan && !isLoading
                      ? 'bg-[var(--color-brand-deep)] text-white shadow-[0_18px_30px_rgba(15,61,60,0.18)] hover:-translate-y-0.5 hover:bg-[var(--color-brand)]'
                      : 'cursor-not-allowed bg-slate-200 text-slate-500'
                  }`}
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Scan'}
                </button>
                {(currentScan || result) && (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="rounded-full border border-[var(--color-border-strong)] bg-white px-6 py-3 text-sm font-semibold text-[var(--color-ink-soft)] transition-colors hover:bg-slate-50"
                  >
                    Clean Start
                  </button>
                )}
              </div>
            </div>

            <div className="panel-muted p-5">
              <p className="section-kicker">Session Guide</p>
              <div className="mt-4 space-y-4">
                {SESSION_GUIDE.map((item) => (
                  <ChecklistItem key={item}>{item}</ChecklistItem>
                ))}
              </div>

              <div className="soft-divider my-6" />

              <div className="grid gap-3">
                <div className="rounded-[20px] border border-[var(--color-border)] bg-white/88 p-4">
                  <p className="section-kicker">Accepted Formats</p>
                  <p className="mt-2 text-sm leading-6 text-[var(--color-copy)]">JPG and PNG up to 20 MB per image.</p>
                </div>
                <div className="rounded-[20px] border border-[var(--color-border)] bg-white/88 p-4">
                  <p className="section-kicker">Review Outputs</p>
                  <p className="mt-2 text-sm leading-6 text-[var(--color-copy)]">Prediction, quality metrics, Grad-CAM views, explanation, and report export.</p>
                </div>
              </div>
            </div>
          </div>

          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="mt-10"
              >
                <div className="soft-divider mb-8" />

                <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="section-kicker">Clinical Output</p>
                    <h2 className="font-display mt-2 text-3xl font-bold tracking-tight text-[var(--color-ink)]">Analysis results</h2>
                    <p className="mt-2 text-sm text-[var(--color-copy)]">Everything below is generated from the backend response for this uploaded study.</p>
                  </div>
                  <StatusBadge status={result.status} />
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className={`rounded-[28px] border p-6 shadow-[0_18px_42px_rgba(15,23,42,0.05)] ${result.prediction === 'PNEUMONIA' ? 'border-red-200 bg-[linear-gradient(135deg,rgba(254,242,242,0.98),rgba(255,255,255,0.96))]' : 'border-emerald-200 bg-[linear-gradient(135deg,rgba(236,253,245,0.98),rgba(255,255,255,0.96))]'}`}>
                    <p className="section-kicker">Final Prediction</p>
                    <p className={`font-display mt-4 text-4xl font-bold tracking-tight ${result.prediction === 'PNEUMONIA' ? 'text-red-700' : 'text-emerald-700'}`}>
                      {result.prediction}
                    </p>
                    <div className="mt-5 h-2 rounded-full bg-white/90">
                      <div
                        className={`h-2 rounded-full ${result.prediction === 'PNEUMONIA' ? 'bg-red-500' : 'bg-emerald-500'}`}
                        style={{ width: `${(Number(result.confidence ?? 0) * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <p className="mt-3 text-sm font-medium text-[var(--color-copy)]">
                      Confidence {(Number(result.confidence ?? 0) * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div className="rounded-[28px] border border-[var(--color-border)] bg-white/92 p-6 shadow-[0_18px_42px_rgba(15,23,42,0.05)]">
                    <p className="section-kicker">Backend Summary</p>
                    <div className="mt-5 grid grid-cols-2 gap-4">
                      <div className="panel-muted p-4">
                        <p className="section-kicker">Severity</p>
                        <p className="mt-2 font-display text-2xl font-bold text-[var(--color-ink)]">{result.severity}</p>
                      </div>
                      <div className="panel-muted p-4">
                        <p className="section-kicker">Focus Score</p>
                        <p className="mt-2 font-display text-2xl font-bold text-[var(--color-ink)]">{Number(result.focus_score ?? 0).toFixed(3)}</p>
                      </div>
                    </div>
                    <p className="mt-4 text-sm leading-7 text-[var(--color-copy)]">{result.explanation}</p>
                  </div>
                </div>

                <div className="mt-6">
                  <QualityMetricsCard metrics={result.quality_metrics} />
                </div>

                <div className="mt-6 rounded-[28px] border border-[var(--color-border)] bg-white/92 p-6 shadow-[0_18px_42px_rgba(15,23,42,0.05)]">
                  <div className="mb-5">
                    <p className="section-kicker">Per-model Outputs</p>
                    <p className="mt-2 text-sm text-[var(--color-copy)]">Direct labels and confidence values returned by the backend ensemble.</p>
                  </div>
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <ModelResultCard name="DenseNet121" model={result.models?.DenseNet || result.models?.densenet} />
                    <ModelResultCard name="ResNet50" model={result.models?.ResNet || result.models?.resnet} />
                  </div>
                </div>

                <div className="mt-6">
                  <GradCamViewer result={result} originalImage={currentScan?.preview} />
                </div>

                <div className="mt-6">
                  <ClinicalReportPanel result={result} originalImage={currentScan?.preview} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>

        <div className="mt-6 text-center text-sm font-medium text-[var(--color-copy-muted)]">
          Built for clinical review workflows. Always verify with a qualified radiologist or physician.
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { getDetailedExplanation } from '../services/api';

function formatPercent(value) {
  return `${(Number(value ?? 0) * 100).toFixed(1)}%`;
}

function formatTimestamp(date) {
  return new Intl.DateTimeFormat('en-IN', {
    dateStyle: 'long',
    timeStyle: 'short',
  }).format(date);
}

function normalizeExplanation(text) {
  return String(text || '')
    .replace(/\r/g, '')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/^[\-\u2022]\s+/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function buildRecommendations(result) {
  const recommendations = [];
  const prediction = result?.prediction;
  const severity = result?.severity;
  const quality = result?.quality_metrics;

  if (prediction === 'PNEUMONIA') {
    recommendations.push('Arrange prompt physician or radiology review and correlate with symptoms, oxygen saturation, temperature, and respiratory rate.');
    recommendations.push('Consider CBC, CRP, and microbiology workup if clinically indicated before definitive treatment decisions.');
    recommendations.push('Assess need for antibiotics, oxygen support, or escalation of care based on the overall clinical picture rather than AI output alone.');
  } else {
    recommendations.push('Correlate the normal AI result with symptoms and physical examination before excluding active disease.');
    recommendations.push('If fever, cough, dyspnea, or hypoxia persist, consider repeat imaging or specialist radiology review.');
  }

  if (severity === 'SEVERE' || severity === 'MODERATE') {
    recommendations.push('Document severity assessment and monitor the patient closely for worsening respiratory symptoms or reduced oxygenation.');
  }

  if (quality?.is_poor_quality) {
    recommendations.push('Because the uploaded image quality is flagged for review, consider repeat radiography if the clinical question remains unresolved.');
  }

  recommendations.push('Use this report strictly as decision support and confirm all findings with a qualified clinician.');
  return recommendations;
}

function summarizeHotspots(result) {
  const denseHotspot = result?.models?.DenseNet?.hotspots?.[0] || result?.models?.densenet?.hotspots?.[0];
  const resHotspot = result?.models?.ResNet?.hotspots?.[0] || result?.models?.resnet?.hotspots?.[0];
  const parts = [];

  if (denseHotspot) {
    parts.push(
      `DenseNet peak attention around x ${denseHotspot.bbox.x}, y ${denseHotspot.bbox.y} with region size ${denseHotspot.bbox.width} by ${denseHotspot.bbox.height}.`
    );
  }

  if (resHotspot) {
    parts.push(
      `ResNet peak attention around x ${resHotspot.bbox.x}, y ${resHotspot.bbox.y} with region size ${resHotspot.bbox.width} by ${resHotspot.bbox.height}.`
    );
  }

  if (parts.length === 0) {
    return 'No hotspot region crossed the current attention threshold.';
  }

  return parts.join(' ');
}

function getModelBundle(result) {
  return [
    { name: 'DenseNet121', model: result?.models?.DenseNet || result?.models?.densenet },
    { name: 'ResNet50', model: result?.models?.ResNet || result?.models?.resnet },
  ].filter((item) => item.model);
}

function getPrimaryHeatmap(result) {
  return getModelBundle(result)
    .filter((item) => item.model?.heatmap)
    .sort((left, right) => Number(right.model?.confidence ?? 0) - Number(left.model?.confidence ?? 0))[0] || null;
}

function inferImageFormat(imageSource) {
  if (typeof imageSource !== 'string') {
    return 'PNG';
  }

  if (imageSource.startsWith('data:image/jpeg') || imageSource.startsWith('data:image/jpg')) {
    return 'JPEG';
  }

  return 'PNG';
}

export default function ClinicalReportPanel({ result, originalImage }) {
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [reportDate] = useState(() => new Date());
  const [detailedExplanation, setDetailedExplanation] = useState('');
  const [isPreparingNarrative, setIsPreparingNarrative] = useState(false);

  const recommendations = buildRecommendations(result);
  const hotspotSummary = summarizeHotspots(result);
  const finalExplanation = detailedExplanation || result?.explanation || 'No detailed explanation is available for this report.';
  const quality = result?.quality_metrics || {};
  const modelBundle = getModelBundle(result);

  useEffect(() => {
    let isCancelled = false;

    async function fetchExplanation() {
      if (!result) {
        return;
      }

      setIsPreparingNarrative(true);
      setDetailedExplanation('');

      try {
        const text = await getDetailedExplanation(
          result?.prediction || 'NORMAL',
          result?.confidence || 0,
          result?.severity || 'NONE',
          'English'
        );

        if (!isCancelled) {
          setDetailedExplanation(normalizeExplanation(text));
        }
      } catch (error) {
        console.error(error);
        if (!isCancelled) {
          setDetailedExplanation('');
        }
      } finally {
        if (!isCancelled) {
          setIsPreparingNarrative(false);
        }
      }
    }

    fetchExplanation();

    return () => {
      isCancelled = true;
    };
  }, [result]);

  const handleDownloadPdf = async () => {
    if (isGeneratingPdf || isPreparingNarrative) {
      return;
    }

    setIsGeneratingPdf(true);

    try {
      const { jsPDF } = await import('jspdf');
      const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      const fileStamp = reportDate.toISOString().replace(/[:.]/g, '-');
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 12;
      const contentWidth = pageWidth - (margin * 2);
      const headingColor = [15, 23, 42];
      const bodyColor = [51, 65, 85];
      let y = margin;

      const ensureSpace = (heightNeeded) => {
        if (y + heightNeeded <= pageHeight - margin) {
          return;
        }

        doc.addPage();
        y = margin;
      };

      const drawParagraph = (text, options = {}) => {
        const value = String(text || '').trim();

        if (!value) {
          return;
        }

        const maxWidth = options.maxWidth ?? contentWidth;
        const fontSize = options.fontSize ?? 11;
        const lineHeight = options.lineHeight ?? 5.5;
        const gap = options.gap ?? 4;
        const x = options.x ?? margin;
        const lines = doc.splitTextToSize(value, maxWidth);

        ensureSpace((lines.length * lineHeight) + gap);
        doc.setFont('helvetica', options.fontStyle ?? 'normal');
        doc.setFontSize(fontSize);
        doc.setTextColor(...(options.color ?? bodyColor));
        doc.text(lines, x, y);
        y += (lines.length * lineHeight) + gap;
      };

      const drawSectionTitle = (title) => {
        ensureSpace(10);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(12);
        doc.setTextColor(...headingColor);
        doc.text(title, margin, y);
        y += 6;
      };

      const drawMetricCard = (x, top, width, title, value) => {
        doc.setDrawColor(226, 232, 240);
        doc.setFillColor(248, 250, 252);
        doc.roundedRect(x, top, width, 18, 3, 3, 'FD');
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(8);
        doc.setTextColor(100, 116, 139);
        doc.text(title.toUpperCase(), x + 4, top + 5);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(...headingColor);
        doc.text(String(value || 'N/A'), x + 4, top + 13);
      };

      doc.setFillColor(15, 23, 42);
      doc.rect(0, 0, pageWidth, 42, 'F');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(11);
      doc.setTextColor(191, 219, 254);
      doc.text('PNEUMOAI CLINICAL REPORT', margin, 12);
      doc.setFontSize(24);
      doc.setTextColor(255, 255, 255);
      doc.text(String(result?.prediction || 'UNKNOWN'), margin, 25);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(11);
      doc.text('AI-assisted chest X-ray interpretation for clinical review.', margin, 33);

      doc.setFillColor(30, 41, 59);
      doc.roundedRect(pageWidth - 64, 10, 52, 22, 3, 3, 'F');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8);
      doc.setTextColor(191, 219, 254);
      doc.text('GENERATED', pageWidth - 59, 16);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);
      doc.setTextColor(255, 255, 255);
      doc.text(formatTimestamp(reportDate), pageWidth - 59, 22);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8);
      doc.setTextColor(191, 219, 254);
      doc.text('STATUS', pageWidth - 59, 28);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      doc.setTextColor(255, 255, 255);
      doc.text(String(result?.status || 'N/A'), pageWidth - 59, 34);

      y = 52;

      const metricGap = 6;
      const metricWidth = (contentWidth - metricGap) / 2;

      drawMetricCard(margin, y, metricWidth, 'Final confidence', formatPercent(result?.confidence));
      drawMetricCard(margin + metricWidth + metricGap, y, metricWidth, 'Severity', result?.severity);
      y += 22;
      drawMetricCard(margin, y, metricWidth, 'Focus score', Number(result?.focus_score ?? 0).toFixed(3));
      drawMetricCard(margin + metricWidth + metricGap, y, metricWidth, 'Quality flag', quality.is_poor_quality ? 'Review' : 'Pass');
      y += 26;

      drawSectionTitle('Clinical Impression');
      drawParagraph(
        `The AI ensemble returned a final classification of ${result?.prediction} with an overall confidence of ${formatPercent(result?.confidence)}. ` +
        `The status is ${result?.status} and the derived severity grade is ${result?.severity}. ${result?.explanation}`
      );
      drawParagraph(
        `Image quality review shows a blur score of ${Number(quality.blur_score ?? 0).toFixed(1)} and brightness of ${Number(quality.brightness ?? 0).toFixed(1)}. ` +
        `These findings should be interpreted alongside symptoms, examination, and formal radiology review before clinical action is taken.`
      );

      const primaryHeatmap = getPrimaryHeatmap(result);
      const imagePanels = [
        originalImage
          ? { title: 'Original X-ray', image: originalImage, format: inferImageFormat(originalImage) }
          : null,
        primaryHeatmap?.model?.heatmap
          ? {
              title: `${primaryHeatmap.name} Heatmap`,
              image: `data:image/png;base64,${primaryHeatmap.model.heatmap}`,
              format: 'PNG',
            }
          : null,
      ].filter(Boolean);

      if (imagePanels.length > 0) {
        ensureSpace(86);
        drawSectionTitle('Study Visuals');

        const panelGap = 6;
        const panelWidth = imagePanels.length === 1 ? contentWidth : (contentWidth - panelGap) / 2;
        const panelHeight = 72;
        const panelTop = y;

        imagePanels.forEach((panel, index) => {
          const x = margin + (index * (panelWidth + panelGap));

          doc.setDrawColor(226, 232, 240);
          doc.setFillColor(255, 255, 255);
          doc.roundedRect(x, panelTop, panelWidth, panelHeight, 3, 3, 'FD');
          doc.setFont('helvetica', 'bold');
          doc.setFontSize(9);
          doc.setTextColor(...headingColor);
          doc.text(panel.title, x + 4, panelTop + 6);

          try {
            const imageProps = doc.getImageProperties(panel.image);
            const maxWidth = panelWidth - 8;
            const maxHeight = panelHeight - 14;
            const scale = Math.min(maxWidth / imageProps.width, maxHeight / imageProps.height);
            const renderWidth = imageProps.width * scale;
            const renderHeight = imageProps.height * scale;
            const imageX = x + ((panelWidth - renderWidth) / 2);
            const imageY = panelTop + 10 + ((maxHeight - renderHeight) / 2);

            doc.addImage(panel.image, panel.format, imageX, imageY, renderWidth, renderHeight);
          } catch (imageError) {
            console.error(imageError);
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(10);
            doc.setTextColor(...bodyColor);
            doc.text('Image preview unavailable in exported report.', x + 4, panelTop + 18);
          }
        });

        y = panelTop + panelHeight + 8;
      }

      drawSectionTitle('Model Outputs');
      modelBundle.forEach((item) => {
        drawParagraph(
          `${item.name}: ${item.model?.label || 'N/A'} with confidence ${formatPercent(item.model?.confidence)}.`,
          { gap: 2 }
        );
      });
      y += 2;

      drawSectionTitle('Hotspot Summary');
      drawParagraph(hotspotSummary);

      drawSectionTitle('Clinical Recommendations');
      recommendations.forEach((item, index) => {
        drawParagraph(`${index + 1}. ${item}`, { gap: 2 });
      });
      y += 2;

      drawSectionTitle('Detailed Clinical Explanation');
      finalExplanation
        .split(/\n{2,}/)
        .map((paragraph) => paragraph.trim())
        .filter(Boolean)
        .forEach((paragraph) => {
          drawParagraph(paragraph);
        });

      ensureSpace(22);
      doc.setDrawColor(253, 230, 138);
      doc.setFillColor(254, 252, 232);
      doc.roundedRect(margin, y, contentWidth, 18, 3, 3, 'FD');
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9.5);
      doc.setTextColor(120, 53, 15);
      doc.text(
        doc.splitTextToSize(
          'This report is generated by an AI system for clinical decision support. It must not replace physician judgment, formal radiology reporting, or direct patient assessment.',
          contentWidth - 8
        ),
        margin + 4,
        y + 6
      );

      doc.save(`PneumoAI_Clinical_Report_${fileStamp}.pdf`);
    } catch (error) {
      console.error(error);
      alert(`PDF generation failed. ${error instanceof Error ? error.message : 'Please try again.'}`);
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
      className="rounded-[28px] border border-[var(--color-border)] bg-white/92 p-6 shadow-[0_18px_42px_rgba(15,23,42,0.05)]"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="section-kicker">Clinical Report</p>
          <h3 className="font-display mt-2 text-2xl font-bold tracking-tight text-[var(--color-ink)]">Export-ready documentation</h3>
          <p className="mt-2 text-sm text-[var(--color-copy)]">Formatted summary for documentation, review, and PDF export.</p>
        </div>
        <button
          type="button"
          onClick={handleDownloadPdf}
          disabled={isGeneratingPdf || isPreparingNarrative}
          className="rounded-full bg-[var(--color-brand-deep)] px-5 py-3 text-sm font-semibold text-white transition-all hover:-translate-y-0.5 hover:bg-[var(--color-brand)] disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {isPreparingNarrative ? 'Preparing Report...' : isGeneratingPdf ? 'Generating PDF...' : 'Download Report'}
        </button>
      </div>
    </motion.div>
  );
}

import axios from 'axios';

const API_BASE = (import.meta.env.VITE_API_URL || '').trim().replace(/\/$/, '');

function buildApiUrl(path) {
  return API_BASE ? `${API_BASE}${path}` : path;
}

function normalizeModelResult(model = {}) {
  return {
    label: model.label,
    prediction: model.label,
    confidence: model.confidence,
    heatmap: model.heatmap,
    heatmap_mask: model.heatmap_mask,
    original: model.original,
    hotspots: model.hotspots || [],
  };
}

function getApiErrorMessage(error, fallbackMessage) {
  if (axios.isAxiosError(error)) {
    return (
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      fallbackMessage
    );
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallbackMessage;
}

export async function predictPneumonia(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await axios.post(buildApiUrl('/predict'), formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    });

    const data = response.data;
    const normalizedDenseNet = normalizeModelResult(data?.models?.DenseNet);
    const normalizedResNet = normalizeModelResult(data?.models?.ResNet);

    return {
      prediction: data.final_prediction,
      confidence: data.final_confidence,
      status: data.status,
      focus_score: data.focus_score,
      severity: data.severity,
      explanation: data.explanation,
      quality_metrics: data.quality_metrics,
      models: {
        DenseNet: normalizedDenseNet,
        ResNet: normalizedResNet,
        densenet: normalizedDenseNet,
        resnet: normalizedResNet,
      },
      metrics: data.quality_metrics,
    };
  } catch (error) {
    throw new Error(
      getApiErrorMessage(error, 'Prediction request failed. Please check that the backend is running.')
    );
  }
}

export async function getDetailedExplanation(diagnosis, confidence, severity, language = 'English') {
  try {
    const response = await axios.post(buildApiUrl('/explain'), {
      diagnosis,
      confidence,
      severity,
      language,
    });
    return response.data.explanation;
  } catch (error) {
    console.error('Explanation fetch error:', error);
    return getApiErrorMessage(
      error,
      'Failed to fetch detailed explanation. Please check backend configuration.'
    );
  }
}

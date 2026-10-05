const GRADIENT_COLORS = [
  'linear-gradient(90deg, #22c55e, #06b6d4)',
  'linear-gradient(90deg, #06b6d4, #8b5cf6)',
  'linear-gradient(90deg, #8b5cf6, #ec4899)',
  'linear-gradient(90deg, #f59e0b, #ef4444)',
  'linear-gradient(90deg, #64748b, #475569)',
];

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { prediction, top_5, model_used, remedy } = result;

  const MODEL_DISPLAY = {
    custom_cnn: 'Custom CNN',
    mobilenetv2: 'MobileNetV2',
    resnet50: 'ResNet50',
  };

  return (
    <div className="result-section" id="prediction-result">
      <h2 className="section-title">
        <span>🔬</span> Analysis Result
      </h2>

      <div className="result-card">
        {/* Header */}
        <div className="result-header">
          <div className="result-status">
            <div className={`result-status-icon ${prediction.is_healthy ? 'healthy' : 'diseased'}`}>
              {prediction.is_healthy ? '✅' : '⚠️'}
            </div>
            <span className={`result-status-label ${prediction.is_healthy ? 'healthy' : 'diseased'}`}>
              {prediction.is_healthy ? 'Healthy Plant' : 'Disease Detected'}
            </span>
          </div>
          <span className="result-model-tag">
            {MODEL_DISPLAY[model_used] || model_used}
          </span>
        </div>

        {/* Body */}
        <div className="result-body">
          <div className="result-main">
            {/* Left — Info */}
            <div className="result-info">
              <div className="result-info-item">
                <span className="result-info-label">Plant</span>
                <span className="result-info-value">{prediction.plant}</span>
              </div>
              <div className="result-info-item">
                <span className="result-info-label">Condition</span>
                <span className="result-info-value">{prediction.condition}</span>
              </div>
            </div>

            {/* Right — Confidence */}
            <div className="result-info">
              <div className="result-info-item">
                <span className="result-info-label">Confidence</span>
                <span className="result-info-value" style={{ color: 'var(--primary-light)' }}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="confidence-bar-container">
                <div className="confidence-bar">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Top-5 Predictions */}
          {top_5 && top_5.length > 1 && (
            <div className="top5-section">
              <div className="top5-title">Top 5 Predictions</div>
              <div className="top5-list">
                {top_5.map((pred, idx) => (
                  <div className="top5-item" key={idx}>
                    <span className="top5-rank">#{idx + 1}</span>
                    <span className="top5-name">
                      {pred.plant} — {pred.condition}
                    </span>
                    <div className="top5-bar-container">
                      <div
                        className="top5-bar-fill"
                        style={{
                          width: `${pred.confidence * 100}%`,
                          background: GRADIENT_COLORS[idx] || GRADIENT_COLORS[4],
                        }}
                      />
                    </div>
                    <span className="top5-confidence">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Agronomic Recommendations & Remedy */}
          {remedy && (
            <div className="remedy-section" style={{ marginTop: '1.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
              <div className="top5-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: 'var(--primary-light)' }}>
                <span>🌱</span> Treatment & Care Recommendations
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontWeight: '600', color: '#f59e0b', marginBottom: '0.35rem', fontSize: '0.9rem' }}>⚠️ Cause & Symptoms</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>{remedy.cause}</div>
                </div>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontWeight: '600', color: '#06b6d4', marginBottom: '0.35rem', fontSize: '0.9rem' }}>💊 Recommended Treatment</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>{remedy.treatment}</div>
                </div>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontWeight: '600', color: '#22c55e', marginBottom: '0.35rem', fontSize: '0.9rem' }}>🛡️ Prevention Strategy</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>{remedy.prevention}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const GRADIENT_COLORS = [
  'linear-gradient(90deg, #22c55e, #10b981)',
  'linear-gradient(90deg, #06b6d4, #3b82f6)',
  'linear-gradient(90deg, #8b5cf6, #ec4899)',
  'linear-gradient(90deg, #f59e0b, #ef4444)',
  'linear-gradient(90deg, #64748b, #475569)',
];

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { prediction, top_5, model_used, remedy, inference_ms } = result;

  const MODEL_DISPLAY = {
    custom_cnn: 'Custom CNN (Baseline)',
    mobilenetv2: 'MobileNetV2 (Recommended)',
    resnet50: 'ResNet50 (Residual Network)',
  };

  const isHealthy = prediction.is_healthy;

  return (
    <div className="result-container-card" id="prediction-result">
      {/* Result Header Banner */}
      <div className={`diagnosis-banner ${isHealthy ? 'banner-healthy' : 'banner-diseased'}`}>
        <div className="banner-left">
          <div className="banner-status-icon">
            {isHealthy ? '🌿' : '⚠️'}
          </div>
          <div>
            <span className="banner-sub-tag">PATHOLOGY DIAGNOSIS</span>
            <h2 className="banner-title">
              {isHealthy ? 'Healthy Plant Foliage' : `${prediction.condition} Detected`}
            </h2>
          </div>
        </div>

        <div className="banner-right">
          <div className="model-chip">
            <span className="model-chip-dot" />
            <span>{MODEL_DISPLAY[model_used] || model_used}</span>
          </div>
          {inference_ms && (
            <span className="latency-chip">⚡ {inference_ms}ms</span>
          )}
        </div>
      </div>

      <div className="result-card-content">
        {/* Main Grid: Specimen Overview & Confidence */}
        <div className="diagnosis-grid">
          <div className="specimen-meta-panel">
            <div className="meta-row">
              <span className="meta-label">IDENTIFIED HOST PLANT</span>
              <span className="meta-value plant-name">{prediction.plant}</span>
            </div>

            <div className="meta-row">
              <span className="meta-label">PATHOLOGICAL STATE</span>
              <span className={`meta-value condition-name ${isHealthy ? 'text-healthy' : 'text-diseased'}`}>
                {prediction.condition}
              </span>
            </div>

            <div className="meta-row">
              <span className="meta-label">CLINICAL STATUS</span>
              <span className="meta-value">
                <span className={`health-status-pill ${isHealthy ? 'pill-healthy' : 'pill-diseased'}`}>
                  {isHealthy ? '✓ Pathogen Free' : '● Active Infection'}
                </span>
              </span>
            </div>
          </div>

          <div className="confidence-meter-panel">
            <div className="confidence-dial-header">
              <span className="meter-label">AI CONFIDENCE SCORE</span>
              <span className="meter-percentage">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="confidence-track">
              <div
                className={`confidence-bar-fill ${isHealthy ? 'fill-healthy' : 'fill-diseased'}`}
                style={{ width: `${Math.max(prediction.confidence * 100, 4)}%` }}
              />
            </div>
            <div className="confidence-footnote">
              {prediction.confidence > 0.8
                ? 'High diagnostic confidence based on deep feature alignment.'
                : 'Moderate confidence. Verify leaf symptoms with recommendations below.'}
            </div>
          </div>
        </div>

        {/* Top 5 Predictions Breakdown */}
        {top_5 && top_5.length > 1 && (
          <div className="top5-card-section">
            <h3 className="section-mini-heading">
              <span>📊</span>
              <span>Top-5 Class Probability Distribution</span>
            </h3>

            <div className="top5-table">
              {top_5.map((pred, idx) => (
                <div className="top5-row" key={idx}>
                  <div className="top5-rank-col">#{idx + 1}</div>
                  <div className="top5-name-col">
                    <span className="pred-crop">{pred.plant}</span>
                    <span className="pred-divider">—</span>
                    <span className={pred.is_healthy ? 'pred-healthy' : 'pred-condition'}>
                      {pred.condition}
                    </span>
                  </div>
                  <div className="top5-bar-col">
                    <div className="mini-bar-track">
                      <div
                        className="mini-bar-fill"
                        style={{
                          width: `${pred.confidence * 100}%`,
                          background: GRADIENT_COLORS[idx] || GRADIENT_COLORS[4],
                        }}
                      />
                    </div>
                  </div>
                  <div className="top5-pct-col">
                    {(pred.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Agronomic Recommendations Section */}
        {remedy && (
          <div className="remedy-card-section">
            <div className="remedy-heading-row">
              <span className="remedy-badge-icon">🌱</span>
              <div>
                <h3 className="remedy-main-title">Agronomic Management & Remedy Protocol</h3>
                <p className="remedy-main-sub">Actionable treatment recommendations and preventative cultural practices</p>
              </div>
            </div>

            <div className="remedy-tri-grid">
              <div className="remedy-box box-cause">
                <div className="remedy-box-header">
                  <span className="box-icon">⚠️</span>
                  <span className="box-title">Etiology & Symptoms</span>
                </div>
                <p className="box-description">{remedy.cause}</p>
              </div>

              <div className="remedy-box box-treatment">
                <div className="remedy-box-header">
                  <span className="box-icon">💊</span>
                  <span className="box-title">Active Treatment</span>
                </div>
                <p className="box-description">{remedy.treatment}</p>
              </div>

              <div className="remedy-box box-prevention">
                <div className="remedy-box-header">
                  <span className="box-icon">🛡️</span>
                  <span className="box-title">Cultural Prevention</span>
                </div>
                <p className="box-description">{remedy.prevention}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import { useState, useEffect } from 'react';
import { API_URL } from '../config';

const MODEL_CONFIGS = {
  mobilenetv2: {
    badge: '⭐ 95.2% ACCURACY — RECOMMENDED',
    badgeClass: 'badge-recommended',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/>
        <line x1="12" y1="18" x2="12.01" y2="18"/>
      </svg>
    ),
    subtitle: 'MobileNetV2 Transfer Learning',
    speed: '~45ms inference'
  },
  resnet50: {
    badge: '🏗️ 84.2% ACCURACY — DEEP RESIDUAL',
    badgeClass: 'badge-deep',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/>
        <polyline points="2 12 12 17 22 12"/>
      </svg>
    ),
    subtitle: 'ResNet-50 50-Layer Backbone',
    speed: '~120ms inference'
  },
  custom_cnn: {
    badge: '🧠 4-BLOCK BASELINE MODEL',
    badgeClass: 'badge-scratch',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
      </svg>
    ),
    subtitle: 'Custom ConvNet Architecture',
    speed: '~18ms inference'
  }
};

export default function ModelSelector({ selectedModel, onSelect }) {
  const [models, setModels] = useState(getDefaultModels());
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models`);
      if (res.ok) {
        const data = await res.json();
        if (data.models && data.models.length > 0) {
          setModels(data.models);
        }
      }
    } catch {
      // Fallback kept
    }
  };

  function getDefaultModels() {
    return [
      {
        name: 'mobilenetv2',
        display_name: 'MobileNetV2',
        description: 'Fine-tuned inverted residual bottleneck model optimized for ultra-accurate leaf pathology diagnosis.',
        available: true,
        size_mb: 18.34,
        accuracy: 0.952,
      },
      {
        name: 'resnet50',
        display_name: 'ResNet50',
        description: 'Deep 50-layer convolutional residual network featuring identity skip connections for robust feature extraction.',
        available: true,
        size_mb: 96.74,
        accuracy: 0.842,
      },
      {
        name: 'custom_cnn',
        display_name: 'Custom CNN',
        description: '4-block convolutional network with pooling and dropout trained from scratch on PlantVillage dataset.',
        available: true,
        size_mb: 5.37,
        accuracy: 0.070,
      },
    ];
  }

  return (
    <div className="model-selector-section">
      <div className="section-header-row">
        <div>
          <h2 className="section-title">
            <span className="section-title-icon">⚡</span>
            <span>Neural Architecture Engine</span>
          </h2>
          <p className="section-subtitle">Select which trained deep learning model will analyze your crop specimen</p>
        </div>
        <div className="models-count-badge">3 Models Ready</div>
      </div>

      <div className="model-cards-grid">
        {models.map((model) => {
          const cfg = MODEL_CONFIGS[model.name] || MODEL_CONFIGS.mobilenetv2;
          const isSelected = selectedModel === model.name;

          return (
            <div
              key={model.name}
              className={`model-card-premium ${isSelected ? 'is-selected' : ''}`}
              onClick={() => onSelect(model.name)}
              id={`model-${model.name}`}
            >
              {isSelected && <div className="selected-glow-aura" />}

              <div className="card-top-bar">
                <span className={`model-pill-badge ${cfg.badgeClass}`}>
                  {cfg.badge}
                </span>
                <div className="radio-indicator">
                  <div className="radio-inner" />
                </div>
              </div>

              <div className="card-identity">
                <div className="card-icon-container">
                  {cfg.icon}
                </div>
                <div>
                  <h3 className="card-model-title">{model.display_name}</h3>
                  <div className="card-model-subtitle">{cfg.subtitle}</div>
                </div>
              </div>

              <p className="card-model-description">{model.description}</p>

              <div className="card-stats-row">
                <div className="stat-box">
                  <div className="stat-label">ACCURACY</div>
                  <div className="stat-val highlight">
                    {model.accuracy != null ? `${(model.accuracy * 100).toFixed(1)}%` : '95.2%'}
                  </div>
                </div>
                <div className="stat-box">
                  <div className="stat-label">WEIGHTS</div>
                  <div className="stat-val">
                    {model.size_mb != null ? `${model.size_mb} MB` : '18.3 MB'}
                  </div>
                </div>
                <div className="stat-box">
                  <div className="stat-label">SPEED</div>
                  <div className="stat-val speed-val">{cfg.speed}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

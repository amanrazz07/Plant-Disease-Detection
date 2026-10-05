import { useState, useEffect } from 'react';

import { API_URL } from '../config';

const MODEL_ICONS = {
  custom_cnn: '🧠',
  mobilenetv2: '📱',
  resnet50: '🏗️',
};

export default function ModelSelector({ selectedModel, onSelect }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(data.models);
      } else {
        // Fallback when API is unavailable
        setModels(getDefaultModels());
      }
    } catch {
      setModels(getDefaultModels());
    } finally {
      setLoading(false);
    }
  };

  const getDefaultModels = () => [
    {
      name: 'custom_cnn',
      display_name: 'Custom CNN',
      description: '4-block CNN trained from scratch on PlantVillage (~2M params)',
      available: false,
      size_mb: null,
      accuracy: null,
    },
    {
      name: 'mobilenetv2',
      display_name: 'MobileNetV2',
      description: 'MobileNetV2 with transfer learning from ImageNet (~3.5M fine-tuned params)',
      available: false,
      size_mb: null,
      accuracy: null,
    },
    {
      name: 'resnet50',
      display_name: 'ResNet50',
      description: 'ResNet50 with transfer learning from ImageNet (~25.6M fine-tuned params)',
      available: false,
      size_mb: null,
      accuracy: null,
    },
  ];

  if (loading) return null;

  return (
    <div className="model-selector">
      <h2 className="section-title">
        <span>🤖</span> Select Model
      </h2>
      <div className="model-cards">
        {models.map((model) => (
          <div
            key={model.name}
            className={`model-card ${selectedModel === model.name ? 'selected' : ''} ${
              !model.available ? 'unavailable' : ''
            }`}
            onClick={() => model.available && onSelect(model.name)}
            id={`model-${model.name}`}
          >
            <div className="model-card-header">
              <div className="model-card-name">
                {MODEL_ICONS[model.name] || '🤖'} {model.display_name}
              </div>
              <div className="model-card-check">
                {selectedModel === model.name ? '✓' : ''}
              </div>
            </div>
            <div className="model-card-desc">{model.description}</div>
            <div className="model-card-meta">
              {model.accuracy != null && (
                <div className="model-meta-item">
                  📊 Accuracy:{' '}
                  <span className="model-meta-value">
                    {(model.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {model.size_mb != null && (
                <div className="model-meta-item">
                  💾 Size:{' '}
                  <span className="model-meta-value">{model.size_mb} MB</span>
                </div>
              )}
              {!model.available && (
                <div className="model-meta-item" style={{ color: 'var(--warning)' }}>
                  ⚠️ Not trained yet
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

import { useState, useEffect } from 'react';

import { API_URL } from '../config';

export default function ModelComparison() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    fetchComparison();
  }, []);

  const fetchComparison = async () => {
    try {
      const res = await fetch(`${API_URL}/comparison`);
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.comparison.length > 0) {
          setData(json.comparison);
        } else {
          setData(getFallbackData());
        }
      } else {
        setData(getFallbackData());
      }
    } catch {
      setData(getFallbackData());
    } finally {
      setLoading(false);
    }
  };

  const getFallbackData = () => [
    {
      model: 'MobileNetV2',
      accuracy: 0.952,
      precision: 0.951,
      recall: 0.949,
      f1_score: 0.95,
      inference_ms: 45.2,
      size_mb: 18.34,
      parameters: '2,707,238',
      badge: 'Best Accuracy & Efficiency',
      badgeColor: '#22c55e',
    },
    {
      model: 'ResNet50',
      accuracy: 0.842,
      precision: 0.838,
      recall: 0.835,
      f1_score: 0.836,
      inference_ms: 120.4,
      size_mb: 96.74,
      parameters: '24,122,022',
      badge: 'Deep Architecture',
      badgeColor: '#06b6d4',
    },
    {
      model: 'Custom CNN',
      accuracy: 0.0702,
      precision: 0.0266,
      recall: 0.0702,
      f1_score: 0.027,
      inference_ms: 18.1,
      size_mb: 5.37,
      parameters: '463,974',
      badge: 'Scratch Baseline',
      badgeColor: '#8b5cf6',
    },
  ];

  return (
    <div className="model-comparison-container" style={{ margin: '2rem 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 className="section-title" style={{ margin: 0 }}>
          <span>📊</span> Model Benchmarks & Comparison
        </h2>
        <button
          onClick={() => setIsOpen(!isOpen)}
          style={{
            background: 'var(--card-bg)',
            border: '1px solid var(--border-color)',
            color: 'var(--text-primary)',
            padding: '0.45rem 0.9rem',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            transition: 'all 0.2s',
          }}
        >
          {isOpen ? 'Hide Comparison ▲' : 'Show Comparison Table ▼'}
        </button>
      </div>

      {isOpen && (
        <div
          style={{
            background: 'var(--card-bg)',
            border: '1px solid var(--border-color)',
            borderRadius: '16px',
            padding: '1.5rem',
            overflowX: 'auto',
            backdropFilter: 'blur(10px)',
          }}
        >
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
            Comparative evaluation across all three architectures evaluated on the PlantVillage dataset (38 disease classes).
          </p>

          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              textAlign: 'left',
              fontSize: '0.9rem',
            }}
          >
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Model</th>
                <th style={{ padding: '0.75rem 1rem' }}>Accuracy</th>
                <th style={{ padding: '0.75rem 1rem' }}>Precision</th>
                <th style={{ padding: '0.75rem 1rem' }}>Recall</th>
                <th style={{ padding: '0.75rem 1rem' }}>F1-Score</th>
                <th style={{ padding: '0.75rem 1rem' }}>Size</th>
                <th style={{ padding: '0.75rem 1rem' }}>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {data.map((item, idx) => {
                const isBest = item.model.includes('MobileNet');
                return (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: '1px solid rgba(255,255,255,0.06)',
                      background: isBest ? 'rgba(34, 197, 94, 0.05)' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '0.9rem 1rem', fontWeight: '600' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>{item.model}</span>
                        {isBest && (
                          <span
                            style={{
                              background: 'rgba(34, 197, 94, 0.2)',
                              color: '#22c55e',
                              fontSize: '0.7rem',
                              padding: '0.2rem 0.5rem',
                              borderRadius: '12px',
                              fontWeight: '600',
                            }}
                          >
                            Top Performer
                          </span>
                        )}
                      </div>
                    </td>
                    <td style={{ padding: '0.9rem 1rem', color: isBest ? '#22c55e' : 'inherit', fontWeight: 'bold' }}>
                      {(item.accuracy * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '0.9rem 1rem' }}>{(item.precision * 100).toFixed(1)}%</td>
                    <td style={{ padding: '0.9rem 1rem' }}>{(item.recall * 100).toFixed(1)}%</td>
                    <td style={{ padding: '0.9rem 1rem' }}>{(item.f1_score * 100).toFixed(1)}%</td>
                    <td style={{ padding: '0.9rem 1rem' }}>{item.size_mb} MB</td>
                    <td style={{ padding: '0.9rem 1rem', color: 'var(--text-secondary)' }}>{item.parameters}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div
            style={{
              marginTop: '1.25rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '1rem',
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
            }}
          >
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.9rem', borderRadius: '10px' }}>
              <strong style={{ color: 'var(--text-primary)' }}>📱 MobileNetV2 (Recommended):</strong>
              <div style={{ marginTop: '0.25rem' }}>
                Achieves highest accuracy (95.2%) with lightweight 18.3 MB size. Highly suitable for fast real-time inference on mobile or edge devices.
              </div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.9rem', borderRadius: '10px' }}>
              <strong style={{ color: 'var(--text-primary)' }}>🏗️ ResNet50:</strong>
              <div style={{ marginTop: '0.25rem' }}>
                Deep 50-layer residual network achieving 84.2% accuracy using transfer learning, with 24.1M parameters.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { useState } from 'react';
import { API_URL } from '../config';

const SAMPLE_LEAVES = [
  {
    name: 'Apple Healthy',
    badge: 'Healthy Leaf',
    badgeColor: '#22c55e',
    path: '/samples/apple_healthy.jpg',
  },
  {
    name: 'Apple Black Rot',
    badge: 'Fungal Infection',
    badgeColor: '#f59e0b',
    path: '/samples/apple_black_rot.jpg',
  },
  {
    name: 'Tomato Early Blight',
    badge: 'Severe Foliar Blight',
    badgeColor: '#ef4444',
    path: '/samples/tomato_early_blight.jpg',
  },
  {
    name: 'Potato Late Blight',
    badge: 'Water Mold Disease',
    badgeColor: '#f43f5e',
    path: '/samples/potato_late_blight.jpg',
  },
];

export default function ImageUploader({ selectedModel, onResult, onError, onLoading }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [isScanning, setIsScanning] = useState(false);

  const handleFile = (file) => {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      onError('Please upload a valid image file (JPEG, PNG, WebP, or BMP).');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      onError('Image size must be under 10 MB.');
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    onError(null);
    onResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const handleRemove = (e) => {
    e && e.stopPropagation();
    setImage(null);
    setPreview(null);
    onResult(null);
    onError(null);
  };

  const handleLoadSample = async (samplePath) => {
    try {
      const res = await fetch(samplePath);
      const blob = await res.blob();
      const file = new File([blob], samplePath.split('/').pop(), { type: 'image/jpeg' });
      handleFile(file);
    } catch {
      onError('Could not load sample leaf.');
    }
  };

  const handlePredict = async (e) => {
    e && e.stopPropagation();
    if (!image) return;

    setIsScanning(true);
    onLoading(true);
    onError(null);
    onResult(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const res = await fetch(`${API_URL}/predict?model=${selectedModel}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      onResult(data);
    } catch (err) {
      onError(err.message || 'Failed to connect to the API. Is the backend running?');
    } finally {
      setIsScanning(false);
      onLoading(false);
    }
  };

  return (
    <div className="upload-container-card">
      <div className="section-header-row">
        <div>
          <h2 className="section-title">
            <span className="section-title-icon">🔬</span>
            <span>Diagnostic Scanner</span>
          </h2>
          <p className="section-subtitle">Drop a high-resolution leaf photograph or choose a pre-loaded sample below</p>
        </div>
      </div>

      <div
        className={`scanner-dropzone ${dragOver ? 'is-dragover' : ''} ${preview ? 'has-preview' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !preview && document.getElementById('file-input').click()}
        id="upload-zone"
      >
        <div className="corner-bracket top-left" />
        <div className="corner-bracket top-right" />
        <div className="corner-bracket bottom-left" />
        <div className="corner-bracket bottom-right" />

        {isScanning && <div className="scanning-laser-beam" />}

        {!preview ? (
          <div className="dropzone-empty-state">
            <div className="upload-orbit-icon">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <h3 className="dropzone-headline">Drag & Drop Plant Leaf Image Here</h3>
            <p className="dropzone-sub">
              or <span className="browse-link">browse files</span> from your device
            </p>
            <div className="allowed-badges">
              <span>JPG</span>
              <span>PNG</span>
              <span>WEBP</span>
              <span>BMP</span>
              <span className="max-size-pill">MAX 10MB</span>
            </div>
          </div>
        ) : (
          <div className="preview-stage">
            <div className="preview-media-wrapper">
              <img src={preview} alt="Plant specimen preview" className="specimen-image" />
              <div className="specimen-overlay-tag">SPECIMEN LOADED</div>
            </div>

            <div className="specimen-controls">
              <button
                className="btn-action-primary"
                onClick={handlePredict}
                id="predict-btn"
                disabled={isScanning}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="11" cy="11" r="8"/>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                <span>{isScanning ? 'Diagnosing Specimen…' : 'Run Pathology Diagnosis'}</span>
              </button>
              <button className="btn-action-secondary" onClick={handleRemove} id="remove-btn">
                ✕ Clear Specimen
              </button>
            </div>
          </div>
        )}

        <input
          id="file-input"
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={handleInputChange}
          style={{ display: 'none' }}
        />
      </div>

      {/* Quick Test Samples */}
      <div className="quick-samples-bar">
        <div className="samples-heading">
          <span className="sample-pulse" />
          <span>Quick Test with Preloaded Benchmark Specimens:</span>
        </div>
        <div className="sample-chips-row">
          {SAMPLE_LEAVES.map((sample, idx) => (
            <button
              key={idx}
              className="sample-chip"
              onClick={() => handleLoadSample(sample.path)}
            >
              <img src={sample.path} alt={sample.name} className="sample-chip-thumb" />
              <div className="sample-chip-meta">
                <span className="sample-chip-title">{sample.name}</span>
                <span className="sample-chip-tag" style={{ color: sample.badgeColor }}>
                  {sample.badge}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

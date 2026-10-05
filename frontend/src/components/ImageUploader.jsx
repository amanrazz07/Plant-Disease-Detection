import { useState } from 'react';

import { API_URL } from '../config';

export default function ImageUploader({ selectedModel, onResult, onError, onLoading }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);

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
    handleFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const handleRemove = () => {
    setImage(null);
    setPreview(null);
    onResult(null);
  };

  const handlePredict = async () => {
    if (!image) return;

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
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed.');
      }

      const data = await res.json();
      onResult(data);
    } catch (err) {
      onError(err.message || 'Failed to connect to the API. Is the backend running?');
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="upload-section">
      <h2 className="section-title">
        <span>📸</span> Upload Plant Image
      </h2>

      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !preview && document.getElementById('file-input').click()}
        id="upload-zone"
      >
        {!preview ? (
          <>
            <span className="upload-icon">🌱</span>
            <div className="upload-text">
              <h3>Drop your plant image here</h3>
              <p>
                or <span className="highlight">click to browse</span> from your device
              </p>
            </div>
            <div className="upload-formats">
              <span className="format-tag">JPG</span>
              <span className="format-tag">PNG</span>
              <span className="format-tag">WebP</span>
              <span className="format-tag">BMP</span>
            </div>
          </>
        ) : (
          <div className="preview-container">
            <img src={preview} alt="Plant preview" className="preview-image" />
            <div className="preview-actions">
              <button className="btn btn-primary" onClick={handlePredict} id="predict-btn">
                🔍 Analyze Plant
              </button>
              <button className="btn btn-danger" onClick={handleRemove} id="remove-btn">
                ✕ Remove
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
    </div>
  );
}

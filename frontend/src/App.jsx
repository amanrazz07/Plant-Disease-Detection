import { useState } from 'react';
import Header from './components/Header';
import ModelSelector from './components/ModelSelector';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import ModelComparison from './components/ModelComparison';
import Footer from './components/Footer';

export default function App() {
  const [selectedModel, setSelectedModel] = useState('mobilenetv2');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="app">
      <Header />

      <main className="main-content">
        {/* Model Selection */}
        <ModelSelector selectedModel={selectedModel} onSelect={setSelectedModel} />

        {/* Image Upload */}
        <ImageUploader
          selectedModel={selectedModel}
          onResult={setResult}
          onError={setError}
          onLoading={setLoading}
        />

        {/* Error Message */}
        {error && (
          <div className="error-message" id="error-message">
            ⚠️ {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-overlay" id="loading-spinner">
            <div className="spinner" />
            <span className="loading-text">Analyzing plant image…</span>
          </div>
        )}

        {/* Prediction Result */}
        {!loading && result && <PredictionResult result={result} />}

        {/* Model Comparison Table */}
        <ModelComparison />
      </main>

      <Footer />
    </div>
  );
}

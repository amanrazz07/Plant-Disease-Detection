export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-top-row">
          <div className="footer-brand-mini">
            <span className="footer-logo">🌿</span>
            <span className="footer-name">FloraScan AI</span>
            <span className="footer-tagline">— Multi-Model Crop Pathology Engine</span>
          </div>

          <div className="footer-links">
            <a
              href="https://plant-disease-detection-r3c1.onrender.com/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              API Docs (Swagger)
            </a>
            <span className="footer-sep">•</span>
            <a
              href="https://github.com/amanrazz07/Plant-Disease-Detection"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              GitHub Repository
            </a>
            <span className="footer-sep">•</span>
            <span className="footer-link-text">PlantVillage Dataset (38 Classes)</span>
          </div>
        </div>

        <div className="footer-bottom-row">
          <p className="footer-copyright">
            Designed & Engineered with Deep Learning • PyTorch / TensorFlow 2.x • FastAPI • React (Vite)
          </p>
        </div>
      </div>
    </footer>
  );
}

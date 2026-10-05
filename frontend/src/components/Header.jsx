export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <span className="header-logo" role="img" aria-label="Plant">🌿</span>
          <div>
            <div className="header-title">Plant Disease Detection</div>
            <div className="header-subtitle">AI-Powered Diagnostics</div>
          </div>
        </div>
        <div className="header-badge">
          Powered by Deep Learning
        </div>
      </div>
    </header>
  );
}

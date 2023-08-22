export const MobileControls = ({ onLeft, onZoomIn, onRight, onZoomOut }) => {
  return (
    <div className="mobile-controls">
      <div className="left-controls">
        <button onClick={onLeft}>←</button>
        <button onClick={onZoomIn}>↖</button>
      </div>
      <div className="right-controls">
        <button onClick={onRight}>→</button>
        <button onClick={onZoomOut}>↗</button>
      </div>
      <style jsx>{`
        .mobile-controls {
          position: absolute;
          top: 0;
          width: 100%;
          display: flex;
          justify-content: space-between;
          z-index: 10;
        }
        .left-controls,
        .right-controls {
          display: flex;
          flex-direction: column;
          padding: 10px;
        }
      `}</style>
    </div>
  )
}

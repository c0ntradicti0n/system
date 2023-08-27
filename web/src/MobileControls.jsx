import { useEffect } from 'react'
import ShareModal from './ShareModal'

export const MobileControls = ({
  onLeft,
  onZoomIn,
  onRight,
  onZoomOut,
  linkId,
}) => {
  console.log('LINKID', linkId)
  useEffect(() => {
    const handleKeyDown = (e) => {
      switch (e.key) {
        case 'ArrowLeft':
          onLeft()
          break
        case 'ArrowRight':
          onRight()
          break
        case 'ArrowUp':
          onZoomIn()
          break
        case 'ArrowDown':
          onZoomOut()
          break
        default:
          return
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [onLeft, onRight, onZoomIn, onZoomOut])

  return (
    <div className="mobile-controls">
      <button
        onClick={onZoomIn}
        className="button top-controls"
        aria-label="Up"
        title="Hotkey: ArrowUp"
      >
        ↑
      </button>
      <button
        onClick={onLeft}
        className="button left-controls"
        aria-label="Left"
        title="Hotkey: ArrowLeft"
      >
        ←
      </button>
      <ShareModal linkId={linkId} />

      <button
        onClick={onRight}
        className="button right-controls"
        aria-label="Right"
        title="Hotkey: ArrowRight"
      >
        →
      </button>
      <button
        onClick={onZoomOut}
        className="button bottom-controls"
        aria-label="Down"
        title="Hotkey: ArrowDown"
      >
        ↓
      </button>
    </div>
  )
}

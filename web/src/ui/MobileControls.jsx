import { useEffect, useState } from 'react'
import ShareModal from './ShareModal'
import { Button, Input } from 'antd'
import { SearchOutlined } from '@ant-design/icons'

export const MobileControls = ({
  triggerSearch,
  searchText: _searchText,
  onLeft,
  onZoomIn,
  onRight,
  onZoomOut,
  linkInfo,
  isWindowWide,
  labels,
}) => {
  const [searchText, setSearchText] = useState(_searchText ?? '')
  useEffect(() => {
    setSearchText(_searchText)
  }, [_searchText])
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.id === 'search') {
        return // If it is, exit early and don't process the key event
      }

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

  const mobileStyles = !isWindowWide
            ? { left: 50, top: "27vh", position: 'fixed' }
            : {
                top: '5vh',
              }
  return (
    <div className="mobile-controls" style={{}}>
      <div
        className="top-search"
        style={
          !isWindowWide
            ? {
                position: 'fixed',
                left: labels?.length ? '5vw' : '76vw',
                top: labels?.length ? '40vh' : '1vh',
              }
            : {
                top: '5vh',
              }
        }
      >
        <Input.TextArea
          id="search"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          placeholder="How not to get murdered in the woods"
          autoSize
        />
        <Button
          type="primary"
          icon={<SearchOutlined />}
          onClick={() => triggerSearch(searchText)}
        />
      </div>
      <div className="navigation-controls" style={mobileStyles}>
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
        <ShareModal linkInfo={linkInfo} />
      </div>
    </div>
  )
}

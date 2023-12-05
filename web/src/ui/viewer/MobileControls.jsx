import { useEffect, useState } from 'react'
import { Button, Input } from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { useNavigate } from 'raviger'
import * as PropTypes from 'prop-types'

export const Search = ({ triggerSearch, searchText: _searchText }) => {
  const [searchText, setSearchText] = useState(_searchText ?? '')
  useEffect(() => {
    setSearchText(_searchText)
  }, [_searchText])

  return (
    <div className="top-search">
      <Input
        value={searchText}
        onChange={(e) => setSearchText(e.target.value)}
        placeholder="How not to get murdered in the woods"
        style={{ width: '100%' }}
      />
      <Button
        type="primary"
        icon={<SearchOutlined />}
        onClick={() => triggerSearch(searchText)}
      />
    </div>
  )
}

export function EditorLink(props) {
  return (
    <Button
      className="red"
      href="/editor#activeTab=lib"
      title="Create a new fractal"
      link
      style={{
        color: 'lime',
        //bold font
        fontWeight: 'bold',
        fontSize: '1.5vw',
        height: '3vw',

        // invert colors
        filter: 'invert(100%)',
      }}
    >
      Create a new fractal
    </Button>
  )
}

EditorLink.propTypes = { onClick: PropTypes.func }
export const Navigation = ({
  searchText: _searchText,
  onLeft,
  onZoomIn,
  onRight,
  onZoomOut,
  linkInfo,
}) => {
  const navigate = useNavigate()

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
      <div className="navigation-controls">
        <button
          onClick={onZoomIn}
          className="button red top-controls"
          aria-label="Up"
          title="Hotkey: ArrowUp"
        >
          ↑
        </button>
        <button
          onClick={onLeft}
          className="button red left-controls"
          aria-label="Left"
          title="Hotkey: ArrowLeft"
        >
          ←
        </button>

        <button
          onClick={onRight}
          className="button red right-controls"
          aria-label="Right"
          title="Hotkey: ArrowRight"
        >
          →
        </button>
        <button
          onClick={onZoomOut}
          className="button red bottom-controls"
          aria-label="Down"
          title="Hotkey: ArrowDown"
        >
          ↓
        </button>
      </div>
    </div>
  )
}

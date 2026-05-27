import { useEffect } from 'react'
import { Button } from 'antd'
import {
  MoreOutlined,
  BulbOutlined,
} from '@ant-design/icons'
import { go } from '../../lib/navigate'
import { useTheme } from '../../App'

export const Toolbar = ({ menuVisible, setMenuVisible, fractalRef }) => {
  const { lightMode, toggleLightMode } = useTheme()

  const nav = (direction) => go({ ...(fractalRef?.current ?? {}), direction })

  useEffect(() => {
    const handleKeyDown = (e) => {
      switch (e.key) {
        case 'ArrowLeft':
          nav('left')
          break
        case 'ArrowRight':
          nav('right')
          break
        case 'ArrowUp':
          nav('lower')
          break
        case 'ArrowDown':
          nav('higher')
          break
        default:
          return
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [fractalRef])

  return (
    <div className="toolbar">
      <div className="toolbar-left">
        <Button
          className="toolbar-btn"
          type="text"
          icon={<MoreOutlined />}
          onClick={() => setMenuVisible(!menuVisible)}
          title="Menu"
        />
        <Button
          className="toolbar-btn"
          type="text"
          icon={<BulbOutlined />}
          onClick={toggleLightMode}
          title={lightMode ? 'Dark mode' : 'Light mode'}
        />
      </div>
      <div className="toolbar-right">
        <div className="nav-cross">
          <button
            className="nav-arrow nav-up"
            onClick={() => nav('lower')}
            title="Zoom in (ArrowUp)"
          >↑</button>
          <div className="nav-middle">
            <button
              className="nav-arrow nav-left"
              onClick={() => nav('left')}
              title="Go left (ArrowLeft)"
            >←</button>
            <span className="nav-center-dot"></span>
            <button
              className="nav-arrow nav-right"
              onClick={() => nav('right')}
              title="Go right (ArrowRight)"
            >→</button>
          </div>
          <button
            className="nav-arrow nav-down"
            onClick={() => nav('higher')}
            title="Zoom out (ArrowDown)"
          >↓</button>
        </div>
      </div>
    </div>
  )
}

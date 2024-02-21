import { useEffect, useState } from 'react'
import { Button, Input, Menu } from 'antd'
import { BookOutlined, SearchOutlined } from '@ant-design/icons'
import { useNavigate } from 'raviger'
import * as PropTypes from 'prop-types'
import ShareModal from '../ShareModal'
import { go } from '../../lib/navigate'
import ChatModal from '../ChatModal/ChatModal'
import useChatModal from "../ChatModal/state";
const { Search: AntdSearch } = Input

export const Search = ({
  triggerSearch,
  searchText: _searchText,
  setCollapsed,
}) => {
  return (
    <div style={{ display: 'inline-block', width: '90%' }}>
      <AntdSearch
        placeholder="input search text"
        enterButton
        size="large"
        onSearch={(value, _e, info) => {
          triggerSearch(value)
          setCollapsed(true)
        }}
      />
    </div>
  )
}

export function EditorLink() {
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
        width: '90%',
        filter: 'invert(100%)',
      }}
    >
      Make your own fractal
    </Button>
  )
}

export function ViewerLink() {
  return (
    <Button
      className="red"
      href="/"
      title="Create a new fractal"
      link
      style={{
        color: 'lime',
        //bold font
        fontWeight: 'bold',
        fontSize: '1.5vw',
        height: '3vw',
        width: '90%',
        filter: 'invert(100%)',
      }}
    >
      Back to the fractal
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

export const ViewerMenu = ({
  searchText,
  linkInfo,
  path,
  setPath,
  fractalRef,
}) => {
  const isMobile =
    'ontouchstart' in window ||
    navigator.maxTouchPoints ||
    window.innerWidth < 950
  console.log('isMobile', isMobile)

  const [collapsed, setCollapsed] = useState(true)
 const chatModal = useChatModal()
    console.log(chatModal)

  return (
    <div>
      <Button
        type="primary"
        icon={<BookOutlined />}
        onClick={() => setCollapsed(!collapsed)}
        style={{ width: isMobile ? '90vw' : '45vw', minWidth: '300' }}
      ></Button>
      {!collapsed && (
        <Menu
          mode="vertical"
          theme="dark"
          style={{
            width: '45vw',
          }}
        >
          <Menu.Item key="1" icon={<BookOutlined />}>
            <Search
              searchText={searchText}
              triggerSearch={fractalRef?.current?.triggerSearch}
              setSearchText={fractalRef?.current?.setSearchText}
              setCollapsed={setCollapsed}
            />
          </Menu.Item>
          <Menu.Item key="2" icon={<BookOutlined />}>
            <ShareModal linkInfo={linkInfo} />
          </Menu.Item>
          <Menu.Item
            key="3"
            icon={<BookOutlined />}
            style={{ height: '110px' }}
          >
            <Navigation
              onLeft={() =>
                go({ ...(fractalRef?.current ?? {}), direction: 'left' })
              }
              onZoomIn={() =>
                go({ ...(fractalRef?.current ?? {}), direction: 'lower' })
              }
              onRight={() =>
                go({ ...(fractalRef?.current ?? {}), direction: 'right' })
              }
              onZoomOut={() =>
                go({ ...(fractalRef?.current ?? {}), direction: 'higher' })
              }
            />
          </Menu.Item>

          <Menu.Item key="4" icon={<BookOutlined />}>
            <EditorLink />
          </Menu.Item>
          <Menu.Item key="6" icon={<BookOutlined />} onClick={()=> {console.log("MENUG")+
          chatModal.setVisible(true)}}>

              elaborate a topic
          </Menu.Item>
        </Menu>
      )}
        {chatModal.visible ? <ChatModal
            /> : null}
    </div>
  )
}

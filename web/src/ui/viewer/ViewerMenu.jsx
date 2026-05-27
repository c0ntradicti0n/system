import { useEffect, useRef } from 'react'
import { Button, Input } from 'antd'
import { BookOutlined, SearchOutlined } from '@ant-design/icons'
import ShareModal from '../ShareModal'
import ChatModal from '../ChatModal/ChatModal'
import useChatModal from '../ChatModal/state'
const { Search: AntdSearch } = Input

export const Search = ({
  triggerSearch,
  searchText: _searchText,
  onSearch,
}) => {
  return (
    <div style={{ width: '100%' }}>
      <AntdSearch
        placeholder="Search the fractal…"
        enterButton
        size="small"
        onSearch={(value) => {
          triggerSearch(value)
          onSearch?.()
        }}
      />
    </div>
  )
}

export function EditorLink() {
  return (
    <Button
      className="menu-item-btn"
      href="/editor#activeTab=lib"
      title="Create a new fractal"
      link
    >
      Make your own fractal
    </Button>
  )
}

export function ViewerLink() {
  return (
    <Button
      className="menu-item-btn"
      href="/"
      title="Back to the fractal"
      link
    >
      Back to the fractal
    </Button>
  )
}

export const ViewerMenu = ({
  visible,
  onClose,
  searchText,
  linkInfo,
  fractalRef,
}) => {
  const chatModal = useChatModal()
  const autoCloseRef = useRef(null)
  const menuRef = useRef(null)

  const resetAutoClose = () => {
    if (autoCloseRef.current) {
      clearTimeout(autoCloseRef.current)
      autoCloseRef.current = null
    }
  }

  const startAutoClose = () => {
    resetAutoClose()
    autoCloseRef.current = setTimeout(() => {
      onClose?.()
    }, 5000)
  }

  useEffect(() => {
    if (visible) {
      startAutoClose()
    } else {
      resetAutoClose()
    }
    return resetAutoClose
  }, [visible])

  if (!visible) return null

  return (
    <div
      className="menu-popover-overlay"
      onClick={onClose}
    >
      <div
        ref={menuRef}
        className="menu-popover"
        onClick={(e) => e.stopPropagation()}
        onMouseEnter={resetAutoClose}
        onMouseLeave={startAutoClose}
      >
        <div className="menu-popover-item">
          <SearchOutlined className="menu-item-icon" />
          <Search
            searchText={searchText}
            triggerSearch={fractalRef?.current?.triggerSearch}
            onSearch={onClose}
          />
        </div>

        <div className="menu-popover-item">
          <BookOutlined className="menu-item-icon" />
          <ShareModal linkInfo={linkInfo} />
        </div>

        <div className="menu-popover-item">
          <EditorLink />
        </div>

        <div
          className="menu-popover-item menu-clickable"
          onClick={() => {
            chatModal.setVisible(true)
            onClose?.()
          }}
        >
          <BookOutlined className="menu-item-icon" />
          <span>Ask the Curator</span>
        </div>

        {chatModal.visible ? <ChatModal /> : null}
      </div>
    </div>
  )
}

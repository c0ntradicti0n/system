import React, { useState, useEffect } from 'react'
import { Modal, Button, Input } from 'antd'
import { ShareAltOutlined, CopyOutlined } from '@ant-design/icons'
import { removeMultipleSlashes } from '../lib/nesting'

const ShareModal = ({ url = '', linkInfo }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  const linkInfoNoNull = Object.fromEntries(
    Object.entries(linkInfo)
      .filter(([_, v]) => v !== null)
      .map(([k, v]) => [k, v?.toString().replace(/\//g, '')]),
  )
  const fullUrl =
    `${window.location.protocol}//` +
    removeMultipleSlashes(
      `${window.location.host}${url}#${new URLSearchParams(
        linkInfoNoNull,
      ).toString()}`,
    )

  const handleCopy = () => {
    navigator.clipboard.writeText(fullUrl)
    setIsCopied(true)
    setTimeout(() => {
      setIsCopied(false)
      setIsVisible(false)
    }, 2000) // Reset after 2 seconds
  }

  const handleHotkey = (e) => {
    if (e.target.id === 'search') {
      return // If it is, exit early and don't process the key event
    }
    if (e.key === 's') {
      setIsVisible(true)
    }
  }

  return (
    <div style={{ zIndex: 112323213214213 }}>
      <Button
        onClick={() => setIsVisible(true)}
        icon={<ShareAltOutlined />}
        className="share-button red"
        aria-label="Share (Hotkey: s)"
      />
      <Modal
        title="Share this URL"
        open={isVisible}
        onCancel={() => setIsVisible(false)}
        footer={null}
        centered
      >
        <Input
          value={fullUrl}
          addonAfter={
            <Button
              style={{ background: 'none', border: 'none' }}
              icon={<CopyOutlined />}
              onClick={handleCopy}
              aria-label={'Copy'}
            ></Button>
          }
          readOnly
          className={isCopied ? 'copied-animation' : ''}
        />
      </Modal>
    </div>
  )
}

export default ShareModal

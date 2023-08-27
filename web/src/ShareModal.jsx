import React, { useState, useEffect } from 'react'
import { Modal, Button, Input } from 'antd'
import { ShareAltOutlined, CopyOutlined } from '@ant-design/icons'
import { removeMultipleSlashes } from './nesting'

const ShareModal = ({ linkId }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  console.log(linkId)
  const fullUrl =
    `${window.location.protocol}//` +
    removeMultipleSlashes(`${window.location.host}#${linkId}`)

  const handleCopy = () => {
    navigator.clipboard.writeText(fullUrl)
    setIsCopied(true)
    setTimeout(() => {
      setIsCopied(false)
    }, 2000) // Reset after 2 seconds
  }

  const handleHotkey = (e) => {
    if (e.key === 's') {
      setIsVisible(true)
    }
  }

  useEffect(() => {
    window.addEventListener('keydown', handleHotkey)
    return () => {
      window.removeEventListener('keydown', handleHotkey)
    }
  }, [])

  return (
    <div style={{ zIndex: 112323213214213 }}>
      <Button
        onClick={() => setIsVisible(true)}
        icon={<ShareAltOutlined />}
        aria-label="Share (Hotkey: s)"
      />
      <Modal
        title="Share this URL"
        visible={isVisible}
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

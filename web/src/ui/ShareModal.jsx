import React, { useState } from 'react'
import { Modal, Button, Input } from 'antd'
import { ShareAltOutlined, CopyOutlined } from '@ant-design/icons'
import { removeMultipleSlashes } from '../lib/nesting'

const ShareModal = ({ url = '', linkInfo }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  const linkInfoNoNull = Object.fromEntries(
    Object.entries(linkInfo)
      .filter(([_, v]) => v !== null && v !== undefined)
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
  return (
    <>
      <Button
        onClick={() => setIsVisible(true)}
        icon={<ShareAltOutlined />}
        aria-label="Share (Hotkey: s)"
        style={{ width: '90%' }}
      >
        Share
      </Button>
      <Modal
        title="Share this URL"
        open={isVisible}
        onCancel={() => setIsVisible(false)}
        footer={null}
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
    </>
  )
}

export default ShareModal

import React, { useEffect, useState } from 'react'
import { Modal, Button, Input } from 'antd'
import { EditOutlined } from '@ant-design/icons'

const { TextArea } = Input

const TextMetaModal = ({ socket, text, hash, meta }) => {
  const [isModalVisible, setIsModalVisible] = useState(false)
  const [_text, _setText] = useState(null)
  const [_meta, _setMeta] = useState(null)
  const showModal = () => {
    setIsModalVisible(true)
  }

  console.log('text', text)
  useEffect(() => {
    console.log('get_full_text', hash, isModalVisible, text)
    if (hash && isModalVisible) {
      console.log('emitting')

      socket.emit('get_full_text', hash, (__text) => {
        console.log('get_full_text replied', __text)
        _setText(__text)
      })
    }
  }, [hash, isModalVisible])

  const handleOk = () => {
    setIsModalVisible(false)
    console.log('set_text', _text, _meta)
    socket.timeout(3000).emit('save_text_meta', _text, _meta)
  }

  const handleCancel = () => {
    setIsModalVisible(false)
  }
  console.log('text', text)

  return (
    <>
      <Button className="red" icon={<EditOutlined />} onClick={showModal}>
        Text
      </Button>
      <Modal
        title="Setup new text"
        visible={isModalVisible}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <h4>Text</h4>

        <TextArea
          showCount
          value={_text ?? text}
          maxLength={1000000}
          style={{
            height: 120,
            marginBottom: 24,
            backgroundColor: '#111',
            color: '#fff !important',
          }}
          onChange={(e) => {
            _setText(e.target.value)
          }}
          placeholder="Paste your paragraphed text here"
        />
        <h4>Bibtex-Metadata</h4>
        <TextArea
          showCount
          value={_meta ?? meta}
          maxLength={1000}
          style={{
            height: 120,
            marginBottom: 24,
            backgroundColor: '#111',
            color: '#fff',
          }}
          onChange={(e) => {
            _setMeta(e.target.value)
          }}
          placeholder="Paste some Metadata here, can have bib-text-formatted references"
        />
        <Button onClick={handleOk}>Send Text</Button>
      </Modal>
    </>
  )
}

export default TextMetaModal

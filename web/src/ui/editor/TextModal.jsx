import React, { useState } from 'react'
import { Modal, Button, Input } from 'antd'
import { EditOutlined } from '@ant-design/icons'

const { TextArea } = Input

const TextModal = ({ socket, text, setText, _text, _setText, hash }) => {
  const [isModalVisible, setIsModalVisible] = useState(false)

  const showModal = () => {
    setIsModalVisible(true)
  }

  const handleOk = () => {
    setText(_text)
    setIsModalVisible(false)
    // Assuming `setHash` is a function you've defined elsewhere
    // setHash(null);
    console.log('set_text', _text)
    // Assuming `socket` is defined elsewhere
    socket.timeout(3000).emit('save_text', _text)
  }

  const handleCancel = () => {
    setIsModalVisible(false)
  }

  return (
    <div
      className="red"
      style={{
        position: 'fixed',

        width: 'min-content',
        zIndex: 112323213214213,
      }}
    >
      <Button className="red" icon={<EditOutlined />} onClick={showModal}>
        ✏️
      </Button>
      <Modal
        title="Edit Text"
        visible={isModalVisible}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <TextArea
          showCount
          value={_text}
          maxLength={1000000}
          style={{
            height: 120,
            marginBottom: 24,
            backgroundColor: '#111',
            color: '#fff',
          }}
          onChange={(e) => {
            _setText(e.target.value)
          }}
          placeholder="Paste your paragraphed text here"
        />
        <Button onClick={handleOk}>Send Text</Button>
      </Modal>
    </div>
  )
}

export default TextModal

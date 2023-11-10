import React, { useState } from 'react'
import { Modal, Button, Input } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'

const { TextArea } = Input

const MetaModal = ({ socket, meta, setMeta, hash }) => {
  const [isModalVisible, setIsModalVisible] = useState(false)

  const showModal = () => {
    setIsModalVisible(true)
  }

  const handleOk = () => {
    setIsModalVisible(false)
    console.log('set_meta', meta)
    socket.timeout(3000).emit('save_meta', hash, meta)
  }

  const handleCancel = () => {
    setIsModalVisible(false)
  }

  return (
    <div
      className="red"
      style={{
        width: 'min-content',
        zIndex: 112323213214213,
        position: 'fixed',
          top:"5vh"
      }}
    >
      <Button className="red" icon={<InfoCircleOutlined />} onClick={showModal}>
        🗃️
      </Button>
      <Modal
        title="Metadata"
        visible={isModalVisible}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <TextArea
          showCount
          value={meta}
          maxLength={1000}
          style={{
            height: 120,
            marginBottom: 24,
            backgroundColor: '#111',
            color: '#fff',
          }}
          onChange={(e) => {
            setMeta(e.target.value)
          }}
          placeholder="Paste some Metadata here, can have bib-text-formatted references"
        />
        <Button onClick={handleOk}>Set Metadata</Button>
      </Modal>
    </div>
  )
}

export default MetaModal

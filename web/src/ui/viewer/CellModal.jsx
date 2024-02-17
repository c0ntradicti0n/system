import React from 'react'
import { Modal, Input } from 'antd'

const CellModal = ({
  visible,
  value,
  keys,
  setValue,
  onClose,
  kind = 'title',
}) => {
  const md_regex = /(\.md)$/g
  console.log("CellModal", value, keys)
  const endswith_md = md_regex.test(value)
  value = value.replaceAll(md_regex, '')
  return (
    <Modal
      title={'Change the ' + kind + keys.join('.')}
      visible={visible}
      onCancel={onClose}
      footer={null}
    >
      <Input
        placeholder="Enter a better text"
        defaultValue={value}
        onPressEnter={(e) => {
          if (endswith_md) {
            setValue(keys.filter(x=>x), e.target.value + '.md')
          }
          setValue(keys.filter(x=>x), e.target.value)
          onClose()  }
      }
      />
    </Modal>
  )
}

export default CellModal

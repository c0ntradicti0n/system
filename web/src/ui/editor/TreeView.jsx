import React, { useState } from 'react'
import { Tree } from 'antd'
import { convertToAntdTreeData } from '../../lib/convertToAntdTreeData'

export function TreeView(props) {
  const treeData = convertToAntdTreeData(props.state)
  const [expandedKeys, setExpandedKeys] = useState([])
  const onExpand = (expandedKeysValue) => {
    console.log('onExpand', expandedKeysValue)
    setExpandedKeys(expandedKeysValue)
  }
  return (
    <Tree
      showIcon
      showLine
      checkStrictly={true}
      expandedKeys={expandedKeys}
      onExpand={onExpand}
      treeData={treeData}
      titleHeight={'10px'}
      loadData={async (node) => {
        console.log(node)
      }}
    />
  )
}

import React, {useState} from "react";
import {Tree} from "antd";
import {convertToAntdTreeData} from "../Tooltips";

export function TreeView(props) {
      const treeData = convertToAntdTreeData(props.state)
      const [expandedKeys, setExpandedKeys] = useState([])
  const [openedKeys, setOpenedKeys] = useState(null)
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
          setOpenedKeys(node.key.replace(/-/g, '/').slice(0, -2))
        }}
      />

    )
}
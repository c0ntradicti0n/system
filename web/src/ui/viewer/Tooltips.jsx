import React, { useEffect, useMemo, useState } from 'react'
import { useQuery } from 'react-query'
import { Tree } from 'antd'
import { mergeDeep } from '../../lib/nesting'
import {
  convertToAntdTreeData,
  generateExpandedKeys,
  nestTexts,
} from '../../lib/convertToAntdTreeData'

export const Tooltips = ({ tree: _tree, path, setTooltipData, refill }) => {
  const tree = structuredClone(_tree)

  const [expandedKeys, setExpandedKeys] = useState([])
  const [openedKeys, setOpenedKeys] = useState(null)

  console.log()
  const onExpand = (expandedKeysValue) => {
    console.log('onExpand', expandedKeysValue)
    setExpandedKeys(expandedKeysValue)
  }
  const fetchTexts = async () => {
    if (!refill) return {}
    const p = openedKeys ?? path ?? ''
    const res = await fetch(`/api/text/${p}`)

    if (!res.ok) {
      throw new Error('Network response was not ok')
    }
    const newData = await res.json()
    console.log('fetched, ', p, newData)
    return newData
  }

  const { data: texts } = useQuery(['triangle', path, openedKeys], fetchTexts, {
    onError: console.error,
    keepPreviousData: true,
  })

  const data = useMemo(() => {
    const nestedTextObject = nestTexts(openedKeys ?? path, texts)

    return mergeDeep(tree, nestedTextObject)
  }, [openedKeys, path, texts, tree])

  const onSelect = (selectedKeys, info) => {
    console.log('selected', selectedKeys, info)
  }

  const treeData = convertToAntdTreeData(data)
  const paths = ['', '1', '2', '3'].map(
    (key) =>
      path.replace(/^\/+|\/+$/g, '').replace(/\//g, '-') +
      (key ? '-' + key : ''),
  )
  useEffect(() => {
    const baseKey = path.replace(/^\/+|\/+$/g, '').replace(/\//g, '-')
    const keys = ['', '1', '2', '3'].map(
      (key) => baseKey + (key ? '-' + key : ''),
    )
    const allExpandedKeys = keys.flatMap((key) => generateExpandedKeys(key))

    // To avoid duplicates, use a Set and then convert back to an array
    setExpandedKeys([...new Set(allExpandedKeys)])
  }, [path])
  return (
    <div className="tooltips">
      <div
        style={{
          position: 'absolute',
          top: 0,
          right: 15,
          color: '#000',
          zIndex: 9999999999,
          cursor: 'pointer',
        }}
        onClick={() => setTooltipData(null)}
      >
        &#x2716;
      </div>
      <Tree
        showIcon
        showLine
        checkStrictly={true}
        defaultExpandedKeys={paths}
        expandedKeys={expandedKeys}
        onExpand={onExpand}
        onSelect={onSelect}
        treeData={treeData}
        titleHeight={'10px'}
        loadData={async (node) => {
          console.log(node)
          setOpenedKeys(node.key.replace(/-/g, '/').slice(0, -2))
        }}
      />
    </div>
  )
}

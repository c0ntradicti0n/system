import { postProcessTitle } from '../lib/position'
import React, { useEffect, useMemo, useState } from 'react'
import { useQuery } from 'react-query'
import { Tree } from 'antd'
import { mergeDeep } from '../lib/nesting'
import ReactMarkdown from 'react-markdown'

const generateExpandedKeys = (path) => {
  const segments = path.split('-')
  const keys = []

  for (let i = 1; i <= segments.length; i++) {
    keys.push(segments.slice(0, i).join('-'))
  }

  return keys
}
const customSort = (a, b) => {
  if (a === '.text') return -1
  if (b === '.text') return 1
  if (a === '_text') return 1
  if (b === '_text') return -1
  return Number(a) - Number(b)
}
export const convertToAntdTreeData = (node, prefix = '') => {
  const result = []

  if (!node) return []
  const sortedKeys = Object.keys(node).sort(customSort)
  for (let key of sortedKeys) {
    if (key === '.' || key === '_') continue
    const childNode = node[key]
    const currentKey = prefix ? `${prefix}-${key}` : key
    const isObject = typeof childNode === 'object'

    let title = key
    if (key.includes('text')) {
      title = (
        <div
          style={{
            textAlign: 'justify',
            textAlignLast: 'none',
            background: '#fff2d4',
          }}
        >
          {' '}
          <ReactMarkdown>{postProcessTitle(node[key])}</ReactMarkdown>
        </div>
      )
    }
    if (isObject) {
      title = postProcessTitle(childNode?.['.']) ?? ''
    }

    let treeNode = {
      title: title,
      key: currentKey,
      icon: isObject ? key : key === 'text' ? '◬' : '▰',
    }

    // Check if the node has children (ignoring special keys like "." and "_")
    if (
      typeof childNode === 'object' &&
      key !== '.' &&
      key !== '_' &&
      !key.includes('text')
    ) {
      treeNode.children = convertToAntdTreeData(childNode, currentKey)
    }

    result.push(treeNode)
  }

  return result
}

const nestTexts = (path, texts) => {
  if (!texts) return {}
  if (!path) return texts

  const keys = path.split('/').filter(Boolean) // Split by '/' and filter out empty strings
  let currentObject = {}

  keys.reduce((obj, key, index) => {
    if (index === keys.length - 1) {
      // If we are at the last key in the path
      obj[key] = Object.fromEntries(
        Object.entries(texts).map(([key, value]) => [
          ['.', '_'].includes(key) ? key + 'text' : key,
          ['.', '_'].includes(key) ? value : { text: value },
        ]),
      )
    } else {
      obj[key] = {} // Else create an empty object for the next level
    }
    return obj[key] // Return the nested object for the next iteration
  }, currentObject)

  return currentObject
}

export const Tooltips = ({
  tree: _tree,
  path,
  isWindowWide,
  setTooltipData,
    refill
}) => {
  const tree = structuredClone(_tree)

  const [expandedKeys, setExpandedKeys] = useState([])
  const [openedKeys, setOpenedKeys] = useState(null)

  console.log()
  const onExpand = (expandedKeysValue) => {
    console.log('onExpand', expandedKeysValue)
    setExpandedKeys(expandedKeysValue)
  }
  const fetchTexts = async () => {
    if (!refill)
      return {}
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

import { postProcessTitle } from '../lib/position'
import React, { useEffect, useMemo, useState } from 'react'
import { useQuery } from 'react-query'
import { Tree } from 'antd'
import { mergeDeep } from '../lib/nesting'

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
const convertToAntdTreeData = (node, prefix = '') => {
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
          {postProcessTitle(node[key])}
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
    if (typeof childNode === 'object' && key !== '.' && key !== '_') {
      treeNode.children = convertToAntdTreeData(childNode, currentKey)
    }

    result.push(treeNode)
  }

  return result
}

const updateTreeData = (list, key, children) =>
  list.map((node) => {
    if (node.key === key) {
      return {
        ...node,
        children,
      }
    }
    if (node.children) {
      return {
        ...node,
        children: updateTreeData(node.children, key, children),
      }
    }
    return node
  })

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

export const Tooltips = ({ tree: _tree, path, isWindowWide, setTooltipData }) => {
  const tree = structuredClone(_tree)

  const [expandedKeys, setExpandedKeys] = useState([])

  const fetchTexts = async () => {
    const res = await fetch(`/api/text/${path ?? ''}`)
    if (!res.ok) {
      throw new Error('Network response was not ok')
    }
    const newData = await res.json()
    return newData
  }

  const { data: texts } = useQuery(['triangle', path], fetchTexts, {
    onError: console.error,
    keepPreviousData: true,
  })

  const data = useMemo(() => {
    const nestedTextObject = nestTexts(path, texts)

    return mergeDeep(tree, nestedTextObject)
  }, [path, texts, tree])

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
    <div
      style={{
        position: 'relative',
        top: 0,
        right: isWindowWide ? 0 : 'auto',
        height: isWindowWide ? '100vh' : '30vh',
        width: isWindowWide ? '40vw' : 'vw',
        zIndex: 99999999,
        overflow: 'auto',
        resize: 'both',
      }}
    >
      <div style={{ position: 'absolute', top: 0, right: 15, color: "#000", zIndex: 9999999999, cursor: "pointer" }}
      onClick={() => setTooltipData(null) }
      >&#x2716;</div>
      <Tree
        showIcon
        showLine
        checkStrictly={true}
        defaultExpandedKeys={paths}
        expandedKeys={expandedKeys}
        onExpand={setExpandedKeys}
        onSelect={onSelect}
        treeData={treeData}
      />
    </div>
  )
}

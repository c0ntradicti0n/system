import ReactMarkdown from 'react-markdown'
import React from 'react'
import { postProcessTitle } from './position'

export const generateExpandedKeys = (path) => {
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

export const nestTexts = (path, texts) => {
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

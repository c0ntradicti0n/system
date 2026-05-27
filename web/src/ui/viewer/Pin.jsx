import { useEffect, useState } from 'react'
import LeaderLine from 'leader-line-new'
import useLinkedElementsStore from '../../lib/PinnedElements'

function findElementById(id) {
  let currentId = id
  let element

  while (!element && currentId.length > 0) {
    element = document.getElementById(currentId)
    if (!element) {
      currentId = currentId.slice(0, -1)
    }
  }
  if (!element) {
    console.log('Could not find element with id', id)
  }

  return element
}

function tryIt(callback) {
  try {
    callback()
  } catch (e) {
    console.log(e)
  }
}

function useElementById(baseId, shorterIdFinder = false, pollInterval = 200) {
  const [element, setElement] = useState(findElementById(baseId))

  useEffect(() => {
    const interval = setInterval(() => {
      let fetchedElement
      if (shorterIdFinder) fetchedElement = findElementById(baseId)
      else fetchedElement = document.getElementById(baseId)
      if (fetchedElement !== element) {
        setElement(fetchedElement)
      }
    }, pollInterval)

    return () => clearInterval(interval) // Cleanup on unmount or id change
  }, [baseId, element, pollInterval, shorterIdFinder])

  return element
}

function lineColor() {
  const val = getComputedStyle(document.documentElement).getPropertyValue('--pin-line-color').trim()
  return val || 'lawngreen'
}

function createLine(start, end) {
  if (!start || !end) return
  console.log(start, end)
  const mobile = window.innerWidth <= 600
  return new LeaderLine(start, end, {
    color: lineColor(),
    size: 2,
    endPlugSize: 1.5,
    startSocket: mobile ? 'right' : 'left',
    endSocket: 'top',
  })
}

export const Pin = ({ value, setHiddenId, setTooltipData }) => {
  const cleanPath = value.path.replace(/\//g, '').replace('.', '')
  const displayPath = value.path
    .split('/')
    .filter(s => s && s !== '_')
    .join('.')
  const pinId = 'pin-label-' + cleanPath
  const triangleId = 'triangle-' + cleanPath

  function renderContent() {
    if (value.answer) return <div className="search-item-text">{value.answer}</div>
    if (!value.content) return null
    const clean = value.content
      .replace(/^["\s]+|["\s]+$/g, '')
      .split('\n')
      .map(l => l.replace(/^[._]\s*/, '').replace(/["]+/g, '').trim())
      .filter((l, i, a) => l && a.indexOf(l) === i)
      .join(' ')
      .replace(/^(\S+)\s+\1\b/, '$1')
    const leaf = cleanPath.split('.').pop()
    if (!clean || clean === leaf) return null
    return <div className="search-item-text">{clean}</div>
  }
  const labelIdRef = useElementById(pinId)
  const triangleIdRef = useElementById(triangleId, true)
  const [line, setLine] = useState(null)
  console.log('pin', { labelIdRef, triangleIdRef, line })

  const { addElement, removeElement } = useLinkedElementsStore()

  useEffect(() => {
    if (triangleIdRef) {
      addElement({ id: triangleId, data: labelIdRef })
    }
    return () => {
      if (triangleIdRef) {
        removeElement(triangleId)
      }
    }
  }, [triangleIdRef, triangleId, addElement, labelIdRef, removeElement])

  useEffect(() => {
    if (!labelIdRef || !triangleIdRef) {
      if (line) {
        tryIt(() => line.remove())
        setLine(null)
      }
      return
    }
    if (line) return
    const newLine = createLine(labelIdRef, triangleIdRef)
    setLine(newLine)
  }, [line, labelIdRef, triangleIdRef])

  useEffect(() => {
    return () => tryIt(() => line?.remove())
  }, [line])

  const triangleBounds = triangleIdRef?.getBoundingClientRect()
  const labelBounds = labelIdRef?.getBoundingClientRect()

  const triangleLeft = triangleBounds?.left
  const triangleTop = triangleBounds?.top
  const labelLeft = labelBounds?.left
  const labelTop = labelBounds?.top

  useEffect(() => {
    if (line)
      try {
        console.log(
          'reposition',
          triangleLeft,
          triangleTop,
          labelLeft,
          labelTop,
        )
        line.position()
      } catch (e) {
        console.log(e)
      }
  }, [line, triangleLeft, triangleTop, labelIdRef?.left, labelTop, labelLeft])

  return (
    <div
      key={'pin-label-' + cleanPath}
      id={'pin-label-' + cleanPath}
      className="search-item"
      onClick={() => setHiddenId(value.path.slice(0, -2))}
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setTooltipData?.(value.path.slice(0, -2));
      }}
    >
      <div className="search-item-content">
        <div className="search-item-path">{displayPath}</div>
        {renderContent()}
      </div>
    </div>
  )
}

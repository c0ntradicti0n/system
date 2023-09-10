import { useEffect, useRef, useState } from 'react'
import LeaderLine from 'leader-line'

function findElementById(id) {
  let currentId = id
  let element

  while (!element && currentId.length > 0) {
    element = document.getElementById(currentId)
    if (!element) {
      currentId = currentId.slice(0, -1) // remove the last character
    }
  }

  return element
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
  }, [baseId, element, pollInterval])

  return element
}

function createLine(start, end) {
  if (!start || !end) return
  console.log(start, end)
  return new LeaderLine(start, end, {
    color: 'lawngreen',
    size: 2,
    endPlugSize: 1.5,
    startSocket: 'bottom',
    endSocket: 'top',
  })
}

export const Pin = ({ value, setHiddenId }) => {
  const cleanPath = value.path.replace(/\//g, '').replace('.', '')
  const labelIdRef = useElementById('pin-label-' + cleanPath)
  const triangleIdRef = useElementById('triangle-' + cleanPath, true)
  const [line, setLine] = useState(null)
  console.log('pin', { labelIdRef, triangleIdRef, line })

  useEffect(() => {
    if (!labelIdRef || !triangleIdRef)
      if (line) line.remove()
      else return
    if (line) return
    const newLine = createLine(labelIdRef, triangleIdRef)
    setLine(newLine)
  }, [line, labelIdRef, triangleIdRef])

  useEffect(() => {
    return () => line?.remove()
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
  }, [line, triangleLeft, triangleTop, labelIdRef?.left, labelTop])

  return (
    <div
      key={'pin-label-' + cleanPath}
      id={'pin-label-' + cleanPath}
      className="item"
      onClick={() => setHiddenId(value.path.slice(0, -2))}
    >
      <div className="item-content">
        {cleanPath}
        <br />
        {value.answer ?? value.content}
      </div>
    </div>
  )
}

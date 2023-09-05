import React, { useEffect, useRef } from 'react'
import Muuri from 'muuri'
import '../muuri.css'
import LeaderLine from 'leader-line'

function try_to_find_start_end(triangleId, labelId) {
  let currentId = triangleId
  let end

  while (currentId.length > 0) {
    end = document.getElementById(currentId)
    if (end) {
      break // We've found an element that matches the ID
    }
    currentId = currentId.slice(0, -1) // Strip the last character
  }
  const start = document.getElementById(labelId)
  console.log({ start, end }, { labelId, triangleId, currentId })
  return { end, start }
}

function createLine(triangleId, labelId) {
  let { end, start } = try_to_find_start_end(triangleId, labelId)
  if (!start || !end) return
  const line = new LeaderLine(start, end, {
    color: 'lawngreen',
    size: 4,
    endPlugSize: 1.5,
    dash: { animation: true },
    startSocket: 'right',
    endSocket: 'top',
  })

  return line
}

const MuuriComponent = ({ labels, setHiddenId, onScroll }) => {
  const [lines, setLines] = React.useState([])
  const gridRef = useRef(null)

  useEffect(() => {
    if (!gridRef.current) return
    const grid = new Muuri(gridRef.current, {
  dragEnabled: true,
  layout: { horizontal: true, alignRight: true }
    })

    // Clean up Muuri instance when component unmounts
    return () => {
      grid.destroy()
    }
  }, [gridRef])

  useEffect(() => {
    if (!labels) return
    setTimeout(() => {
      setLines(
        labels.map((value) => {
          const labelId =
            'pin-label-' + value.path.replace(/\//g, '').replace('.', '')
          const triangleId =
            'triangle-' + value.path.replace(/\//g, '').replace('.', '')
          return [triangleId, labelId, createLine(triangleId, labelId)]
        }),
      )
    }, '1 second')
  }, [labels])

  useEffect(() => {
    const definedLseni = lines.filter(([triangleId, labelId, line]) => line)
    if (definedLseni && definedLseni.length !== lines.length)
      setLines(definedLseni)

    return () => lines.forEach(([triangleId, labelId, line]) =>
    {try {
            line?.remove()
        } catch (e) {

    }})
  }, [lines])

  console.log(lines)

  lines.forEach(([triangleId, labelId, line]) => {
    const { start, end } = try_to_find_start_end(triangleId, labelId)
    if (!start || !end) {
      try {
        line.remove()
      } catch (e) {}
    }

    try {
      line?.position()
    } catch (e) {}
  })
  if (!labels) return null

  return (
    <div ref={gridRef} className="grid">
      {labels.map((value) => {
        const labelId =
          'pin-label-' + value.path.replace(/\//g, '').replace('.', '')
        return (
          <div
            key={labelId}
            id={labelId}
            className="item "
            onClick={() => {
              setHiddenId(value.path)
            }}
          >
            <div className="item-content">
            {' '}
            {value.path.replace(/\//g, '')}
            <br />
            {value.answer ?? value.content}
              </div>
          </div>
        )
      })}
    </div>
  )
}

export { MuuriComponent }

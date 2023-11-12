import React, { useEffect, useRef } from 'react'
import Muuri from 'muuri'

function computeParallelLines(itemSize, margin) {
  const lines = []
  const lineHeight = itemSize.height + margin
  const maxY = window.innerHeight

  for (let y = itemSize.height/2; y < maxY; y += lineHeight) {
    lines.push(y)
  }

  return lines
}

function isPointWithinShape(point, shape) {
  switch (shape.type) {
    case 'circle':
      return isPointInCircle(point, shape)
    case 'triangle':
      return isPointInTriangle(point, shape.vertices)
    case 'rectangle':
      return isPointInRectangle(point, shape)
    default:
      throw `Unknown shape type: ${shape.type}`
  }
}

function isPointInRectangle(point, rectangle) {
  const [A, B, C, D] = rectangle.vertices
  const [x, y, width, height] = [A.x, A.y, C.x - A.x, C.y - A.y]
  return (
    point.x >= x &&
    point.x <= x + width &&
    point.y >= y &&
    point.y <= y + height
  )
}

function isPointInTriangle(p, triangle) {
  const [A, B, C] = triangle

  function sign(p1, p2, p3) {
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
  }

  const d1 = sign(p, A, B)
  const d2 = sign(p, B, C)
  const d3 = sign(p, C, A)

  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0

  return !(hasNeg && hasPos)
}

function isPointInCircle(point, circle) {
  const dx = point.x - circle.cx
  const dy = point.y - circle.cy

  return dx * dx + dy * dy <= circle.r * circle.r
}
function findFirstAvailablePositionInShapes(
  xStart,
  xEnd,
  y,
  itemSize,
  margin,
  shapes,
  occupiedPositions,
) {
  for (let x = xStart; x < xEnd; x += itemSize.width + margin) {
    const point = { x, y }
    const isOccupied = occupiedPositions.some(
      (pos) => pos.x === point.x && pos.y === point.y,
    )

    if (
      !isOccupied &&
      shapes.some((shape) => isPointWithinShape(point, shape))
    ) {
      occupiedPositions.push(point)
      return point
    }
  }
  return null
}

function positionItemsOnLines(lines, itemSize, margin, shapes, items) {
  const positions = []
  const occupiedPositions = []
  const xStart = itemSize.width / 2
  const xEnd = window.innerWidth - itemSize.width / 2

  console.log('positionItemsOnLines', {
    lines,
    itemSize,
    margin,
    shapes,
    items,
  })

  for (let item of items) {
    for (let lineY of lines) {
      const position = findFirstAvailablePositionInShapes(
        xStart,
        xEnd,
        lineY,
        itemSize,
        margin,
        shapes,
        occupiedPositions,
      )
      if (position) {
        positions.push(position)
        break // Proceed to next item
      }
    }
    if (positions.length === items.length) break // All items positioned
  }

  return positions
}

let POINTS = []

function layout(areas, cssPrefix) {
  return (grid, layoutId, items, width, height, callback) => {
    console.log(`css prefix: ${cssPrefix}`, items)
    let itemSize = { width: 0, height: 0 }
    try {
      itemSize = {
        width: items[0].getWidth(),
        height: items[0].getHeight(),
      }
    } catch (e) {
      console.error('Empty items in Muuri layout')
    }
    const margin = 0
    const lines = computeParallelLines(itemSize, margin)
    const shapes = Object.values(areas)

    const positions = positionItemsOnLines(
      lines,
      itemSize,
      margin,
      shapes,
      items,
    )
    console.log('layout', { layoutId, items, width, height, positions })

    items.forEach((item, index) => {
      if (index < positions.length) {
        const position = positions[index]
        item.getElement().style.left = `${position.x - itemSize.width / 2}px`
        item.getElement().style.top = `${position.y - itemSize.height / 2}px`
      }
    })
    POINTS = positions
    console.log('layout', { layoutId, items, width, height, positions })
    callback(layout)
  }
}

const ControlContainer = ({ children, areas, cssPrefix, debug = false }) => {
  const gridRef = useRef(null)

  useEffect(() => {
    if (!gridRef.current) return
    try {
      const grid = new Muuri(gridRef.current, {
        items: `.${cssPrefix}-item`,
        dragEnabled: true,
        rounding: true,
        layout: layout(areas, cssPrefix),
      })

      return () => {
        grid.destroy()
      }
    } catch (e) {
        console.error('Error in Muuri layout', e)
    }
  }, [areas, cssPrefix, gridRef, children])

  console.log('ControlContainer', { children, areas, cssPrefix, debug })
  const nonNullitems = children?.filter((child) => child)
    if (!nonNullitems) {
      console.error('No children in ControlContainer')
      return null
    }
  return (
    <>
      <div ref={gridRef} className="grid">
        {nonNullitems.map((child, i) => (
          <div
            key={i}
            className={`${cssPrefix}-item`}
            style={{ zIndex: 999999999999 }}
          >
            {child}
          </div>
        ))}
      </div>
      {debug && (
        <svg
          key={'name'}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: -10,
          }}
        >
          <text
            x={window.innerWidth / 2}
            y={window.innerHeight / 2}
            fill="red"
            fontSize="10px"
          >
            {JSON.stringify(POINTS)}
          </text>
          <text
            x={(window.innerWidth * 2) / 3}
            y={(window.innerHeight * 2) / 3}
            fill="red"
            fontSize="40px"
          >
            {cssPrefix}
          </text>

          {Object.entries(areas)
            .filter(([name, { type }]) => type !== 'circle')
            .map(([name, area], i) => (
              <polygon
                key={i}
                points={area.vertices?.map(({ x, y }) => `${x},${y}`).join(' ')}
                fill="rgba(50,233,10,0.3)"
              />
            ))}
          {Object.entries(areas)
            .filter(([name, { type }]) => type === 'circle')
            .map(([name, area], i) => (
              <circle
                key={i}
                cx={area.cx}
                cy={area.cy}
                r={area.r}
                fill="rgba(50,233,10,0.3)"
              />
            ))}

          {POINTS.map(({ x, y }, i) => (
            <circle key={i} cx={x} cy={y} r={5} fill="red" />
          ))}
        </svg>
      )}
    </>
  )
}

export { ControlContainer }

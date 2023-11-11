import React, { useEffect, useRef } from 'react'
import Muuri from 'muuri'
import '../muuri.css'

const areas = {
  leftTriangle: {
    type: 'triangle',

    vertices: [
      { x: 0, y: window.innerHeight * 2 / 3 }, // Bottom left
      { x: 0, y: 0 }, // Top left
      { x: window.innerWidth  /  3, y: 0 }, // Middle top
    ],
  },
  rightTriagnle: {
    type: 'triangle',

    vertices: [
      { x: window.innerWidth * 0.66, y: 0 }, // Middle top
      { x: window.innerWidth, y: 0 }, // Top right
      { x: window.innerWidth, y: window.innerHeight *2 / 3 }, // Bottom right
    ],
  },

  circle: {
    type: 'circle',
    cx: window.innerWidth / 2,
    cy: window.innerHeight / 2,
    r: 150,
  },
}

function computeParallelLines(itemSize, margin) {
  const lines = []
  const lineHeight = itemSize.height + margin
  const maxY = window.innerHeight

  for (let y = itemSize.height ; y < maxY; y += lineHeight) {
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
    default:
      return false // Or handle other shape types
  }
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
  occupiedPositions
) {
  for (let x = xStart; x < xEnd; x += itemSize.width + margin) {
    const point = { x, y };
    const isOccupied = occupiedPositions.some(
      pos => pos.x === point.x && pos.y === point.y
    );

    if (!isOccupied && shapes.some(shape => isPointWithinShape(point, shape))) {
      occupiedPositions.push(point); // Mark this position as occupied
      return point;
    }
  }
  return null;
}

function positionItemsOnLines(lines, itemSize, margin, shapes, items) {
  const positions = [];
  const occupiedPositions = []; // Array to keep track of occupied positions
  const xStart = itemSize.width /2 ;
  const xEnd = window.innerWidth - itemSize.width/2;

  for (let item of items) {
    for (let lineY of lines) {
      const position = findFirstAvailablePositionInShapes(
        xStart,
        xEnd,
        lineY,
        itemSize,
        margin,
        shapes,
        occupiedPositions
      );
      if (position) {
        positions.push(position);
        break; // Proceed to next item
      }
    }
    if (positions.length === items.length) break; // All items positioned
  }

  return positions;
}

let POINTS = []

function layout(grid, layoutId, items, width, height, callback) {
  const itemSize = {
    width: items[0].getWidth(),
    height: items[0].getHeight(),
  }
  const margin = 10 // adjust as needed
  const lines = computeParallelLines(itemSize, margin)
  const shapes = [areas.leftTriangle, areas.rightTriagnle] // define your shapes

  const positions = positionItemsOnLines(lines, itemSize, margin, shapes, items)

  items.forEach((item, index) => {
    console.log(item)
    if (index < positions.length) {
      const position = positions[index]
      item.getElement().style.left = `${position.x - itemSize.width/2}px`
      item.getElement().style.top = `${position.y-itemSize.height/2}px`
    }
  })
  POINTS = positions
  callback(layout)
}

const ControlContainer = ({ children }) => {
  const gridRef = useRef(null)

  useEffect(() => {
    if (!gridRef.current) return
    const grid = new Muuri(gridRef.current, {
      dragEnabled: true,
      rounding: true,
      layout,
    })

    return () => grid.destroy()
  }, [gridRef])

  return (
    <>
      <div ref={gridRef} className="grid">
        {children.map((child) => (
          <div className="item" style={{ zIndex: 999999999999 }}>
            {child}
          </div>
        ))}
      </div>
      {/*
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
        {Object.entries(areas)
          .filter(({ type }) => type !== 'circle')
          .map(([name, area]) => (
            <polygon
              points={area.vertices?.map(({ x, y }) => `${x},${y}`).join(' ')}
              fill="rgba(50,233,10,0.3)"
            />
          ))}

        {POINTS.map(({ x, y }) => (
          <circle cx={x} cy={y} r={5} fill="red" />
        ))}
      </svg>
      */}
      )
    </>
  )
}

export { ControlContainer }

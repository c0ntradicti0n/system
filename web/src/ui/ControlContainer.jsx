import React, { useEffect, useRef } from 'react'
import Muuri from 'muuri'
import '../../muuri.css'

const areas = {
  leftTriangle: {
    vertices: [
      { x: 0, y: window.innerHeight }, // Bottom left
      { x: 0, y: 0 }, // Top left
      { x: window.innerWidth / 2, y: 0 }, // Middle top
    ],
  },
  rightTriangle: {
    vertices: [
      { x: window.innerWidth, y: window.innerHeight }, // right left
      { x: window.innerWidth, y: 0 }, // Top right
      { x: window.innerWidth / 2, y: 0 }, // Middle top
    ],
  },
}

function calculatePositionInArea(item, area) {
  if (area.vertices) {
    // Assuming area is a triangle
    const [v1, v2, v3] = area.vertices

    // Randomly choose a point inside the triangle
    const r1 = Math.random()
    const r2 = Math.random()
    const sqrtR1 = Math.sqrt(r1)

    const x =
      (1 - sqrtR1) * v1.x + sqrtR1 * (1 - r2) * v2.x + sqrtR1 * r2 * v3.x
    const y =
      (1 - sqrtR1) * v1.y + sqrtR1 * (1 - r2) * v2.y + sqrtR1 * r2 * v3.y

    return { x, y }
  }
}

function determineAreaForItem(item, areas, index, totalItems) {
  // Example: Alternate between left and right triangles
  if (index % 2 === 0) {
    return areas.leftTriangle
  } else {
    return areas.rightTriangle
  }
}

function layout(grid) {
  const items = grid.getItems()

  items.forEach((item) => {
    // Decide in which area the item should be
    const area = determineAreaForItem(item, areas)

    // Calculate position based on the area
    const position = calculatePositionInArea(item, area)

    // Set the item's left and top CSS properties
    item.getElement().style.left = `${position.x}px`
    item.getElement().style.top = `${position.y}px`
  })

  // Update the grid after positioning items
  grid.refreshItems().layout()
}

const ControlContainer = ({ children }) => {
  const gridRef = useRef(null)

  useEffect(() => {
    if (!gridRef.current) return
    const grid = new Muuri(gridRef.current, {
      dragEnabled: true,
      rounding: true,
      width: 1000,
      layout,
    })

    return () => grid.destroy()
  }, [gridRef])

  return (
    <div ref={gridRef} className="grid">
      {children}
    </div>
  )
}

export { ControlContainer }

import React, { useEffect, useRef, useState } from 'react'
import Muuri from 'muuri'
import '../../puzzle.css'
import { stringToColour } from '../../lib/color'
import { postProcessTitle } from '../../lib/position'
import useMuuriGrid from '../../lib/useMuuriGrid'
import calculateFontSize from '../../lib/FontSize'
import { DEBUG } from '../../config/const'

// Define the base length for the largest triangle
const BASE_LENGTH = window.innerHeight * 0.5 // Modify as needed
let TRIANGLE_CENTERS = []

export const positionToId = (x, y) => {
  return TRIANGLE_CENTERS.sort(
    (a, b) =>
      Math.sqrt((a.x - x) ** 2 + (a.y - y) ** 2) -
      Math.sqrt((b.x - x) ** 2 + (b.y - y) ** 2),
  )[0].id
}
function idToPosition(id) {
  let x = 0 // Start from the bottom left corner
  let y = window.innerHeight * 0.5
  let power

  // Iterate through all characters in the ID
  for (let charIndex = 0; charIndex < id.length; charIndex++) {
    const char = id.charAt(charIndex)
    power = Math.pow(2, charIndex)

    // Adjust `x` and `y` based on the character
    if (char === '2') {
      x += BASE_LENGTH / power
    } else if (char === '3') {
      x += BASE_LENGTH / power / 2
      y -= BASE_LENGTH / power
    }
    if (charIndex > 0) {
      y += BASE_LENGTH / (power / 2) / 2
    }
    // No need to adjust for '1', since it means we keep the current column
  }
  return { x, y, power }
}
export const idToSize = (id) => {
  return `${BASE_LENGTH / Math.pow(2, id.length - 1)}px`
}
export const idToZIndex = (id) => {
  return 1000 * (id.length + 1)
}
// Custom layout function for Muuri
function layout(grid, layoutId, items, width, height, callback) {
  // Initialize the layout object
  const layout = {
    id: layoutId,
    items: items,
    slots: [],
    styles: null, // No specific styles needed for the grid itself
    layoutOnResize: true,
  }
  TRIANGLE_CENTERS = []

  // Calculate positions for each item
  items.forEach((item) => {
    const isDragging = item.isDragging()

    if (isDragging) {
      // For the item being dragged, don't change its position
      const x = parseFloat(item.getElement().style.left)
      const y = parseFloat(item.getElement().style.top)
      layout.slots.push(x, y)
      return
    }
    const id = item.getElement().getAttribute('id') // Assuming the ID is stored in a data attribute
    let { x, y, power } = idToPosition(id)

    const d_center = BASE_LENGTH / (power / 2) / 4
    let d_y = 0
    if (id.length < 3) {
      d_y = BASE_LENGTH / 5.2 / (id.length + 1)
    }
    TRIANGLE_CENTERS.push({
      id,
      x: x + d_center,
      y: y + d_center + d_y,
    })

    // Push the calculated position to the slots array
    layout.slots.push(x, y)
  })

  // Call the callback function with the computed layout
  callback(layout)
}

const muuriOptions = (size) => ({
  dragEnabled: true,
  layout: layout,

  rounding: true,
  itemClass: 'puzzle-item',
  containerClass: 'puzzle-container',
  itemDraggingClass: 'puzzle-item-dragging',
  itemReleasingClass: 'puzzle-item-releasing',
  itemVisibleClass: 'puzzle-item-visible',
  itemHiddenClass: 'puzzle-item-hidden',
  itemPositioningClass: 'puzzle-item-positioning',
  itemPlaceholderClass: 'puzzle-item-placeholder',
  dragSortPredicate: function (item, e) {
    return Muuri.ItemDrag.defaultSortPredicate(item, {
      action: 'swap',
      threshold: 75,
    })
  },
  dragContainer: document.body,
  size: size,
})

const MutableTriangle = ({
  nest,
  fullId,
  data,
  _key,
  size,
  level = 0,
  action = null,
}) => {
  const isLeafNode =
    data && !Object.values(data).some((value) => typeof value === 'object')
  const title = postProcessTitle(data['.'])
  const { shortTitle, fontSize } = calculateFontSize(
    size,
    title?.slice(0, 100),
    1,
  )

  return (
    <>
      {!isLeafNode && data && (
        <>
          <MutableTriangle
            nest={nest + 1}
            fullId={fullId + '3'}
            data={data[3] ?? { '.': '' }}
            _key={fullId + '3'}
            size={size / 2}
            action={action}
          />

          <MutableTriangle
            nest={nest + 1}
            fullId={fullId + '2'}
            data={data[2] ?? { '.': '' }}
            _key={fullId + '2'}
            size={size / 2}
            action={action}
          />

          <MutableTriangle
            nest={nest + 1}
            fullId={fullId + '1'}
            data={data[1] ?? { '.': '' }}
            _key={fullId + '1'}
            size={size / 2}
            action={action}
          />
        </>
      )}
      <div
        className="triangle puzzle-item"
        style={{
          backgroundColor: stringToColour(fullId),
          height: idToSize(fullId),
          width: idToSize(fullId),
          zIndex: idToZIndex(fullId),
        }}
        key={_key}
        id={fullId}
      >
        <div
          style={{
            position: 'absolute',
            top: size / 2,
            left: size / 2,
            textAlign: 'left',
            maxWidth: `${size / 4}px`,
            zIndex: 1000 * nest,
            fontSize: (size / 10).toString() + 'px',
          }}
        >
          <div
            className="puzzle-item-content triangle-content"
            style={{
              fontSize,
              color: 'black',
              whiteSpace: 'pre-wrap',
              overflowWrap: 'break-word',
              width: size,
              transform: 'translateX(-25%)',
              zIndex: 1000000000 - 1000 * nest,
            }}
            title={title} // Full title as a tooltip
            onMouseDown={(e) => {
              e.preventDefault()
              e.stopPropagation()

              if (action !== null) {
                const nodeId = /\[([\d.a-zA-Z]+)]/.exec(shortTitle)[1]
                console.log('perform action', fullId, nodeId, shortTitle)

                action(nodeId)
              }
            }}
          >
            <div
              style={{ zIndex: level + 100000 * nest, position: 'relative' }}
            >
              {_key}{' '}
            </div>
            <span title={title}>{shortTitle}</span>{' '}
          </div>
        </div>
      </div>
    </>
  )
}

export const Puzzle = ({ data = undefined, action = null, props }) => {
  const gridRef = useRef(null)
  const [debug, setDebug] = useState(() => process.env['DEBUG'])

  const [items, setItems] = useState(JSON.parse(JSON.stringify(data)))

  useEffect(() => {
    setItems(JSON.parse(JSON.stringify(data)))
  }, [data])

  useMuuriGrid(
    gridRef,
    muuriOptions(BASE_LENGTH, 1),

    BASE_LENGTH,
    props,
  )

  if (!items) return null

  console.log('Puzzle', { items })

  return (
    <>
      <div ref={gridRef} className="puzzle-grid" id="#puzzle-drag-container">
        {Object.entries(items)
          .sort(([a], [b]) => b - a)
          .reverse()
          .map(([key, item]) => (
            <MutableTriangle
              key={key}
              nest={2}
              fullId={key}
              data={item}
              _key={key}
              size={BASE_LENGTH}
              action={action}
            />
          ))}
      </div>
      {DEBUG &&
        TRIANGLE_CENTERS.map(({ id, x, y }) => (
          <div
            key={id}
            style={{
              position: 'fixed',
              top: y,
              left: x,
              textAlign: 'left',
              border: '1px solid lime',
              color: 'lime',
              fontSize: '10px',
              zIndex: 1000000,
            }}
          >
            {id}
          </div>
        ))}
    </>
  )
}

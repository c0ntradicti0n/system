import React, { useEffect, useRef, useState } from 'react'
import Muuri from 'muuri'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'
import './puzzle.css'
import { stringToColour } from '../../lib/color'
import { pointInTriangle, postProcessTitle } from '../../lib/position'
import useMuuriGrid from '../../lib/useMuuriGrid'
import calculateFontSize from '../../lib/FontSize'

import { DEBUG } from '../../config/const'

const handleInteractionStart = (event) => {
  let mouseX, mouseY
  if (event.touches?.length) {
    const touch = event.touches[0]

    // Get the touch coordinates
    mouseX = touch.clientX
    mouseY = touch.clientY
  } else {
    mouseX = event.clientX
    mouseY = event.clientY
  }
  console.log('handleInteractionStart', mouseX, mouseY)
  if (mouseX === undefined || mouseY === undefined) return

  const elements = document
    .elementsFromPoint(mouseX, mouseY)
    .filter((e) => e.matches('.puzzle-item'))

  for (let element of elements) {
    const rect = element.getBoundingClientRect()

    const ax = rect.left
    const ay = rect.bottom
    const bx = rect.right
    const by = rect.bottom
    const cx = rect.left + rect.width / 2
    const cy = rect.top

    if (pointInTriangle(mouseX, mouseY, ax, ay, bx, by, cx, cy)) {
      event.preventDefault()
      event.stopPropagation()
    }
  }
}

const handleInterActionElement = (element, event) => {
  let mouseX, mouseY
  if (event.touches?.length) {
    const touch = event.touches[0]

    // Get the touch coordinates
    mouseX = touch.clientX
    mouseY = touch.clientY
  } else {
    mouseX = event.clientX
    mouseY = event.clientY
  }

  const rect = element.getBoundingClientRect()
  const ax = rect.left
  const ay = rect.bottom
  const bx = rect.right
  const by = rect.bottom
  const cx = rect.left + rect.width / 2
  const cy = rect.top

  const id = element.getAttribute('id') // Assuming the ID is stored in a data attribute

  if (pointInTriangle(mouseX, mouseY, ax, ay, bx, by, cx, cy)) {
    console.log('inside', id)
    return true
  }
  console.log('outside', id)
  return false
}
// Define the base length for the largest triangle
const max_length = Math.min(window.innerWidth, window.innerHeight)

const BASE_LENGTH = max_length * 0.5 // Modify as needed
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
  const origin_x = (window.innerWidth - max_length) / 2
  const origin_y = (window.innerHeight - max_length) / 2

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
      x: origin_x + x + d_center,
      y: origin_y + y + d_center + d_y,
    })

    // Push the calculated position to the slots array
    layout.slots.push(origin_x + x, origin_y + y)
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
  dragStartPredicate: (item, event) => {
    console.log('dragStartPredicate', item, event)
    return handleInterActionElement(item.getElement(), event)
  },
  dragSortPredicate: function (item, event) {
    // sort by size and if mouse event was in element
    //const id = item.getElement().getAttribute('id') // Assuming the ID is stored in a data attribute
    //console.log("dragSortPredicate", id, event, id.length)
    return Muuri.ItemDrag.defaultSortPredicate(item, {
      //index: id.length,
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
    0.7,
  )

  return (
    <>
      <svg
        className=" puzzle-item"
        style={{
          height: idToSize(fullId),
          width: idToSize(fullId),

          zIndex: fullId.length,

          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          dominantBaseline: 'middle',
          textAnchor: 'middle',

          clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)',
          pointerEvents: 'visiblePainted',
          backgroundColor: stringToColour(fullId, 0.5),
        }}
        key={_key}
        id={fullId}
        onMouseDown={(e) => {
          if (action !== null) {
            const nodeId = /\[([\d.a-zA-Z]+)]/.exec(shortTitle)[1]
            console.log('perform action', fullId, nodeId, shortTitle)

            action(nodeId)
          }
        }}
        title={title} // Full title as a tooltip
      >
        <foreignObject class="node" width="100%" height="100%">
          <div
            className="triangle-text dragHandle"
            style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              width: '100%',
              zIndex: 10000 - fullId.length,
              transform: 'translateY(10%)',
              fontSize,
              textAlign: 'justify',
              position: 'relative',
            }}
          >
            <div className="top-left"></div>
            <div className="top-right"></div>

            <p title={title}>{shortTitle}</p>
          </div>
        </foreignObject>
      </svg>
      {!isLeafNode && data && (
        <>
          <MutableTriangle
            nest={nest + 1}
            fullId={fullId + '3'}
            data={data[3] ?? { '.': '' }}
            _key={fullId + '3'}
            size={size / 2}
            action={action}
            //  zIndex={idToZIndex(fullId + '3')}
          />
          <MutableTriangle
            nest={nest + 1}
            fullId={fullId + '2'}
            data={data[2] ?? { '.': '' }}
            _key={fullId + '2'}
            size={size / 2}
            action={action}
            //  zIndex={idToZIndex(fullId + '2')}
          />
          <MutableTriangle
            nest={nest + 1}
            fullId={fullId + '1'}
            data={data[1] ?? { '.': '' }}
            _key={fullId + '1'}
            size={size / 2}
            action={action}
            // zIndex={idToZIndex(fullId + '1')}
          />
        </>
      )}
    </>
  )
}

export const Puzzle = ({ data = undefined, action = null, props }) => {
  const gridRef = useRef(null)
  const [items, setItems] = useState(JSON.parse(JSON.stringify(data)))

  useEffect(() => {
    setItems(JSON.parse(JSON.stringify(data)))
  }, [data])

  useMuuriGrid(gridRef, muuriOptions(BASE_LENGTH, 1), BASE_LENGTH, {
    ...props,
    action,
  })

  if (!items) return null

  return (
    <div className="puzzle">
      <TransformWrapper
        initialScale={1}
        initialPositionX={0}
        initialPositionY={0}
        centerOnInit
        options={{ centerContent: true, limitToBounds: false }}
      >
        {({ zoomIn, zoomOut, resetTransform, ...rest }) => (
          <>
            <TransformComponent>
              <div
                ref={gridRef}
                className="puzzle-grid"
                id="#puzzle-drag-container"
                onMouseDown={handleInteractionStart}
                onMouseDownCapture={handleInteractionStart}
                onTouchStartCapture={handleInteractionStart}
              >
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
              {/* <EventBroadcaster className={'puzzle-item'} predicate={handleInteractionStart2}>
                  </EventBroadcaster>*/}
            </TransformComponent>
          </>
        )}
      </TransformWrapper>

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
            }}
          >
            {id}
          </div>
        ))}
    </div>
  )
}

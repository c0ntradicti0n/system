import React, { useEffect, useRef, useState } from 'react'
import Muuri from 'muuri'
import '../puzzle.css'
import { stringToColour } from '../lib/color'
import { postProcessTitle } from '../lib/position'
import useMuuriStore from '../lib/MuuriStore'
import useMuuriGrid from '../lib/useMuuriGrid'
import jsonPatch from 'fast-json-patch'
import calculateFontSize from '../lib/FontSize'

const SIZE = 300

const moptions = (size, nest, all_muris) => ({
  dragEnabled: true,
  layout: function (grid, layoutId, items, width, height, callback) {
    var layout = {
      id: layoutId,
      items: items,
      slots: [],
      styles: {},
    }
    console.log('layout', size, size / SIZE, Math.pow(2, nest))

    // Size of the equilateral triangle

    for (var i = 0; i < items.length; i++) {
      var x, y

      // For the first item, place it at the top middle
      if (i === 0) {
        x = 0.5 * size * (1 / nest)
        y = 0
      }
      // For the second item, place it at the bottom left
      else if (i === 1) {
        x = size * (1 / nest)
        y = size * (1 / nest)
      }
      // For the third item, place it at the bottom right
      else if (i === 2) {
        x = 0
        y = size * (1 / nest)
      }

      layout.slots.push(x, y)
      //console.log('layout', { x, y })
    }

    // Call the callback function to apply the layout.
    callback(layout)
  },
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
  dragStartPredicate: (item, event) => {
    try {
      event.srcEvent.stopPropagation()
      return Muuri.ItemDrag.defaultStartPredicate(item, event)
    } catch (e) {
      console.log('dragStartPredicate', e)
    }
  },
  dragSort: () => {
    console.log('dragSort', all_muris)
    return all_muris
  },
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
  const gridRef = useRef(null)
  const { addInstance, removeInstance, muuriInstances } = useMuuriStore()

  //console.log('Triangle', { data, _key, fullId, size })

  useMuuriGrid(
    gridRef,
    moptions(size, 2, muuriInstances),
    addInstance,
    removeInstance,
    size,
  )

  const isLeafNode =
    data && !Object.values(data).some((value) => typeof value === 'object')
  const title = postProcessTitle(data['.'])?.slice(0, 100)
  const { shortTitle, fontSize } = calculateFontSize(size, title, 2)

  return (
    <div
      className="triangle puzzle-item"
      style={{
        backgroundColor: stringToColour(_key),

        height: size.toString() + 'px',
        width: size.toString() + 'px',
        zIndex: 1,
      }}
      key={_key}
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
          }}
        >
          <div style={{ zIndex: level + 1000 * nest, position: 'relative' }}>
            {_key}{' '}
          </div>
          {shortTitle}
        </div>
      </div>
      {!isLeafNode && data && (
        <div
          ref={gridRef}
          style={{ zIndex: nest + 10 }}
          className="puzzle-grid"
          onMouseDown={(e) => {
            e.preventDefault()
            e.stopPropagation()
            console.log('ACTION !!!', fullId, action)

            if (action !== null) {
              console.log('do action !!!', fullId, action)
              action(fullId)
            } else console.log('haLLOOOOO', fullId, action)
          }}
        >
          {
            <MutableTriangle
              nest={nest + 1}
              fullId={fullId + '3'}
              data={data[3] ?? { '.': '' }}
              _key={fullId + '3'}
              size={size / 2}
              action={action}
            />
          }
          {
            <MutableTriangle
              nest={nest + 1}
              fullId={fullId + '2'}
              data={data[2] ?? { '.': '' }}
              _key={fullId + '2'}
              size={size / 2}
              action={action}
            />
          }
          {
            <MutableTriangle
              nest={nest + 1}
              fullId={fullId + '1'}
              data={data[1] ?? { '.': '' }}
              _key={fullId + '1'}
              size={size / 2}
              action={action}
            />
          }
        </div>
      )}
    </div>
  )
}

export const Puzzle = ({
  children,
  data = undefined,
  applyPatch,
  action = null,
  ...props
}) => {
  const gridRef = useRef(null)
  const { addInstance, removeInstance, muuriInstances } = useMuuriStore()

  const [items, setItems] = useState(JSON.parse(JSON.stringify(data)))

  useEffect(() => {
    setItems(JSON.parse(JSON.stringify(data)))
  }, [data])
  console.log('ACTION', action)

  useMuuriGrid(
    gridRef,
    moptions(SIZE, 1, muuriInstances),
    addInstance,
    removeInstance,
    SIZE,
  )

  if (!items) return null

  console.log('Puzzle', { items })

  return (
    <div ref={gridRef} className="puzzle-grid" id="#puzzle-drag-container">
      {Object.entries(items)
        .sort(([a], [b]) => b - a)
        .map(([key, item]) => (
          <MutableTriangle
            key={key}
            nest={2}
            fullId={key}
            data={item}
            _key={key}
            size={SIZE}
            action={action}
          />
        ))}
    </div>
  )
}

import React, { useEffect, useRef, useState } from 'react'
import Muuri from 'muuri'
import '../puzzle.css'
import { stringToColour } from '../lib/color'
import { postProcessTitle } from '../lib/position'
import useMuuriStore from '../lib/MuuriStore'
import useMuuriGrid from '../lib/useMuuriGrid'
import jsonPatch from 'fast-json-patch'

const SIZE = 300
const  getPosition = (key, size, nest) => {
  ""
  "Calculate the position of the triangle based on its key."
  ""
  let x = 0
  let y = 0
  if (key === 1) {
    x = 0.5 * size * (1 / nest)
    y = 0
  }
  if (key === 2) {

    x = 0
    y = size * (1 / nest)
  }
  if (key === 3) {

    x = size * (1 / nest)
    y = size * (1 / nest)
  }

  return [x, y]
}
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
      console.log('layout ITEM', items[i].getElement())

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
      console.log('layout', { x, y })
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
    event.srcEvent.stopPropagation()
    return Muuri.ItemDrag.defaultStartPredicate(item, event)
  },
  dragSort: () => {
    console.log('dragSort', all_muris)
    return all_muris
  },
  size: size,
})

const MutableTriangle = ({ nest, fullId, data, _key, size }) => {
  const gridRef = useRef(null)
  const { addInstance, removeInstance, muuriInstances } = useMuuriStore()

  console.log('Triangle', { data, _key, fullId, size })

  useMuuriGrid(
    gridRef,
    moptions(size, 2, muuriInstances),
    addInstance,
    removeInstance,
    size,
  )

  const isLeafNode =
    data && !Object.values(data).some((value) => typeof value === 'object')

  return (
    <div
      className="triangle puzzle-item"
      style={{
        backgroundColor: stringToColour(_key),

        height: size.toString() + 'px',
        width: size.toString() + 'px',
        fontSize: (size / 10).toString() + 'px',
        zIndex: 1000 * nest,
      }}
      key={_key}
    >
      <div
        style={{
          position: 'absolute',
          top: size / 2,
          textAlign: 'center',
          width: size,
          fontSize: (size / 10).toString() + 'px',
        }}
      >
        {_key} <br />
        {postProcessTitle(data['.'])}
      </div>

      <div className="puzzle-item-content triangle-content"></div>
      {!isLeafNode && data && (
        <div ref={gridRef} className="puzzle-grid">
          {Object.entries(data).sort(([a], [b]) => b - a

          ).map(
            ([key, item]) =>
              data && key !== "." &&(
                <MutableTriangle
                  nest={nest + 1}
                  fullId={fullId + _key}
                  data={item}
                  _key={key}
                  size={size / 2}
                />
              ),
          )}
        </div>
      )}
    </div>
  )
}

export const Puzzle = ({
  children,
  data = undefined,
  applyPatch,
  ...props
}) => {
  const gridRef = useRef(null)
  const { addInstance, removeInstance, muuriInstances } = useMuuriStore()

  const [items, _setItems] = useState(
    data ?? {
        1: {
          '.': 'Be.md',
        },
        2: {
          '.': 'Nothing.md',
        },
        3: {
          1: {
            '.': 'Transition.md',
          },
          2: {
            '.': 'Coming-into-Being and Ceasing-to-Be.md',
          },
          3: {
            '.': 'Sublation.md',

            1: {
              '.': 'Becoming.md',
            },
            2: {
              '.': 'Sameness-Different.md',
            },
            3: {
              '.': 'Contradiction.md',
            },
          },

          '.': 'Becoming.md',
        },
        '.': 'Ontology.md',
      }
  )

  const setItems = (items) => {
    _setItems(items)
    const patch = jsonPatch.compare(data, items)
    applyPatch(patch)
  }

  useEffect(() => {
    if (data) {
      _setItems(data)
    }
  }, [data])

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
      {Object.entries(items).sort(([a], [b]) => b - a).map(([key, item]) => (
        <MutableTriangle
          key={key}
          nest={2}
          fullId={key}
          data={item}
          _key={key}
          size={SIZE}
        />
      ))}
    </div>
  )
}

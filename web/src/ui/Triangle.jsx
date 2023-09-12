import React, { useEffect, useCallback, useState } from 'react'
import {
  getLeftPosition,
  getTopPosition,
  isElementInViewportAndBigAndNoChildren,
  postProcessTitle,
} from '../lib/position'
import { addHoverObject, hoverObjects, removeHoverObject } from '../lib/hover'
import { MAX_LEVEL } from '../config/const'
import { stringToColour } from '../lib/color'
import useLinkedElementsStore from '../lib/PinnedElements'

function Triangle({
  id,
  detailId,
  setCurrentId,
  fullId,

  data,

  size,
  left,
  scale = null,
  level = MAX_LEVEL,

  top,

  transformState,
  setContent,
  setTooltipData,
  setHoverId,
}) {
  const [_hover, _setHover] = useState(false)
  const devicePixelRatio = window.devicePixelRatio || 1
  const fontSize = size / 30 / Math.log1p(devicePixelRatio)

  const ref = React.useRef(null)
  const triangleId = 'triangle-' + fullId.replace(/\//g, '')

  useEffect(() => {
    if (!ref?.current) return
    const isWithinViewport = isElementInViewportAndBigAndNoChildren(
      ref?.current,
    )

    if (isWithinViewport) {
      addHoverObject(fullId)
    } else {
      removeHoverObject(fullId)
    }
  }, [transformState, scale, fullId])

  const { linkedElements, linkedElementsEmpty, linkedElementsHas } = useLinkedElementsStore()

    console.log('linkedElements', linkedElements)

  if (typeof data === 'string' || !data) {
    const title = data
    return (
      <div
        ref={ref}
        id={fullId}
        style={{
          position: 'absolute',
          width: size,
          height: size,
          left: left,
          top: top,
        }}
      >
        <div
          className="triangle"
          style={{
            verticalAlign: 'middle',
            textAlign: 'center',
            border: '10px solid black !important',
            zIndex: 1000 - level,
          }}
          onClick={() => {
            console.log('click on ', data)
            setContent(data)
          }}
        >
          <div style={{ position: 'relative', top: size / 1.5 }}>
            {id && (
              <div
                id={'triangle-' + fullId.replace(/\//g, '')}
                style={{ fontSize }}
                className="triangle-title"
              >
                {postProcessTitle(title)}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }
  const hover = hoverObjects.has(fullId)
  if (hover) {
    setHoverId(fullId)
  }

  const title = data?.['.']
  const anto = data?.['_']

  return (
    <div
      ref={ref}
      id={fullId}
      style={{
        position: 'absolute',
        width: size,
        height: size,
        left: left,
        top: top,
        filter: _hover ? 'invert(1)' : 'invert(0)',
        transform: _hover ? 'scale(1.05)' : 'scale(1)', // this line scales the triangle up a bit on hover
        visibility: true
          ? 'visible'
          : linkedElementsHas(fullId)
          ? 'hidden'
          : 'visible',
      }}
      onClick={(e) => {
        console.log(scale)
        e.preventDefault()
      }}
      onMouseEnter={() => {
        addHoverObject(fullId)
      }}
      onMouseLeave={() => {
        removeHoverObject(fullId)
      }}
    >
      <div
        key={fullId}
        className="triangle"
        style={{
          verticalAlign: 'middle',
          textAlign: 'center',
          border: '10px solid black !important',
          backgroundColor: stringToColour(fullId.replace('/', ''), 1),
          zIndex: 1000 - level,
          filter: hover ? 'invert(1)' : 'invert(0)',
          display: 'flex',
          justifyContent: 'center' /* For horizontal alignment */,
          alignItems: 'center' /* For vertical alignment */,
          color: '#000',
        }}
      >
        {[1, 2, 3].map((subTriangleDir, index) => (
          <div key={fullId + '-' + index}>
            <Triangle
              fullId={`${fullId}/${subTriangleDir}`}
              id={subTriangleDir}
              key={fullId + '-' + index}
              data={data[subTriangleDir]}
              size={size / 2}
              left={getLeftPosition(index, size)}
              top={getTopPosition(index, size)}
              level={level + 1}
              {...{
                scale,
                setCurrentId,
                setContent,
                setTooltipData,
                setHoverId,
                index,
                detailId,
              }}
            />
          </div>
        ))}

        <div
          onClick={() => setTooltipData(fullId)}
          style={{
            position: 'absolute',
            top: '70%',
            transform: 'translateY(-50%)',
            fontSize,
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            zIndex: 10000,
            width: `${size / 3}px`,
          }}
          onMouseEnter={() => _setHover(true)}
          onMouseLeave={() => _setHover(false)}
        >
          {title && (
            <div>
              <div id={triangleId} className="triangle-title">
                {fullId.replace(/\//g, '.')}. {postProcessTitle(title)}
              </div>

              {postProcessTitle(anto)
                ?.split(/[\s-]+/)
                .map((word, index) => (
                  <div
                    key={index}
                    style={{
                      whiteSpace: 'pre-wrap',
                      overflowWrap: 'break-word',
                    }}
                  >
                    {word}
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Triangle

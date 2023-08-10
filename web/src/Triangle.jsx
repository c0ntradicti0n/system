import React, { useEffect, useState } from 'react'
import {
  getLeftPosition,
  getTopPosition,
  isElementInViewportAndBigAndNoChildren,
  postProcessTitle,
  stringToColour,
} from './position'
import { addHoverObject, hoverObjects, removeHoverObject } from './hover'

const MAX_LEVEL = 1

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
  const [hovered, setIsHovering] = useState(false)

  const fontSize = size / 30 + Math.sqrt(level)
  const ref = React.useRef(null)
  const fetchData = async () => {
    console.log('fetchData', fullId)
    setCurrentId(fullId)
  }

  useEffect(() => {
    if (!ref?.current) return
    const isWithinViewport = isElementInViewportAndBigAndNoChildren(
      ref?.current,
    )

    if (isWithinViewport) {
      fetchData()
    }
  }, [transformState, scale, fetchData])

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
              <div style={{ fontSize }} className="triangle-title">
                {postProcessTitle(title)}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (hoverObjects[hoverObjects.length - 1] === fullId) {
    setTooltipData(data)
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
      }}
      onClick={(e) => {
        console.log(scale)
        e.preventDefault()
      }}
      onMouseEnter={() => {
        addHoverObject(fullId)
        setIsHovering(true)
      }}
      onMouseLeave={() => {
        removeHoverObject(fullId)
        setIsHovering(false)
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
          filter: hovered ? 'invert(1)' : 'invert(0)',
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
          style={{
            position: 'absolute',
            top: '70%',
            transform: 'translateY(-50%)',
            fontSize,
            width: '100%',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
          }}
        >
          {fullId.slice(-1).replace(/\//g, '.')}.{' '}
          {title && (
            <>
              <div className="triangle-title">{postProcessTitle(title)}</div>

              {postProcessTitle(anto)
                ?.split(/[\s-]+/)
                .map((word, index) => (
                  <div
                    key={index}
                    style={{
                      fontFamily: 'serif',
                      whiteSpace: 'pre-wrap',
                      overflowWrap: 'break-word',
                    }}
                  >
                    {word}
                  </div>
                ))}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default Triangle

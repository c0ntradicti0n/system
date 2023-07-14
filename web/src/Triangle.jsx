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
    setTooltipData
}) {
  const [isHovering, setIsHovering] = useState(false)

  const fontSize = size/30
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
  }, [transformState, scale])

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

  if (      hoverObjects[hoverObjects.length - 1] === fullId)
      setTooltipData(data)
  const title = data?.['.']
  const anto = data?.['_']
if (hoverObjects[hoverObjects.length - 1] === fullId) {
  console.log(
    hoverObjects,
    fullId,
    hoverObjects[hoverObjects.length - 1] === fullId,
      data,
  )}

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
      onMouseEnter={(e) => {
        addHoverObject(fullId)
        setIsHovering(true)
      }}
      onMouseLeave={() => {
        removeHoverObject(fullId)
        setIsHovering(false)
      }}
    >
      <div
        className="triangle"
        style={{
          verticalAlign: 'middle',
          textAlign: 'center',
          border: '10px solid black !important',
          backgroundColor: stringToColour(fullId.replace('/', ''), 1),
          zIndex: 1000 - level,
        }}
      >
        {[1, 2, 3].map((subTriangleDir, index) => (
          <>
            <Triangle
              detailId={detailId}
              fullId={`${fullId}/${subTriangleDir}`}
              id={subTriangleDir}
              key={index}
              data={data[subTriangleDir]}
              index={index}
              size={size / 2}
              left={getLeftPosition(index, size)}
              top={getTopPosition(index, size)}
              level={level - 1}
              scale={scale}
              setCurrentId={setCurrentId}
              setContent={setContent}
              setTooltipData={setTooltipData}
            />
          </>
        ))}

        <div style={{ position: 'relative', top: size / 2, fontSize }}>
          <br /> {fullId.slice(1, 1000).replace(/[/]/g, '.')}.{' '}
          {title && (
            <div className="triangle-title">
              {postProcessTitle(title)} <br /> {postProcessTitle(anto)}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Triangle

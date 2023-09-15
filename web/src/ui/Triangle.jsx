import React, { useEffect, useState } from 'react'
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

function getRandomElement(arr) {
  const randomIndex = Math.floor(Math.random() * arr.length)
  return arr[randomIndex]
}
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
  animate,
}) {
  const [_hover, _setHover] = useState(false)
  const devicePixelRatio = window.devicePixelRatio || 1
  const fontSize = size / 30 / Math.log1p(devicePixelRatio)
  const [animationClass, setAnimationClass] = useState('')
  useEffect(() => {
    setAnimationClass(
      getRandomElement(['fold-foreground', 'fold-right-top', 'fold-left-top']),
    )
  }, [])
  const [animationTime] = useState(Math.random() / 4)
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
  if (!data) return null

  //const { linkedElements, linkedElementsHas } = useLinkedElementsStore()

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
        //filter: _hover ? 'invert(1)' : 'invert(0)',
        /*//transform: _hover ? 'scale(1.05)' : 'scale(1)', // this line scales the triangle up a bit on hover
        visibility: true
          ? 'visible'
          : linkedElementsHas(fullId)
          ? 'hidden'
          : 'visible',*/
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
        className={'triangle ' + (animate ? animationClass ?? '' : '')}
        onAnimationEnd={(div) => setAnimationClass(null)}
        style={{
          backgroundColor: stringToColour(fullId.replace('/', ''), 1),
          zIndex: 1000 - level,
          filter: hover ? 'invert(1)' : 'invert(0)',
          animationDuration: `${animationTime}s`,
        }}
      >
        {[1, 2, 3].map((subTriangleDir, index) =>
          data[subTriangleDir] ? (
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
                animate={animationClass === null}
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
          ) : null,
        )}

        <div
          className="triangle-content"
          onClick={() => setTooltipData(fullId)}
          style={{
            fontSize,

            width: `${size / 3}px`,
          }}
          onMouseEnter={() => _setHover(true)}
          onMouseLeave={() => _setHover(false)}
        >
          {title && (
            <div>
              <div>{animate ? animationClass ?? '' : ''}</div>
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

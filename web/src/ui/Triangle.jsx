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
import { trim } from '../lib/string'

function getRandomElement(arr) {
  const randomIndex = Math.floor(Math.random() * arr.length)
  return arr[randomIndex]
}

const isMobile =
  /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent,
  )

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
  setTooltipData,
  setHoverId,
  animate,
}) {
  const [_hover, _setHover] = useState(false)
  const devicePixelRatio = window.devicePixelRatio || 1
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
    if (!ref?.current || !isMobile) return
    const isWithinViewport = isElementInViewportAndBigAndNoChildren(
      ref?.current,
    )
    console.log(isWithinViewport)

    if (isWithinViewport) {
      console.log({ isWithinViewport, fullId, hoverObjects })

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
  // Calculate base font size
  let baseFontSize = size / 30 / Math.log1p(devicePixelRatio)
  const shortTitle = (postProcessTitle(title ?? '') ?? '')
    .split(' ')
    .slice(0, 100)
    .join(' ')
  // Check the combined text length
  const combinedTextLength = shortTitle.length

  // Adjust font size if the combined text length is more than 100 chars
  if (combinedTextLength > 50) {
    baseFontSize *= 0.5 // you can adjust this factor to your needs
  }

  const fontSize = baseFontSize

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
          backgroundColor: fullId
            ? stringToColour(fullId.replace('/', ''), 1)
            : 'black',
          zIndex: 1000 - level,
          filter: _hover ? 'invert(1)' : 'invert(0)',
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
            color: fullId ? 'black' : 'white',

            width: `${size / 3}px`,
          }}
          onMouseEnter={() => _setHover(true)}
          onMouseLeave={() => _setHover(false)}
        >
          {title && (
            <div>
              <div>{animate ? animationClass ?? '' : ''}</div>
              <div id={triangleId} className="triangle-title">
                {trim(fullId.replace(/\//g, '.'), '.')}{' '}
                {postProcessTitle(shortTitle.slice(0, 100))}
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

import React, { useEffect, useRef, useState, useMemo } from 'react'
import { useQuery } from 'react-query'
import Triangle from './Triangle'
import { mergeDeep, lookupDeep, shiftIn, shift } from './nesting'
import {
  TransformWrapper,
  TransformComponent,
  ReactZoomPanPinchRef,
} from 'react-zoom-pan-pinch'
import { Tooltips } from './Tooltips'
import { MAX_LEVEL } from './const'
import { MobileControls } from './MobileControls'

const maxZoomThreshold = 2
const minZoomThreshold = 0.6

function beamDataTo(
  newHiddenId,
  collectedData: {},
  hoverId,
  newRestId,
  detailId,
  transformComponentRef,
  setHiddenId,
  setVisualData,
) {
  const newData = lookupDeep(newHiddenId, collectedData)
  console.log('reset zoom in', {
    ids: {
      shiftOnNestedHoverSomething: [newHiddenId, newRestId],
      newHiddenId,
      detailId,
      hoverId,
    },
    newData,
    collectedData,
    transformComponentRef: transformComponentRef.current,
  })

  setHiddenId(newHiddenId || '')
  transformComponentRef.current.setTransform(0, 0, 1, 0, 0) // <-- reset the zoom
  setVisualData(newData)
}

const go = ({
  direction,
  hiddenId,
  setDetailId,
  collectedData,
  hoverId,
  detailId,
  transformComponentRef,
  setHiddenId,
  setVisualData,
}) => {
  const newHiddenId = shift(hiddenId, direction)
  console.log(hiddenId, hiddenId)
  setDetailId(newHiddenId || '')
  return beamDataTo(
    newHiddenId,
    collectedData,
    hoverId,
    '',
    detailId,
    transformComponentRef,
    setHiddenId,
    setVisualData,
  )
}

const Fractal = ({ setContent }) => {
  const [detailId, setDetailId] = useState(null)
  const [transformState, setTransformState] = useState(null)
  const [collectedData, setCollectedData] = useState({})
  const [visualData, setVisualData] = useState(null)
  const [hiddenId, setHiddenId] = useState('')
  const [scale, setScale] = useState(null)
  const size =
    window.innerHeight < window.innerWidth
      ? window.innerHeight
      : window.innerWidth
  const left = (window.innerWidth - size) * 0
  const top = (window.innerHeight - size) * 0.6
  const transformComponentRef = useRef<ReactZoomPanPinchRef | null>(null)
  const [tooltipData, setTooltipData] = useState(null)
  const [isWindowWide, setIsWindowWide] = useState(
    window.innerWidth > window.innerHeight,
  )
  const [hoverId, setHoverId] = useState(null)
  const params = useMemo(
    () => ({
      hiddenId,
      setDetailId,
      collectedData,
      hoverId,
      detailId,
      transformComponentRef,
      setHiddenId,
      setVisualData,
    }),
    [hiddenId, setDetailId, collectedData, detailId, hoverId],
  )
  useEffect(() => {
    const handleResize = () => {
      setIsWindowWide(window.innerWidth > window.innerHeight)
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  const fetchTree = async () => {
    const res = await fetch(`/api/toc/${detailId ?? ''}`)
    if (!res.ok) {
      console.error('Network response was not ok', res)
      return
    }
    const newData = await res.json()
    console.log('fetchFiles', { detailId, res, newData })

    const mergedData = mergeDeep(
      collectedData,
      newData[''] ? newData[''] : newData,
    )
    setCollectedData(mergedData)
    return mergedData
  }

  const { status, error } = useQuery(['triangle', detailId], fetchTree, {
    keepPreviousData: true,
  })

  useEffect(() => {
    const handleKeyDown = (e) => {
      switch (e.key) {
        case 'ArrowLeft':
          go({ ...params, direction: 'left' })
          break
        case 'ArrowRight':
          go({ ...params, direction: 'right' })
          break
        case 'ArrowUp':
          go({ ...params, direction: 'lower' })
          break
        case 'ArrowDown':
          go({ ...params, direction: 'higher' })
          break
        default:
          return
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [
    params,
    detailId,
    hoverId,
    transformComponentRef,
    collectedData,
    hiddenId,
  ])

  useEffect(() => {
    if (scale === null || detailId === null) return

    if (!hoverId) {
      console.error('hoverId is null', hoverId, detailId)
      //return
    }

    if (scale > maxZoomThreshold) {
      const [newHiddenId, newRestId] = shiftIn(detailId, hoverId ?? detailId, {
        left: true,
      })
      beamDataTo(
        newHiddenId,
        collectedData,
        hoverId,
        newRestId,
        detailId,
        transformComponentRef,
        setHiddenId,
        setVisualData,
      )
    }
    if (scale < minZoomThreshold) {
      console.log(detailId, hoverId)

      const [newHiddenId, newRestId] = shiftIn(detailId, hoverId ?? detailId, {
        left: false,
      })
      beamDataTo(
        newHiddenId,
        collectedData,
        hoverId,
        newRestId,
        detailId,
        transformComponentRef,
        setHiddenId,
        setVisualData,
      )
    }
  }, [scale, collectedData, detailId, hoverId])

  if (status === 'loading') {
    return <span>Loading...</span>
  }

  if (status === 'error') {
    return <span>{JSON.stringify(error)}</span>
  }

  return (
    <div
      className="App"
      style={{
        top: '0',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <TransformWrapper
        style={{ zIndex: 0 }}
        limitToBounds={false}
        maxScale={1000}
        minScale={0.01}
        ref={transformComponentRef}
        onTransformed={(ref) => {
          setTransformState(ref?.instance?.transformState)
          setScale(ref?.instance?.transformState?.scale)
        }}
      >
        {(utils) => (
          <React.Fragment>
            <MobileControls
              onLeft={() => go({ ...params, direction: 'left' })}
              onZoomIn={() => go({ ...params, direction: 'lower' })}
              onRight={() => go({ ...params, direction: 'right' })}
              onZoomOut={() => go({ ...params, direction: 'higher' })}
            />{' '}
            <TransformComponent>
              <div
                style={{
                  position: 'relative',
                  left,
                  top,
                  width: isWindowWide && tooltipData ? '70vw' : '100vw',
                  height: window.innerHeight,
                }}
              >
                <Triangle
                  id={''}
                  fullId={hiddenId}
                  data={visualData ?? collectedData}
                  size={size}
                  left={0}
                  top={0}
                  level={MAX_LEVEL}
                  setCurrentId={setDetailId}
                  {...{
                    hoverId,
                    setHoverId,
                    setContent,
                    detailId,
                    transformState,
                    scale,
                    setTooltipData,
                  }}
                />
              </div>
            </TransformComponent>
          </React.Fragment>
        )}
      </TransformWrapper>
      {tooltipData && (
        <Tooltips
          tree={collectedData}
          path={tooltipData}
          isWindowWide={isWindowWide}
        />
      )}
    </div>
  )
}

export default Fractal

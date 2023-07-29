import React, { useEffect, useRef, useState } from 'react'
import { useQuery } from 'react-query'
import Triangle from './Triangle'
import { mergeDeep, lookupDeep, splitKey, shiftIn } from './nesting'
import {
  TransformWrapper,
  TransformComponent,
  ReactZoomPanPinchRef,
} from 'react-zoom-pan-pinch'
import { Tooltips } from './Tooltips'

const Controls = ({ zoomIn, zoomOut, resetTransform }) => (
  <>
    <button onClick={() => zoomIn()}>+</button>
    <button onClick={() => zoomOut()}>-</button>
    <button onClick={() => resetTransform()}>x</button>
  </>
)

const maxZoomThreshold = 2
const minZoomThreshold = 0.6
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
  const left = (window.innerWidth - size) * 0.84
  const top = (window.innerHeight - size) * 0.84
  const transformComponentRef = useRef<ReactZoomPanPinchRef | null>(null)
  const [tooltipData, setTooltipData] = useState(null)
  const [isWindowWide, setIsWindowWide] = useState(
    window.innerWidth > window.innerHeight,
  )
  const [hoverId, setHoverId] = useState(null)

  useEffect(() => {
    const handleResize = () => {
      setIsWindowWide(window.innerWidth > window.innerHeight)
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  const fetchFiles = async () => {
    const res = await fetch(
      `${process.env['REACT_APP_HOST']}/api/${detailId ?? ''}`,
    )
    if (!res.ok) {
      throw new Error('Network response was not ok')
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

  const { status, error } = useQuery(['triangle', detailId], fetchFiles, {
    keepPreviousData: true,
  })

  useEffect(() => {
    if (scale === null || detailId === null) return
    if (scale > maxZoomThreshold) {
const [newHiddenId, newRestId] = shiftIn(detailId, hoverId, {left:true})
      const newData = lookupDeep(newHiddenId, collectedData)
      console.log('reset zoom out', {
        hoverId,
        shiftIn:[newHiddenId, newRestId],
        newData,
        detailId,
        collectedData,
        transformComponentRef: transformComponentRef.current,
      })

      setHiddenId(newHiddenId || '')
      transformComponentRef.current.setTransform(0, 0, 1, 0, 0) // <-- reset the zoom
      setVisualData(newData)
    }
    if (scale < minZoomThreshold) {

      console.log(detailId, hoverId)

      const [newHiddenId, newRestId] = shiftIn(detailId, hoverId, {left:false})
      const newData = lookupDeep(newHiddenId, collectedData)
      console.log('reset zoom out', {
        hoverId,
        shiftIn:[newHiddenId, newRestId],
        newData,
        detailId,
        collectedData,
        transformComponentRef: transformComponentRef.current,
      })

      setHiddenId(newHiddenId || '')
      transformComponentRef.current.setTransform(0, 0, 1, 0, 0) // <-- reset the zoom
      setVisualData(newData)
    }
  }, [scale])

  if (status === 'loading') {
    return <span>Loading...</span>
  }

  if (status === 'error') {
    return <span>Error: {error.message}</span>
  }

  //console.log(scale, detailId, collectedData)

  console.log('render', { hoverId })

  return (
    <div
      className="App"
      style={{
        top: '100',
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {tooltipData && (
        <Tooltips
          data={tooltipData}
          id={detailId}
          isWindowWide={isWindowWide}
        />
      )}
      <TransformWrapper
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
            <Controls {...utils} />
            <TransformComponent>
              <div
                style={{
                  position: 'relative',
                  left,
                  top,
                  width: window.innerWidth,
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
                  level={2}
                  setCurrentId={setDetailId}
                  {...{ hoverId, setHoverId, setContent, detailId, transformState, scale, setTooltipData }}
                />
              </div>
            </TransformComponent>
          </React.Fragment>
        )}
      </TransformWrapper>
    </div>
  )
}

export default Fractal

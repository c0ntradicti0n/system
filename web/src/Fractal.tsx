import React, { useEffect, useRef, useState, useMemo } from 'react'
import { useQuery } from 'react-query'
import Triangle from './Triangle'
import { mergeDeep, lookupDeep, shiftIn, shift, slashIt } from './nesting'
import {
  TransformWrapper,
  TransformComponent,
  ReactZoomPanPinchRef,
} from 'react-zoom-pan-pinch'
import { go, beamDataTo } from './navigate'

import { Tooltips } from './Tooltips'
import { MAX_LEVEL } from './const'
import { MobileControls } from './MobileControls'

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
  const [initialPageLoad, setInitialPageLoad] = useState(true) // New state to track initial page load

  useEffect(() => {
    if (initialPageLoad) {
      const hash = window.location.hash.substring(1) // Remove the '#' from the start
      const parsedId = slashIt(hash.split('/').join('')) // Convert "/1/3/2" to "132", adapt as needed
      setHiddenId(parsedId) // Update the hiddenId state
      setDetailId('')
      setTooltipData(parsedId) // Update the tooltipData state
      setInitialPageLoad(false) // Mark that the initial page load logic is done
      console.log('initial load', { hash, parsedId })
    }
  }, [initialPageLoad]) // Depend on initialPageLoad so that this useEffect runs only once

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
    const id = (hiddenId ?? '') + (detailId ?? '')
    const res = await fetch(`/api/toc/${id}`)
    if (!res.ok) {
      console.error('Network response was not ok', res)
      return
    }
    const newData = await res.json()
    console.log('fetch TOC', { detailId, res, newData })

    const mergedData = mergeDeep(
      collectedData,
      newData[''] ? newData[''] : newData,
    )
    setCollectedData(mergedData)
    if (hiddenId) {
      const initialVisualData = lookupDeep(id, collectedData)
      setVisualData(initialVisualData)
    }
    return mergedData
  }

  const { status, error } = useQuery(
    ['triangle', hiddenId + '/' + detailId],
    fetchTree,
    {
      // keepPreviousData: true,
      staleTime: 0, // Data will be considered stale immediately after it's fetched, forcing a refetch

      enabled: detailId !== null, // Only execute the query if detailId is not null
    },
  )

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
      console.log('WHY DIS', newHiddenId)

      setDetailId(newHiddenId) // Or set it to another appropriate value.
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
      console.log('WHY DIS', newHiddenId)
      setDetailId(newHiddenId) // Or set it to another appropriate value.
    }
  }, [scale, collectedData, detailId, hoverId])

  if (status === 'loading') {
    return <span>Loading...</span>
  }

  if (status === 'error') {
    return <span>{JSON.stringify(error)}</span>
  }
  const linkId = tooltipData !== '' ? tooltipData : hiddenId

  console.log({
    ids: {
      hiddenId,
      detailId,
      tooltipData,
      linkId,
    },
  })

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
                  setCurrentId={(id) => {
                    console.log('TRIANGLE SETTING DETAILID', id)
                    setDetailId(id)
                  }}
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
          path={tooltipData !== '' ? tooltipData : hiddenId}
          isWindowWide={isWindowWide}
        />
      )}
      <MobileControls
        onLeft={() => go({ ...params, direction: 'left' })}
        onZoomIn={() => go({ ...params, direction: 'lower' })}
        onRight={() => go({ ...params, direction: 'right' })}
        onZoomOut={() => go({ ...params, direction: 'higher' })}
        linkId={linkId}
      />{' '}
    </div>
  )
}

export default Fractal

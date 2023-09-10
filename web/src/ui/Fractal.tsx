import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useQuery } from 'react-query'
import Triangle from './Triangle'
import { mergeDeep, lookupDeep, shiftIn, slashIt } from '../lib/nesting'
import {
  TransformWrapper,
  TransformComponent,
  ReactZoomPanPinchRef,
} from 'react-zoom-pan-pinch'
import { go, beamDataTo } from '../lib/navigate'

import { Tooltips } from './Tooltips'
import { MAX_LEVEL } from '../config/const'
import { MobileControls } from './MobileControls'
import { MuuriComponent } from './MuuriComponent'

const maxZoomThreshold = 2
const minZoomThreshold = 0.6

function makeNoHorizon() {
  const elements = document.querySelectorAll('.react-transform-wrapper')
  elements.forEach((element) => {
    element.style.overflow = 'visible'
  })
}

const Fractal = ({ setContent }) => {
  const [detailId, _setDetailId] = useState(null)
  const [transformState, setTransformState] = useState(null)
  const [collectedData, setCollectedData] = useState({})
  const [visualData, setVisualData] = useState(null)
  const [hiddenId, _setHiddenId] = useState('')
  const setHiddenId = useCallback(
    (id) => {
      // replace multiple slashes with a single slash
      id = slashIt(id)
      _setHiddenId(id)
    },
    [_setHiddenId],
  )
  const setDetailId = useCallback(
    (id) => {
      // replace multiple slashes with a single slash
      id = slashIt(id)
      _setDetailId(id)
    },
    [_setDetailId],
  )

  const [searchText, setSearchText] = useState(null)

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
  const [hoverId, _setHoverId] = useState(null)
  const setHoverId = useCallback(
    (id) => {
      // replace multiple slashes with a single slash
      id = slashIt(id)
      _setHoverId(id)
    },
    [_setHoverId],
  )
  const [animationClass, setAnimationClass] = useState('')

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

  const searchCall = async () => {
    try {
      const response = await fetch('/api/search/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          string: searchText,
        }),
      })

      if (!response.ok) {
        console.error('Network response was not ok', response)
        return
      }

      return await response.json()
    } catch (error) {
      console.error('Error in making searchCall:', error)
    }
  }

  const {
    data: searchResults,
    status: statusSearch,
    error: errorSearch,
  } = useQuery(['search', searchText], searchCall)

  const fetchTree = async () => {
    const id = (hiddenId ?? '') + (detailId ?? '')
    const res = await fetch(`/api/toc/${id}`)
    if (!res.ok) {
      console.error('Network response was not ok', res)
      return
    }
    const newData = await res.json()

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

      setDetailId(newHiddenId) // Or set it to another appropriate value.
    }
    if (scale < minZoomThreshold) {
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
      setDetailId(newHiddenId) // Or set it to another appropriate value.
    }

    makeNoHorizon()
  }, [scale, collectedData, detailId, hoverId])

  if (status === 'loading') {
    return <span>Loading...</span>
  }

  if (status === 'error') {
    return <span>{JSON.stringify(error)}</span>
  }
  const linkId = tooltipData !== '' ? tooltipData : hiddenId

  const triggerAnimation = (searchText) => {
    setAnimationClass('fallAnimation')
    makeNoHorizon()
    setSearchText(searchText)
  }

  return (
    <div className="App" style={{}}>
      <div style={{ position: 'absolute', width: 0, height: 0, top: 0 }}>
        {searchResults ? (
          <MuuriComponent labels={searchResults} setHiddenId={setHiddenId} />
        ) : null}
      </div>
      <TransformWrapper
        ref={transformComponentRef}
        style={{ zIndex: 0 }}
        limitToBounds={false}
        maxScale={1000}
        overFlow="visible"
        minScale={0.01}
        onTransformed={(ref) => {
          setTransformState(ref?.instance?.transformState)
          setScale(ref?.instance?.transformState?.scale)
        }}
      >
        {(utils) => (
          <div
            className={animationClass}
            style={{ perspectiveOrigin: '0% 0%', top: 0 }}
          >
            <TransformComponent>
              <div
                style={{
                  position: 'relative',
                  left,
                  top: 0,
                  width: isWindowWide && tooltipData ? '70vw' : '100vw',
                  height: '100vh',
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
          </div>
        )}
      </TransformWrapper>
      <div className="right-container">
        <MobileControls
          triggerSearch={triggerAnimation}
          onLeft={() => go({ ...params, direction: 'left' })}
          onZoomIn={() => go({ ...params, direction: 'lower' })}
          onRight={() => go({ ...params, direction: 'right' })}
          onZoomOut={() => go({ ...params, direction: 'higher' })}
          linkId={linkId}
          isWindowWide={isWindowWide}
          labels={searchResults}
        />
        {tooltipData && (
          <Tooltips
            tree={collectedData}
            path={tooltipData !== '' ? tooltipData : hiddenId}
            isWindowWide={isWindowWide}
            setTooltipData={setTooltipData}
          />
        )}
      </div>
    </div>
  )
}

export default Fractal

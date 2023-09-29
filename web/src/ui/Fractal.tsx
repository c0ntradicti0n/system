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

function parseHash(hash) {
  const params = {}
  const pairs = (hash[0] === '#' ? hash.substr(1) : hash).split('&')

  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i].split('=')
    params[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1] || '')
  }

  return params
}

const Fractal = ({}) => {
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
    [hiddenId, setDetailId, collectedData, hoverId, detailId, setHiddenId],
  )
  const [initialPageLoad, setInitialPageLoad] = useState(true) // New state to track initial page load

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
      const response = await fetch(
        `/api/search?filter_path=${hiddenId.replace(/\//g, '') ?? ''}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            string: searchText,
          }),
        },
      )

      if (!response.ok) {
        console.error('Network response was not ok', response)
        return
      }

      return await response.json()
    } catch (error) {
      console.error('Error in making searchCall:', error)
    }
  }

  const { data: searchResults } = useQuery(
    ['search', searchText, hiddenId],
    searchCall,
  )

  const fetchTree = async () => {
    const id = hiddenId ?? ''
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
      staleTime: 0,
      initialData: null,
    },
  )

  useEffect(() => {
    console.log('useEffect', { scale, detailId, hoverId })
    if (scale === null || hoverId === null) return

    if (!hoverId) {
      console.error('hoverId is null', hoverId, detailId)
    }

    if (scale > maxZoomThreshold) {
      const [newHiddenId, newRestId] = shiftIn(hiddenId, hoverId ?? detailId, {
        left: true,
      })
      console.log('scale in', {
        scale,
        minZoomThreshold,
        maxZoomThreshold,
        newHiddenId,
        newRestId,
        detailId,
        hoverId,
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

      setHiddenId(newHiddenId)
      setDetailId('')
    }
    if (scale < minZoomThreshold) {
      const [newHiddenId, newRestId] = shiftIn(hiddenId, hoverId ?? detailId, {
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
      setHiddenId(newHiddenId)
      setDetailId('')
    }

    makeNoHorizon()
  }, [scale, collectedData, detailId, hoverId, setHiddenId, setDetailId])

  const linkInfo = {
    hiddenId,
    searchText,
    tooltipData: tooltipData?.replace(/\//g, ''),
  }

  const triggerAnimation = (searchText) => {
    setAnimationClass('fallAnimation')
    makeNoHorizon()
    setSearchText(searchText)
  }

  useEffect(() => {
    if (initialPageLoad) {
      const params = parseHash(window.location.hash)

      if (params.hiddenId !== undefined) {
        setHiddenId(slashIt(params.hiddenId))
      }
      if (params.searchText !== undefined) {
        console.log('searchTEXT', params)
        triggerAnimation(params.searchText)
      }
      if (params.tooltipData !== undefined) {
        setTooltipData(slashIt(params.tooltipData))
      }

      setInitialPageLoad(false) // Mark that the initial page load logic is done
    }
  }, [initialPageLoad, setDetailId, setHiddenId]) // Depend on initialPageLoad so that this useEffect runs only once
  console.log(visualData)
  return (
    <div className="Fractal" style={{}}>
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
                  width: '100vw',
                  height: '100vh',
                  perspective: '1000px',
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
                    detailId,
                    transformState,
                    scale,
                    setTooltipData,
                  }}
                  animate={true}
                />
              </div>
            </TransformComponent>
          </div>
        )}
      </TransformWrapper>
      <div className="right-container">
        <MobileControls
          searchText={searchText}
          triggerSearch={triggerAnimation}
          onLeft={() => go({ ...params, direction: 'left' })}
          onZoomIn={() => go({ ...params, direction: 'lower' })}
          onRight={() => go({ ...params, direction: 'right' })}
          onZoomOut={() => go({ ...params, direction: 'higher' })}
          linkInfo={linkInfo}
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

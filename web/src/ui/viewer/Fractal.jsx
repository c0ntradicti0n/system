import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useImperativeHandle,
} from 'react'
import { useQuery } from 'react-query'
import Triangle from './Triangle'
import { mergeDeep, lookupDeep, shiftIn, slashIt } from '../../lib/nesting'

import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'
import { beamDataTo } from '../../lib/navigate'

import { Tooltips } from './Tooltips'
import { MAX_LEVEL } from '../../config/const'
import { SearchResults } from './SearchResults'
import './fractal.css'

const maxZoomThreshold = 2
const minZoomThreshold = 0.6

function makeNoHorizon() {
  const elements = document.querySelectorAll('.react-transform-wrapper')
  elements.forEach((element) => {
    element.style.overflow = 'visible'
  })
}

const Fractal = React.forwardRef(
  (
    { PRESET_DATA = undefined, inParent, showTooltips, editData = null },
    ref,
  ) => {
    const [detailId, _setDetailId] = useState(null)
    const [searchResults, setSearchResults] = useState(null)
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

    const [scale, setScale] = useState(null)
    const size =
      window.innerHeight < window.innerWidth
        ? window.innerHeight
        : window.innerWidth
    const left = (window.innerWidth - size) * 0
    const transformComponentRef = useRef(null)

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
    const triggerSearch = async (searchText) => {
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

        setSearchResults(await response.json())
      } catch (error) {
        console.error('Error in making searchCall:', error)
      }
    }

    // Expose specific functionality to the parent component
    useImperativeHandle(
      ref,
      () => ({
        hiddenId,
        setDetailId,
        collectedData,
        hoverId,
        detailId,
        transformComponentRef,
        setHiddenId,
        setVisualData,
        triggerSearch,
      }),
      [
        hiddenId,
        setDetailId,
        collectedData,
        hoverId,
        detailId,
        setHiddenId,
        triggerSearch,
      ],
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
      let newData = null
      const id = hiddenId ?? ''

      if (!PRESET_DATA) {
        const res = await fetch(`/api/toc/${id}`)
        if (!res.ok) {
          console.error('Network response was not ok', res)
          return
        }
        newData = await res.json()
      } else {
        newData = PRESET_DATA
      }

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

    useQuery(['triangle', hiddenId + '/' + detailId], fetchTree, {
      staleTime: 0,
      initialData: null,
    })

    useEffect(() => {
      //console.log('useEffect', { scale, detailId, hoverId })
      if (scale === null || hoverId === null) return

      if (!hoverId) {
        console.error('hoverId is null', hoverId, detailId)
      }

      if (scale > maxZoomThreshold) {
        const [newHiddenId, newRestId] = shiftIn(
          hiddenId,
          hoverId ?? detailId,
          {
            left: true,
          },
        )
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
        const [newHiddenId, newRestId] = shiftIn(
          hiddenId,
          hoverId ?? detailId,
          {
            left: false,
          },
        )
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

    return (
      <div className="fractal">
        <div style={{ position: 'absolute', width: 0, height: 0, top: 0 }}>
          {searchResults ? (
            <SearchResults labels={searchResults} setHiddenId={setHiddenId} />
          ) : null}
        </div>
        <TransformWrapper
          ref={transformComponentRef}
          limitToBounds={false}
          maxScale={1000}
          overFlow="visible"
          minScale={0.01}
          onTransformed={(ref) => {
            setTransformState(ref?.instance?.transformState)
            setScale(ref?.instance?.transformState?.scale)
          }}
        >
          {() => (
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
                    width: '100%',
                    height: '100%',
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
                    editData={editData}
                  />
                </div>
              </TransformComponent>
            </div>
          )}
        </TransformWrapper>
        <div className="right-container">
          {showTooltips && tooltipData && (
            <Tooltips
              refill={!PRESET_DATA}
              tree={collectedData}
              path={tooltipData !== '' ? tooltipData : hiddenId}
              isWindowWide={isWindowWide}
              setTooltipData={setTooltipData}
            />
          )}
        </div>
      </div>
    )
  },
)

export default Fractal

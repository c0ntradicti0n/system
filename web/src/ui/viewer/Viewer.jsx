import { ControlContainer } from '../ControlContainer'
import { CONTROL_AREAS } from '../../config/areas'
import { ViewerMenu } from './ViewerMenu'
import React, { useEffect, useRef, useState } from 'react'
import Fractal from './Fractal'
import { parseHash } from '../../lib/read_link_params'
import { slashIt } from '../../lib/nesting'

export const Viewer = () => {
  const [searchText, setSearchText] = useState(null)
  const [path, setPath] = useState('111')
  const fractalRef = useRef(null)

  const linkInfo = {
    hiddenId: fractalRef?.current?.hiddenId,
    tooltipData: fractalRef?.current?.tooltipData?.replace(/\//g, ''),
  }

  useEffect(() => {
    const handleHashChange = () => {
      const params = parseHash(window.location.hash)

      if (params.hiddenId !== undefined) {
        fractalRef?.current?.setHiddenId(slashIt(params.hiddenId))
      }
      if (params.searchText !== undefined) {
        console.log('searchTEXT', params)
        fractalRef?.current?.triggerSearch(params.searchText)
      }
      if (params.tooltipData !== undefined) {
        fractalRef?.current?.setTooltipData(slashIt(params.tooltipData))
      }
    }

    // Listen for hash changes
    window.addEventListener('hashchange', handleHashChange)

    // Call the handler function in case the component mounts with a hash already in the URL
    handleHashChange()

    // Cleanup
    return () => {
      window.removeEventListener('hashchange', handleHashChange)
    }
  }, []) // Empty dependency array ensures this runs once on mount and once on unmount

  console.log('REF', fractalRef)
  return (
    <>
      <Fractal ref={fractalRef} />
      <ControlContainer areas={CONTROL_AREAS} cssPrefix="fractal">
        <ViewerMenu
          {...{
            searchText,
            linkInfo,
            path,
            fractalRef,
            setPath,
          }}
        />
      </ControlContainer>
    </>
  )
}

import { ViewerMenu } from './ViewerMenu'
import React, { useEffect, useRef, useState } from 'react'
import Fractal from './Fractal'
import { Toolbar } from './Toolbar'
import { parseHash } from '../../lib/read_link_params'
import { slashIt } from '../../lib/nesting'
import { useTheme } from '../../App'

export const Viewer = () => {
  const [searchText, setSearchText] = useState(null)
  const [path, setPath] = useState('1313')
  const [menuVisible, setMenuVisible] = useState(false)
  const fractalRef = useRef(null)
  const { lightMode } = useTheme()

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

    window.addEventListener('hashchange', handleHashChange)
    handleHashChange()
    return () => {
      window.removeEventListener('hashchange', handleHashChange)
    }
  }, [])

  return (
    <>
      <Fractal ref={fractalRef} showTooltips={true} pastel={lightMode} fontMultiplier={1} />
      <Toolbar {...{ menuVisible, setMenuVisible, fractalRef }} />
      <ViewerMenu
        visible={menuVisible}
        onClose={() => setMenuVisible(false)}
        {...{ searchText, linkInfo, fractalRef }}
      />
    </>
  )
}

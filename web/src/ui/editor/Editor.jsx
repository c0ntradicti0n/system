import React, { useState, useMemo, useEffect } from 'react'

import { LibraryView } from './LibraryView'
import { PuzzleView } from './PuzzleView'
import './editor.css'
import { useSocket } from '../../query/useSocket'
import { parseHash } from '../../lib/read_link_params'
import LibraryMenu from './LibraryMenu'
import { RIGHT_BIG_TRIANGLE } from '../../config/areas'
import { ControlContainer } from '../ControlContainer'

const conicColors = { '0%': '#87d068', '50%': '#ffe58f', '100%': '#ffccc7' }

const renderActiveTabContent = ({
  action,
  setAction,
  mods,
  deleteMod,
  resetMod,
  setHashId,
  setActiveTab,
  socket,
  hash,
  text,
  state,
  params,
  isPaused,
  setIsPaused,
  activeTab,
}) => {
  switch (activeTab) {
    case 'lib':
      return (
        <LibraryView
          {...{ mods, deleteMod, resetMod, setHashId, setActiveTab }}
        />
      )
    case 'puzzle':
      return (
        <PuzzleView
          {...{
            action,
            setAction,
            socket,
            hash,
            text,
            state,
            params,
            isPaused,
            setIsPaused,
            activeTab,
          }}
        />
      )
    // Add other cases for each tab
    default:
      return null
  }
}

export const Editor = () => {
  const [hashId, setHashId] = useState(null)
  const [activeTab, setActiveTab] = useState('lib')
  const [initialPageLoad, setInitialPageLoad] = useState(true)
  const {
    state,
    hash,
    mods,
    deleteMod,
    resetMod,
    i,
    meta,
    setIsPaused,
    isPaused,
    progress,
    status,
    text,
    params,
    socket,
  } = useSocket(hashId)
  const [action, _setAction] = useState(null)
  const setAction = (action) => {
    console.log('Setting action to', action)
    _setAction(() => action)
  }

  //console.log({ progress, state, params, meta, status, i })
  const sumPercent = useMemo(
    () => Object.values(progress ?? {}).reduce((a, b) => a + b, 0) / 3,
    [progress],
  )

  useEffect(() => {
    if (initialPageLoad) {
      const params = parseHash(window.location.hash)
      if (params.hash) {
        setHashId(params.hash)
      }
      if (params.activeTab) {
        setActiveTab(params.activeTab)
      }
      setInitialPageLoad(false)
    }
  })

  return (
    <div className="App" style={{ overflow: 'hidden' }}>
      {renderActiveTabContent({
        mods,
        deleteMod,
        resetMod,
        setHashId,
        setActiveTab,
        socket,
        hash,
        text,
        state,
        params,
        isPaused,
        setIsPaused,
        activeTab,
        action,
        setAction,
      })}
      <ControlContainer areas={RIGHT_BIG_TRIANGLE} cssPrefix="puzzle">
        <LibraryMenu
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          meta={meta}
          sumPercent={sumPercent}
          progress={progress}
          status={status}
          i={i}
          hash={hash}
          socket={socket}
          setAction={setAction}
          action={action}
          params={params}
          isPaused={isPaused}
          setIsPaused={setIsPaused}
          activeTab={activeTab}
        />
      </ControlContainer>
    </div>
  )
}

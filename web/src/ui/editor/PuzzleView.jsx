import React, { useState } from 'react'
import { Puzzle } from './Puzzle'
import PuzzleControls from './PuzzleControls'
import './puzzle-controls.css'

export function PuzzleView(props) {
  const [action, _setAction] = useState(null)
  const setAction = (action) => {
    console.log('Setting action to', action)
    _setAction(() => action)
  }

  return (
    <>
      <Puzzle data={props.state} action={action} props={props} />
      <PuzzleControls
        hash={props.hash}
        socket={props.socket}
        setAction={setAction}
        action={action}
        params={props.params}
        isPaused={props.isPaused}
        setIsPaused={props.setIsPaused}
        activeTab={props.activeTab}
      />
    </>
  )
}

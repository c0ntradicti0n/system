import React, { useState } from 'react'
import { Puzzle } from '../Puzzle'
import PuzzleControls from './Controls'
import { Button } from 'antd'

export function PuzzleView(props) {
  const [action, _setAction] = useState(null)
  const setAction = (action) => {
    console.log('Setting action to', action)
    _setAction(() => action)
  }
    console.log("ACTION", action)


  return (
    <div id="abc123">

      <Button
        style={{
          zIndex: 9999999,
          position: 'absolute',
          top: 0,
          left: 0,
        }}
      >
        {action === null ? 'Selecting' : 'Select Start Node'}
      </Button>
      <PuzzleControls
        hash={props.hash}
        socket={props.socket}
        setAction={setAction}
        action={action}
      />
      <Puzzle
        data={props.state}
        applyPatch={props.applyPatch}
        action={action}
      />
    </div>
  )
}

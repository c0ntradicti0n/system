import React, { useState } from 'react'
import { Button } from 'antd'

const SelectStartNodeButton = ({ setAction,action, socket, hash }) => {
  const [selectionMode, setSelectionMode] = useState(false)

  const handleButtonClick = () => {
    // Toggle selection mode
    setSelectionMode(!selectionMode)
    if (selectionMode) {
      setAction(null)
    } else {
      console.log('Setting action to handleTriangleClick')
      setAction(handleTriangleClick)
    }
  }

  const handleTriangleClick = (start_node) => {
      // Send the hash via socket
      console.log('Sending start node hash', hash)
      socket.timeout(3000).emit('set_start_node', start_node, hash)

      // Exit selection mode
      setSelectionMode(false)

  }

  return (
      <>
    <Button
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
                    zIndex: 9999999,

      }}
      onClick={handleButtonClick}
      className="button"
      aria-label="Select Start Node"
      title="Select Start Node"
    >
        🚩
    </Button>
      <Button
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '60px',
          zIndex: 9999999,
      }}
      onClick={() => {
        socket.timeout(3000).emit('set_start_node',null,  hash)
      }}
      className="button"
      aria-label="Remove Start Node"
      title="Remove Start Node"
    >
        🚫
    </Button>
          </>)
}

const PuzzleControls = ({ setAction, action, socket, hash }) => {
  return (
    <div>
      <SelectStartNodeButton
        setAction={setAction}
        action={action }
        socket={socket}
        hash={hash}
      />
    </div>
  )
}

export default PuzzleControls

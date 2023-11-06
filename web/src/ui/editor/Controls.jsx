import React, { useState } from 'react'
import { Button, Slider } from 'antd'
import * as PropTypes from 'prop-types'

const SelectStartNodeButton = ({ setAction, action, socket, hash, params }) => {
  const [selectionMode, setSelectionMode] = useState(false)
  const startNode = params?.startNode
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
    socket
      .timeout(3000)
      .emit('save_params', { ...params, startNode: start_node }, hash)

    // Exit selection mode
    setSelectionMode(false)
  }

  return (
    <div
      className="red"
style={{
    display: 'flex', // This line will make the child elements align in a row
    alignItems: 'center', // This will vertically center the items in the div
    justifyContent: 'space-between', // This will add space between the items
    position: 'fixed',
    top: 0,
    right: '0px',
    zIndex: 9999999,
  }}
    >
      start {startNode}
      <Button
        className=" button red"
        onClick={handleButtonClick}
        aria-label="Select Start Node"
        title="Select Start Node"
      >
        🚩
      </Button>
      <Button
        className=" button red"
        onClick={() => {
          socket
            .timeout(3000)
            .emit('save_params', { ...params, startNode: null }, hash)
        }}
        aria-label="Remove Start Node"
        title="Remove Start Node"
      >
        🚫
      </Button>
    </div>
  )
}
const SelectDepth = ({ socket, hash, params }) => {
  const onSliderChange = (depth) => {
    console.log('Sending depth', depth)
    // This function will be triggered when the slider value changes
    socket.timeout(3000).emit('save_params', { ...params, depth }, hash)
  }

  const marks = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
  }

  return (
    <div
      className="red"
      style={{
        position: 'fixed',
        top: '0px',
        left: '100px',
        width: '20%',
        zIndex: 9999999,
      }}
    >
      Depth {params?.depth}
      <Slider
        min={1}
        max={10}
        marks={marks}
        onChange={onSliderChange}
        value={params?.depth ?? 3}
        tooltipVisible={false} // Hide tooltip if you don't need it
      />
    </div>
  )
}

const PuzzleControls = ({ setAction, action, socket, hash, params }) => {
  return (
    <div>
      <div className=" button red" style={{ color: 'white !important' }}>
        {JSON.stringify(params ?? {}) ?? 'no params'}
      </div>
      <SelectStartNodeButton
        setAction={setAction}
        action={action}
        socket={socket}
        hash={hash}
        params={params}
      />
      <SelectDepth socket={socket} hash={hash} value={params} params={params} />
    </div>
  )
}

export default PuzzleControls

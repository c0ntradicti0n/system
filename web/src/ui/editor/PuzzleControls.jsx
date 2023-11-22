import React, { useState } from 'react'
import { Button, Menu, Popconfirm, Slider } from 'antd'
import { DeleteOutlined } from '@ant-design/icons'
import { ControlContainer } from '../ControlContainer'
import { CONTROL_AREAS, RIGHT_BIG_TRIANGLE } from '../../config/areas'

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
    <div className="red ">
      Base node {startNode}
      <Button
        className="button red "
        onClick={handleButtonClick}
        aria-label="Select Start Node"
        title="Select Start Node"
        style={{ marginLeft: '30px' }}
      >
        Set 🚩
      </Button>
      <Button
        className="button red "
        onClick={() => {
          socket
            .timeout(3000)
            .emit('save_params', { ...params, startNode: null }, hash)
        }}
        aria-label="Remove Start Node"
        title="Remove Start Node"
        style={{ marginLeft: '30px' }}
      >
        Unset 🚫
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
    <div className="red">
      Depth {params?.depth}
      <Slider
        min={1}
        max={10}
        marks={marks}
        onChange={onSliderChange}
        value={params?.depth ?? 3}
      />
    </div>
  )
}

const PauseButton = ({ isPaused, setIsPaused }) => {
  const handlePauseClick = () => {
    setIsPaused(!isPaused)
  }

  return (
    <Button
      className="button red"
      onClick={handlePauseClick}
      aria-label={isPaused ? 'Resume' : 'Pause'}
      title={isPaused ? 'Resume' : 'Pause'}
    >
      {isPaused ? '▶️' : '⏸️'}
    </Button>
  )
}

const GenericSlider = ({ label, value, min, max, step, onChange }) => {
  return (
    <div className="red" key={label}>
      <label>
        {label} = <span>{value}</span>
      </label>
      <Slider
        min={min}
        max={max}
        step={step}
        onChange={onChange}
        value={value}
      />
    </div>
  )
}

const UserInteractionMenu = ({ params, onDeleteAction }) => {
  return (
    <div
      className="user-interaction-menu red"
      style={{
        overflowY: 'scroll',
        height: '20vw',
      }}
    >
      User action record
      <Menu mode="vertical" className=" red">
        {(params?.actions ?? [])
          .map((action, index) => [action, index])
          .reverse()
          .map(([action, index]) => (
            <Menu.Item key={index} className="user-action red">
              {index}
              <Popconfirm
                title="Are you sure to delete this action?"
                onConfirm={() => onDeleteAction(index)}
                okText="Yes"
                cancelText="No"
              >
                <Button type="link" style={{ color: '#fff' }}>
                  <DeleteOutlined />
                </Button>
              </Popconfirm>
              {action.source}↦{JSON.stringify(action.target)}
            </Menu.Item>
          ))}
      </Menu>
    </div>
  )
}

const PuzzleControls = ({
  setAction,
  action,
  socket,
  hash,
  params,
  isPaused,
  setIsPaused,
}) => {
  const handleDeleteAction = (index) => {
    // Remove an action from the userActions array
    const updatedActions = [...params.actions]
    updatedActions.splice(index, 1)
    socket
      .timeout(3000)
      .emit('save_params', hash, { ...params, actions: updatedActions })
  }

  return (
    <ControlContainer areas={RIGHT_BIG_TRIANGLE} cssPrefix="puzzle">
      <SelectStartNodeButton
        setAction={setAction}
        action={action}
        socket={socket}
        hash={hash}
        params={params}
      />
      <SelectDepth socket={socket} hash={hash} value={params} params={params} />
      <PauseButton isPaused={isPaused} setIsPaused={setIsPaused} />
      <GenericSlider
        label="Similarity"
        value={params?.weight_similarity || 0}
        min={0}
        max={1}
        step={0.01}
        onChange={(value) =>
          socket
            .timeout(3000)
            .emit('save_params', { ...params, weight_similarity: value }, hash)
        }
      />
      <GenericSlider
        label="Subsumtion vs Threerarchy"
        value={params?.weight_vs || 0}
        min={0}
        max={1}
        step={0.01}
        onChange={(value) =>
          socket
            .timeout(3000)
            .emit('save_params', { ...params, weight_vs: value }, hash)
        }
      />
      <GenericSlider
        label="Normal Text Sequence"
        value={params?.weight_vs || 0}
        min={0}
        max={1}
        step={0.01}
        onChange={(value) =>
          socket
            .timeout(3000)
            .emit('save_params', { ...params, weight_sequence: value }, hash)
        }
      />
      <GenericSlider
        label="Importance of Text"
        value={params?.weight_vs || 0}
        min={0}
        max={1}
        step={0.01}
        onChange={(value) =>
          socket
            .timeout(3000)
            .emit('save_params', { ...params, weight_importance: value }, hash)
        }
      />
      <UserInteractionMenu
        params={params}
        onDeleteAction={handleDeleteAction}
      />
    </ControlContainer>
  )
}

export default PuzzleControls

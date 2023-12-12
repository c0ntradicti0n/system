import React, { useMemo, useState } from 'react'
import { Button, Menu, Popconfirm, Slider } from 'antd'
import {
  UndoOutlined,
  MenuOutlined,
  EditOutlined,
  OrderedListOutlined,
  PicLeftOutlined,
  SendOutlined,
  FallOutlined,
  PauseOutlined,
  HeatMapOutlined,
  FastBackwardOutlined,
  BranchesOutlined,
} from '@ant-design/icons'
import { ControlContainer } from '../ControlContainer'
import { RIGHT_BIG_TRIANGLE } from '../../config/areas'
import TextMetaModal from './TextMetaModal'
import ShareModal from '../ShareModal'
import SubMenu from 'antd/es/menu/SubMenu'
import { UnicodeIcon } from '../UnicodeIcon'

export const SelectStartNodeButton = ({
  setAction,
  action,
  socket,
  hash,
  params,
}) => {
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
    socket.emit('save_params', { ...params, startNode: start_node }, hash)

    // Exit selection mode
    setSelectionMode(false)
  }

  return (
    <>
      <Menu.Item key="startnode" style={{ textAlign: 'right' }}>
        <Button
          onClick={handleButtonClick}
          aria-label="Select Start Node"
          title="Select Start Node"
          style={{ backgroundColor: 'unset !important' }}
        >
          Set 🚩
        </Button>
      </Menu.Item>
      {params?.startNode && (
        <Menu.Item key="startnone" style={{ textAlign: 'right' }}>
          <Button
            onClick={() => {
              socket.emit('save_params', { ...params, startNode: null }, hash)
            }}
            aria-label="Remove Start Node"
            title="Remove Start Node"
          >
            Unset 🚫
          </Button>
        </Menu.Item>
      )}
    </>
  )
}

export const SelectDepth = ({ socket, hash, params }) => {
  const onSliderChange = (depth) => {
    console.log('Sending depth', depth)
    // This function will be triggered when the slider value changes
    socket.emit('save_params', { ...params, depth }, hash)
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
    <>
      <Menu.Item key="depthtile">{'Depth ' + params?.depth}</Menu.Item>
      <Menu.Item key="depth">
        <Slider
          min={1}
          max={10}
          marks={marks}
          onChange={onSliderChange}
          value={params?.depth ?? 3}
        />
      </Menu.Item>
    </>
  )
}

export const PauseButton = ({ isPaused, setIsPaused }) => {
  const handlePauseClick = () => {
    setIsPaused(!isPaused)
  }

  return (
    <Button
      onClick={handlePauseClick}
      aria-label={isPaused ? 'Resume' : 'Pause'}
      title={isPaused ? 'Resume' : 'Pause'}
    >
      {isPaused ? 'Create more triangles!' : 'Stop computation'}
    </Button>
  )
}

export const GenericSlider = ({ label, value, min, max, step, onChange }) => {
  return (
    <Slider min={min} max={max} step={step} onChange={onChange} value={value} />
  )
}

export const UserInteractionMenu = ({ params, onDeleteAction }) => {
  return (
    <>
      <Menu.Item key="useraction">{'User actions'}</Menu.Item>
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
                <UndoOutlined />
              </Button>
            </Popconfirm>
            {action.source}↦{JSON.stringify(action.target)}
          </Menu.Item>
        ))}
    </>
  )
}

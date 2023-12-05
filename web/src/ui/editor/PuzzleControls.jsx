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

const SelectStartNodeButton = ({ setAction, action, socket, hash, params }) => {
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

const SelectDepth = ({ socket, hash, params }) => {
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

const PauseButton = ({ isPaused, setIsPaused }) => {
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

const GenericSlider = ({ label, value, min, max, step, onChange }) => {
  return (
    <Slider min={min} max={max} step={step} onChange={onChange} value={value} />
  )
}

const UserInteractionMenu = ({ params, onDeleteAction }) => {
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

const PuzzleControls = ({
  setAction,
  action,
  socket,
  hash,
  params,
  activeTab,
  text,
  meta,
  isPaused,
  setIsPaused,
}) => {
  const [collapsed, setCollapsed] = useState(false)
  const handleDeleteAction = (index) => {
    const updatedActions = [...params.actions]
    updatedActions.splice(index, 1)
    socket
      .timeout(3000)
      .emit('save_params', hash, { ...params, actions: updatedActions })
  }

  return (
    <ControlContainer areas={RIGHT_BIG_TRIANGLE} cssPrefix="puzzle">
      <div>
        <Button
          type="primary"
          icon={<MenuOutlined />}
          onClick={() => setCollapsed(!collapsed)}
        />
        {!collapsed && (
          <Menu theme={'dark'} inlineCollapsed={collapsed} mode="vertical">
            <Menu.Item key="textMeta" icon={<PicLeftOutlined />}>
              <TextMetaModal
                text={text}
                meta={meta}
                socket={socket}
                hash={hash}
              />
            </Menu.Item>
            <Menu.Item key="share" icon={<SendOutlined />}>
              <ShareModal hash={hash} linkInfo={{ activeTab, hash }} />
            </Menu.Item>
            <Menu.Item key="reset" icon={<FastBackwardOutlined />}>
              <Button
                onClick={() => {
                  socket.emit('reset', hash)
                }}
                aria-label="Reset"
                title="Reset"
              >
                Reset
              </Button>
            </Menu.Item>
            <Menu.Item
              key="pauseButton"
              icon={isPaused ? <HeatMapOutlined /> : <PauseOutlined />}
            >
              <PauseButton isPaused={isPaused} setIsPaused={setIsPaused} />
            </Menu.Item>
            <Menu.Item type="divider" />
            <SubMenu
              key="startnode"
              title={`Start node ${params?.startNode ?? ''}`}
              icon={<BranchesOutlined />}
            >
              <SelectStartNodeButton
                setAction={setAction}
                action={action}
                socket={socket}
                hash={hash}
                params={params}
              />
            </SubMenu>
            <SubMenu
              title={`Depth ${params?.depth ?? ''}`}
              icon={<FallOutlined />}
            >
              <SelectDepth
                socket={socket}
                hash={hash}
                value={params}
                params={params}
              />
            </SubMenu>

            <SubMenu
              key="slider1"
              title="Subsumption weight"
              icon={<UnicodeIcon symbol={'⊃'} />}
            >
              <Menu.Item key="similaritySlider">
                <GenericSlider
                  label="Similarity"
                  value={params?.weight_similarity || 0}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(value) =>
                    socket
                      .timeout(3000)
                      .emit(
                        'save_params',
                        { ...params, weight_similarity: value },
                        hash,
                      )
                  }
                />
              </Menu.Item>
            </SubMenu>
            <SubMenu
              key="slider2"
              title="Dialectics weight"
              icon={<UnicodeIcon symbol={'ஃ'} />}
            >
              <Menu.Item key="subsumptionSlider">
                <GenericSlider
                  label="Subsumtion vs Threerarchy"
                  value={params?.weight_vs || 0}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(value) =>
                    socket
                      .timeout(3000)
                      .emit(
                        'save_params',
                        { ...params, weight_vs: value },
                        hash,
                      )
                  }
                />
              </Menu.Item>
            </SubMenu>
            <SubMenu
              key="slider3"
              title="Sequence weight"
              icon={<OrderedListOutlined />}
            >
              <Menu.Item key="sequenceSlider">
                <GenericSlider
                  label="Normal Text Sequence"
                  value={params?.weight_vs || 0}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(value) =>
                    socket
                      .timeout(3000)
                      .emit(
                        'save_params',
                        { ...params, weight_sequence: value },
                        hash,
                      )
                  }
                />
              </Menu.Item>
            </SubMenu>
            <Menu.Item type="divider" />
            <SubMenu title={'User action menu'} icon={<EditOutlined />}>
              <UserInteractionMenu
                params={params}
                onDeleteAction={handleDeleteAction}
              />
            </SubMenu>
          </Menu>
        )}
      </div>
    </ControlContainer>
  )
}

export default PuzzleControls

import React, { useState } from 'react'
import { Button, Menu, Progress, Steps } from 'antd'
import {
  BookOutlined,
  PuzzleOutlined,
  AppstoreOutlined,
  TreeOutlined,
  FileJsonOutlined,
  ReadOutlined,
  MenuOutlined,
  PicLeftOutlined,
  SendOutlined,
  FastBackwardOutlined,
  HeatMapOutlined,
  PauseOutlined,
  BranchesOutlined,
  FallOutlined,
  OrderedListOutlined,
  EditOutlined,
} from '@ant-design/icons'
import BibTeXViewer from './BibtexViewer'
import TextMetaModal from './TextMetaModal'
import ShareModal from '../ShareModal'
import SubMenu from 'antd/es/menu/SubMenu'
import { UnicodeIcon } from '../UnicodeIcon'
import {
  GenericSlider,
  PauseButton,
  SelectDepth,
  SelectStartNodeButton,
  UserInteractionMenu,
} from './PuzzleControls'

const LibraryMenu = ({
  activeTab,
  setActiveTab,
  meta,
  sumPercent,
  status,

  setAction,
  action,
  socket,
  hash,
  params,
  text,
  isPaused,
  setIsPaused,
}) => {
  const [collapsed, setCollapsed] = useState(true)
  const handleDeleteAction = (index) => {
    const updatedActions = [...params.actions]
    updatedActions.splice(index, 1)
    socket
      .timeout(3000)
      .emit('save_params', hash, { ...params, actions: updatedActions })
  }
  return (
    <div>
      <Button
        type="primary"
        icon={<BookOutlined />}
        onClick={() => setCollapsed(!collapsed)}
        style={{ width: '40vw' }}
      >
        <BibTeXViewer entry={meta} short setIsGood={() => null} isGood inline />
        <div style={{ maxWidth: '10vw', display: 'inline' }}>
          {hash && (
            <Progress
              style={{ maxWidth: '30%' }}
              size="small"
              percent={sumPercent * 100 ?? 100}
              strokeColor={{
                '0%': '#87d068',
                '50%': '#ffe58f',
                '100%': '#ffccc7',
              }}
              showInfo={false}
            />
          )}
        </div>
      </Button>
      {!collapsed && (
        <Menu
          onClick={({ key }) => setActiveTab(key)}
          selectedKeys={[activeTab]}
          mode="vertical"
          theme="dark"
          dropdownWidth={300}
        >
          <Menu.Item
            key="lib"
            icon={<BookOutlined />}
            onClick={() => setCollapsed(true)}
          >
            Library
          </Menu.Item>
          <Menu.Item
            key="puzzle"
            icon={<AppstoreOutlined />}
            onClick={() => setCollapsed(true)}
          >
            Puzzle
          </Menu.Item>
          <Menu.Item key="steps" style={{ height: 'fit-content' }}>
            <Steps
              direction="vertical"
              current={
                status === 'end'
                  ? 3
                  : status === 'syn'
                  ? 2
                  : status === 'ant'
                  ? 1
                  : 0
              }
            >
              <Steps.Step title="hyperonym" />
              <Steps.Step title="antonyms" />
              <Steps.Step title="(anti-/syn)-thesis" />
              <Steps.Step title="🏁" />
            </Steps>
          </Menu.Item>

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

          <SubMenu
            type="group"
            title="Puzzle settings"
            icon={<OrderedListOutlined />}
          >
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
          </SubMenu>
        </Menu>
      )}
    </div>
  )
}

export default LibraryMenu

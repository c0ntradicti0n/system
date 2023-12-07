import React, { useState, useMemo, useEffect } from 'react'

import { Progress, Steps, Tabs } from 'antd'
import { JsonView } from './JsonView'
import { TreeView } from './TreeView'
import { TriangleView } from './TriangleView'
import { ExperimentsView } from './ExperimentsView'
import { PuzzleView } from './PuzzleView'
import './controls.css'
import { useSocket } from '../../query/useSocket'
import { parseHash } from '../../lib/read_link_params'
import BibTeXViewer from "./BibtexViewer";

const conicColors = { '0%': '#87d068', '50%': '#ffe58f', '100%': '#ffccc7' }

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
  console.log({ progress, state, params, meta, status, i })
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
    <div className="App" style={{ overflowY: 'scroll' }}>
      <Tabs
        activeKey={activeTab}
        onChange={(key) => {
          setActiveTab(key)
        }}
        tabPosition={'right'}
        width={'100%'}
        items={[
          {
            key: 'lib',
            label: 'Library',
            children: (
              <ExperimentsView
                {...{ mods, deleteMod, resetMod, setHashId, setActiveTab }}
              />
            ),
          },

          {
            key: 'puzzle',
            label: 'Puzzle',
            children: (
              <PuzzleView
                {...{
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
            ),
          },
          {
            key: '3',
            label: 'Triangle',
            children: state && <TriangleView {...{ hash, text, state }} />,
          },
          {
            key: 'tree',
            label: 'Tree',
            children: <TreeView {...{ hash, text, state }} />,
          },
          {
            key: 'json',
            label: 'JSON',
            children: <JsonView {...{ hash, text, state }} />,
          },
          { key: 'bib', label: <BibTeXViewer entry={meta} setIsGood={() => null} isGood />,
          children: null}
        ]}
      />

      {hash && (
        <div
          style={{
            fontFamily: 'monospace',
            position: 'fixed',
            right: '20vw',
            top: 0,
          }}
        >
          epoch {i}
          <Progress
            width={100}
            percent={sumPercent * 100 ?? 100}
            strokeColor={conicColors}
          />
          <Steps
            style={{
              display: 'inline-flex',
            }}
            current={
              status === 'end'
                ? 3
                : status === 'syn'
                ? 2
                : status === 'ant'
                ? 1
                : 0
            }
            items={[
              {
                title: 'hyperonym',
              },
              {
                title: 'antonyms',
              },
              {
                title: '(anti-/syn)-thesis',
              },
              {
                title: '🏁',
              },
            ]}
          />
        </div>
      )}
    </div>
  )
}

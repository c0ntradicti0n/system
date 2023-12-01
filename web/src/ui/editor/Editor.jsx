import React, { useState, useEffect, useCallback, useRef } from 'react'
import io from 'socket.io-client'
import ShareModal from '../ShareModal'
import { parseHash } from '../../lib/read_link_params'
import jsonPatch from 'fast-json-patch'
import { Progress, Steps, Tabs } from 'antd'
import { JsonView } from './JsonView'
import { TreeView } from './TreeView'
import { TriangleView } from './TriangleView'
import { ExperimentsView } from './ExperimentsView'
import { PuzzleView } from './PuzzleView'
import TextModal from './TextModal'
import MetaModal from './MetaModal'
import { ControlContainer } from '../ControlContainer'
import { RIGHT_BOTTOM_CORNER } from '../../config/areas'
import './controls.css'

let taskId = null
const setTaskId = (tI) => (taskId = tI)
const conicColors = { '0%': '#87d068', '50%': '#ffe58f', '100%': '#ffccc7' }

localStorage.debug = '*'
let inited = false
let active_patching = false
let init_phase = false

export const Editor = () => {
  const [socket, setSocket] = useState(
    io('', { transports: ['websocket'], retries: 1 }),
  )
  const [state, setState] = useState(null)
  const [patch, _setPatch] = useState(null)
  const [mods, setMods] = useState(null)
  const [activeTab, setActiveTab] = useState('ex')
  const [params, setParams] = useState({})
  const [I, setI] = useState(0)
  const [status, setStatus] = useState('')
  const [percentages, setPercentages] = useState({})

  const [_text, _setText] = useState('')
  const [meta, setMeta] = useState('')
  const [isPaused, setIsPaused] = useState(false)

  const sumPercent = Object.values(percentages).reduce((a, b) => a + b, 0) / 3

  const [text, setText] = useState('')
  const [hash, setHash] = useState(null)
  const setPatch = (patch) => {
    if (!patch) return
    console.log('setPatch', patch, state)
    let newState
    try {
      newState = jsonPatch.applyPatch(state, patch).newDocument
    } catch (e) {
      console.log('setPatch error', e)
      if (hash) socket.timeout(3000).emit('get_state', hash)
    }
    _setPatch(patch)
    setState(newState)
  }

  const timeoutId = useRef(null)

  const checkAwaitState = useCallback(() => {
    if (taskId && !isPaused) {
      socket.timeout(3000).emit('patch_poll', taskId)

      // Clear previous timeout, just in case
      if (timeoutId.current || isPaused) {
        clearTimeout(timeoutId.current)
      }
      timeoutId.current = setTimeout(checkAwaitState, 1000) // Recursive call
    }
  }, [isPaused])

  useEffect(() => {
    const params = parseHash(window.location.hash)
    console.log(params)
    if (params.hash !== undefined && params.hash !== hash) {
      setHash(params.hash)
      if (!init_phase) {
        console.log('INIT FROM HASH')
        init_phase = true
        socket.emit('get_hash', params.hash)
      }
    }
    if (params.activeTab !== undefined && params.activeTab !== activeTab) {
      setActiveTab(params.activeTab)
    }
  }, [hash])

  useEffect(() => {
    if (hash && state && !active_patching && !isPaused && status !== 'end') {
      console.log('START PATCHING', hash, state)
      socket.timeout(3000).emit('update_state', hash)
      socket.timeout(3000).emit('get_params', hash)
      active_patching = true
    }
  }, [status, isPaused, hash, state])

  const applyPatch = (patch) => {
    console.log('FE patch', patch, state)
  }
  useEffect(() => {
    // Listen for the hash from the server after sending the text
    socket.on('set_hash', (hash) => {
      setHash(hash)
      setPatch(null)
      setState(null)
    })
    // Listen for the hash from the server after sending the text
    socket.on('set_text', (text) => {
      console.log('set_text', text)
      setText(text)
      _setText(text)
      setPatch(null)
      setState(null)
    })
    socket.on('set_params', (params) => {
      console.log('set_params', params)
      setParams(params)
    })
    socket.on('set_meta', (meta) => {
      console.log('set_meta', meta)
      setMeta(meta)
    })
    socket.on('set_mods', (mods) => {
      console.log('set_mods', mods)
      setMods(mods)
    })

    socket.on('set_task_id', (task_id) => {
      if (task_id !== taskId) {
        console.log('set_task_id', task_id)

        setTaskId(task_id)
        checkAwaitState()
      }
    })
    socket.on('initial_state', (data) => {
      console.log('initial_state', hash, data)
      socket.emit('get_state', hash)
    })
    socket.on('set_state', (state) => {
      console.log('set_state', state)
      setState(state)
    })

    socket.on('patch_receive', (result) => {
      if (result.status === 'SUCCESS') {
        console.log('patch_receive SUCCESS')
        setPatch(result.result.patch)
        setI(result.result.i)
        setStatus(result.result.status)
        setPercentages(result.result.percentages)
        active_patching = false
        setTaskId(null)
        if (!isPaused && hash && !isPaused && status !== 'end') {
          socket.timeout(7000).emit('update_state', hash)
          active_patching = true
        }
      }
    })

    socket.on('connect_error', (error) => {
      console.error('Connect Error:', error)
    })

    socket.on('error', (error) => {
      console.error('Error:', error)
    })

    socket.on('connect', () => {
      console.log('Connected to server')
    })

    socket.on('disconnect', (reason) => {
      console.log('Disconnected:', reason)

      inited = false
    })
    inited = true

    return () => {
      socket.off('initial_state')
      socket.off('state_patch')
      socket.off('set_hash')
      socket.off('set_text')
      socket.off('set_params')
      socket.off('set_meta')
      socket.off('set_mods')
      socket.off('set_task_id')
      socket.off('set_state')
      socket.off('patch_receive')
      socket.off('connect_error')
      socket.off('error')
      socket.off('connect')
      socket.off('disconnect')
    }
  }, [hash, state])

  useEffect(() => {
    if (text && hash) {
      socket.timeout(3000).emit('get_state', hash)
      socket.timeout(3000).emit('get_meta', hash)
    }
  }, [hash, text])
  useEffect(() => {
    socket.timeout(3000).emit('get_mods')
  }, [])

  const deleteMod = (hash) => {
    console.log('deleteMod', hash)
    socket.timeout(3000).emit('delete_mod', hash)
  }
  const resetMod = (hash) => {
    console.log('deleteMod', hash)
    socket.timeout(3000).emit('reset_mod', hash)
  }
  return (
    <div className="App" style={{ overflowY: 'scroll' }}>
      <ControlContainer areas={RIGHT_BOTTOM_CORNER} cssPrefix="editor">
        <TextModal
          socket={socket}
          text={text}
          setText={setText}
          _text={_text}
          _setText={_setText}
        />
        {hash && (
          <MetaModal
            socket={socket}
            meta={meta}
            setMeta={setMeta}
            hash={hash}
          />
        )}
        <ShareModal url={'/editor'} linkInfo={{ hash, activeTab }} />
      </ControlContainer>
      <Tabs
        activeKey={activeTab}
        onChange={(key) => {
          setActiveTab(key)
        }}
        tabPosition={'right'}
        width={'100%'}
        items={[
          {
            key: 'ex',
            label: 'Experiments',
            children: <ExperimentsView {...{ mods, deleteMod, resetMod }} />,
          },
          ...(hash
            ? [
                {
                  key: 'pz',
                  label: 'Puzzle',
                  children: state && (
                    <PuzzleView
                      {...{
                        socket,
                        hash,
                        text,
                        patch,
                        state,
                        params,
                        isPaused,
                        setIsPaused,
                      }}
                      applyPatch={applyPatch}
                    />
                  ),
                },
                {
                  key: '3',
                  label: 'Triangle',
                  children: state && (
                    <TriangleView {...{ hash, text, patch, state }} />
                  ),
                },
                {
                  key: 'tree',
                  label: 'Tree',
                  children: <TreeView {...{ hash, text, patch, state }} />,
                },
                {
                  key: 'json',
                  label: 'JSON',
                  children: <JsonView {...{ hash, text, patch, state }} />,
                },
              ]
            : []),
        ]}
      />
      {hash && (
        <div
          style={{
            fontFamily: 'monospace',
            position: 'fixed',
            right: 50,
            bottom: 0,
          }}
        >
          {taskId} epoch {I}
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

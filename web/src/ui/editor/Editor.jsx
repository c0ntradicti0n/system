import React, { useState, useEffect, useCallback, useRef } from 'react'
import io from 'socket.io-client'
import ShareModal from '../ShareModal'
import { parseHash } from '../../lib/read_link_params'
import TextArea from 'antd/es/input/TextArea'
import jsonPatch from 'fast-json-patch'
import { Button, Tabs } from 'antd'
import { JsonView } from './JsonView'
import { TreeView } from './TreeView'
import { TriangleView } from './TriangleView'
import { ExperimentsView } from './ExperimentsView'
import { PuzzleView } from './PuzzleView'
var debounce = require('lodash.debounce')

let taskId = null
const setTaskId = (tI) => (taskId = tI)

localStorage.debug = '*'
const socket = io('', { transports: ['websocket'], retries: 1 })
let inited = false
let active_patching = false
let init_phase = false

export const Editor = () => {
  const [state, setState] = useState(null)
  const [patch, _setPatch] = useState(null)
  const [mods, setMods] = useState(null)
  const [activeTab, setActiveTab] = useState('json')

  const [_text, _setText] = useState('')
  const [meta, setMeta] = useState('')
  const [consumedTaskIds, setConsumedTaskIds] = useState([]) // List of taskIds that have been consumed by the server

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
      if (hash) socket.timeout(3000).emit('set_init_state', hash)
    }
    _setPatch(patch)
    setState(newState)
  }

  const timeoutId = useRef(null)

  const checkAwaitState = useCallback(() => {
    console.log('CHECK ', taskId)
    if (taskId) {
      console.log('CHECK AWAIT STATE', taskId)
      socket.timeout(3000).emit('patch_poll', taskId)

      // Clear previous timeout, just in case
      if (timeoutId.current) {
        clearTimeout(timeoutId.current)
      }
      timeoutId.current = setTimeout(checkAwaitState, 1000) // Recursive call
    }
  }, [taskId])

  useEffect(() => {
    const params = parseHash(window.location.hash)
    console.log(params)
    if (params.hash !== undefined && params.hash !== hash) {
      setHash(params.hash)
      if (!init_phase) {
        console.log('INIT FROM HASH')
        init_phase = true
        socket.timeout(3000).emit('set_init_hash', params.hash)
      }
    }
    if (params.activeTab !== undefined && params.activeTab !== activeTab) {
      setActiveTab(params.activeTab)
    }
  }, [hash])

  useEffect(() => {
    if (hash && state && !active_patching) {
      console.log('START PATCHING', hash, state)
      socket.timeout(3000).emit('update_state', hash)
      active_patching = true
    }
  }, [hash, state])

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
    socket.on('set_meta', (meta) => {
      console.log('set_meta', meta)
      setMeta(meta)
    })
    socket.on('set_mods', (mods) => {
      console.log('set_mods')
      setMods(mods)
    })
    socket.on('set_task_id', (task_id) => {
      if (task_id !== taskId) {
        console.log('set_task_id', task_id, task_id)

        setTaskId(task_id)
        checkAwaitState()
      }
    })
    socket.on('initial_state', (data) => {
      console.log('initial_state', hash)
      socket.timeout(3000).emit('set_init_state', hash)
    })
    socket.on('set_state', (state) => {
      console.log('set_state')
      setState(state)
    })

    socket.on('patch_receive', (result) => {
      console.log('patch_receive')
      if (result.status === 'SUCCESS') {
        console.log('patch_receive SUCCESS')
        setPatch(result.result)
        active_patching = false
        setTaskId(null) // Reset taskId after getting result

        console.log('RESTART PATCHING', hash, state)
        socket.timeout(3000).emit('update_state', hash)
        active_patching = true
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
    }
  }, [hash, state])

  useEffect(() => {
    console.log('useEffect', { hash, text })
    if (text && hash) {
      socket.timeout(3000).emit('set_init_state', hash)
      socket.timeout(3000).emit('get_meta', hash)
    }
  }, [hash, text])
  useEffect(() => {
    socket.timeout(3000).emit('set_initial_mods')
  }, [])

  return (
    <div className="App" style={{ overflowY: 'scroll' }}>
      <ShareModal url={'/editor'} linkInfo={{ hash, activeTab }} />
      <TextArea
        showCount
        value={_text}
        maxLength={1000000}
        style={{
          height: 120,
          marginBottom: 24,
          backgroundColor: '#111 !important',
        }}
        onChange={(e) => {
          _setText(e.target.value)
        }}
        placeholder="Paste your paragraphed text here"
      />
      <Button
        onClick={() => {
          setText(_text)
          setHash(null)
          console.log('set_text', _text)
          socket.timeout(3000).emit('set_init_text', _text)
        }}
      >
        Send Text
      </Button>
      <TextArea
        showCount
        value={meta}
        maxLength={1000}
        style={{
          height: 120,
          marginBottom: 24,
          backgroundColor: '#111 !important',
        }}
        onChange={(e) => {
          setMeta(e.target.value)
        }}
        placeholder="Paste some Metadata here, can have bib-text-formatted references"
      />
      <Button
        onClick={() => {
          console.log('set_meta', meta)
          socket.timeout(3000).emit('set_init_meta', hash, meta)
        }}
      >
        Set Metadata
      </Button>{' '}
      {taskId}
      <Tabs
        activeKey={activeTab}
        onChange={(key) => {
          setActiveTab(key)
        }}
        tabPosition={'left'}
        width={'100%'}
        items={[
          {
            key: 'ex',
            label: 'Experiments',
            children: <ExperimentsView {...{ mods }} />,
          },
          {
            key: 'pz',
            label: 'Puzzle',
            children: state && (
              <PuzzleView
                {...{ hash, text, patch, state }}
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
        ]}
      />
    </div>
  )
}

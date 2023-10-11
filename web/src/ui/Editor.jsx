import React, { useState, useEffect } from 'react'
import io from 'socket.io-client'
import ShareModal from './ShareModal'
import { parseHash } from '../lib/read_link_params'
import TextArea from 'antd/es/input/TextArea'

localStorage.debug = '*'
const socket = io('https://localhost', { transports: ['websocket'] })
let inited = false
export const Editor = () => {
  const [state, setState] = useState(null)
  const [patch, setPatch] = useState(null)

  const [_text, _setText] = useState('')

  const [text, setText] = useState('')
  const [hash, setHash] = useState(null)

  useEffect(() => {
    const params = parseHash(window.location.hash)

    if (params.hash !== undefined && params.hash !== hash) {
      setHash(params.hash)
      socket.emit('set_init_hash', params.hash)
    }
  }, [hash]) // Depend on initialPageLoad so that this useEffect runs only once

  useEffect(() => {
    // Listen for the hash from the server after sending the text
    socket.on('set_hash', (hash) => {
      setHash(hash)
    })
    // Listen for the hash from the server after sending the text
    socket.on('set_text', (text) => {
      console.log('set_text', text)
      setText(text)
      _setText(text)
    })

    socket.on('initial_state', (data) => {
      socket.emit('set_init_state', hash)
    })
    socket.on('set_state', (state) => {
      console.log('set_state', state)
      setState(state)
      if (hash) socket.emit('update_state', hash)
    })

    socket.on('state_patch', (patch) => {
      console.log('patch', patch)
      setPatch(patch)
      socket.emit('update_state', hash)
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
  }, [hash])

  useEffect(() => {
    console.log('useEffect', { hash, text })
    if (text && hash) socket.emit('set_init_state', hash)
  }, [hash, text])

  return (
    <div className="App" style={{overflowY: "scroll"}}>
      <ShareModal url={'/editor'} linkInfo={{ hash }} />
      <TextArea
        showCount
        value={_text}
        maxLength={1000000}
        style={{
          height: 120,
          marginBottom: 24,
        }}
        onChange={(e) => {
          _setText(e.target.value)
        }}
        placeholder="Paste your paragraphed text here"
      />
      <button
        onClick={() => {
          setText(_text)
          setHash(null)
          console.log('set_text', _text)
          socket.emit('set_init_text', _text)
        }}
      >
        Send Text
      </button>
      <pre style={{ color: '#fff', textAlign: 'left', whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(hash, null, 2)}
      </pre>
      <pre style={{ color: '#fff', textAlign: 'left', whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(text, null, 2).slice(0, 100)}
      </pre>
      <pre style={{ color: '#fff', textAlign: 'left', whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(patch, null, 2)}
      </pre>
      <pre style={{ color: '#fff', textAlign: 'left', whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(state, null, 2)}
      </pre>

    </div>
  )
}

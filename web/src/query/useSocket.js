import { useState, useEffect, useCallback, useRef } from 'react'
import io from 'socket.io-client'
const socket = io.connect('', {
  debug: true,
  transports: ['websocket'],
  perMessageDeflate: false,
  retries: 1,
  pingInterval: 5000,
  pingTimeout: 2000,
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 50,
  timeout: 5000,
})

export const useSocket = (hash_id) => {
  const [mods, setMods] = useState(null)
  const [params, setParams] = useState(null)
  const [text, setText] = useState(null)
  const [meta, setMeta] = useState(null)
  const [progress, setProgress] = useState(null)
  const [status, setStatus] = useState(null)
  const [prevHash, setPrevHash] = useState(null)

  const [state, setState] = useState(null)
  const [isPaused, setIsPaused] = useState(false)
  const [i, setI] = useState(0)

  const [once] = useState(false)
  const requestInitialState = useCallback(() => {
    socket.emit('join', hash_id)
  }, [hash_id])
  useEffect(() => {
    requestInitialState()
  }, [once, requestInitialState])

  const deleteMod = useCallback((text, meta) => {
    socket.emit('delete_mod', hash_id)
  })

  useEffect(() => {
    return () => {
      console.log('useSocket unmount')
      if (prevHash && hash_id !== prevHash) {
        socket.emit('leave', prevHash)
      } else {
        setPrevHash(hash_id)
      }
    }
  }, [hash_id, prevHash])

  useEffect(() => {
    socket.on('connect_error', (error) => {
      console.error('Connection Error:', error)
    })
    socket.on('set_state', (result) => {
      if (!isPaused) {
        setState(result)
      }
    })
    socket.on('set_mods', (result) => {
      console.log('set_mods', result)
      setMods(result)
    })
    socket.on('set_params', (result) => {
      setParams(result)
    })
    socket.on('set_text', (result) => {
      setText(result)
    })
    socket.on('set_meta', (result) => {
      setMeta(result)
    })
    socket.on('set_progress', (result) => {
      setProgress(result)
    })
    socket.on('set_status', (result) => {
      setStatus(result)
    })
    socket.on('set_i', (result) => {
      setI(result)
    })

    return () => {
      socket.removeAllListeners()
      console.log('useSocket unmount')
    }
  }, [hash_id, isPaused, i])

  return {
    mods,
    hash: hash_id,
    params,
    text,
    meta,
    progress,
    status,

    socket,
    state,
    setState,
    i,

    requestInitialState,
    deleteMod,
    setIsPaused,
  }
}

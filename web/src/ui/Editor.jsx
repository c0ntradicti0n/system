import React, { useState, useEffect } from 'react'
import io from 'socket.io-client'

localStorage.debug = '*'
const socket = io('http://localhost:9876', { transports: ['websocket'] })
export const Editor = () => {
  const [state, setState] = useState(null)

  useEffect(() => {
    socket.on('initial_state', (data) => {
      setState(data)
      socket.emit('update_state')
    })

    // Listen for state patches from the server
    socket.on('state_patch', (patch) => {
      // Apply the patch to the current state
      // For simplicity, we'll assume the patch is a direct replacement for now
      // In a real-world scenario, you'd use a library like `fast-json-patch` to apply the patch
      console.log('patch', patch)
      setState(patch)
      socket.emit('update_state')
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
    })

    // Cleanup the listener when the component is unmounted
    return () => {
      socket.off('initial_state')
      socket.off('state_patch')
    }
  })

  return (
    <div className="App">
      <h1>React Socket.io Client</h1>
      <pre>{JSON.stringify(state, null, 2)}</pre>
    </div>
  )
}

import { useState, useEffect, useCallback, useRef } from 'react'
import io from 'socket.io-client'
import jsonPatch from 'fast-json-patch' // Assuming you're using fast-json-patch
const socket = io.connect('https://localhost', {
  debug: true,
  transports: ['websocket'],
  perMessageDeflate: false ,
  retries: 1,
  pingInterval: 5000,
  pingTimeout: 2000,
})
export const useSocket = (hash_id) => {
  const [state, setState] = useState(null)
  const [hash] = useState(hash_id)
  const [isPaused, setIsPaused] = useState(false)
  const [i, setI] = useState(0)
  const requestInitialState = useCallback(() => {
    socket.emit('join', hash );
    if (!socket) return
    //console.log('sending get_state', hash)
    socket.emit('set_state', hash, (response) => {
      //console.log('response', response)
      setState(response)
    })


  }, [hash])
  useEffect(() => {
    socket.on('connect_error', (error) => {
      console.error('Connection Error:', error)
    })



socket.on('patch', (result) => {
      //console.log('patch', { result, state })
      if (!state) requestInitialState()
      if (state && result.patch && !isPaused) {
        const newState = jsonPatch.applyPatch(state, result.patch).newDocument
        setState({...newState})
        setI(i + 1)
        //console.log('new state', newState)
      }
    })
    return () => {
        socket.off('patch')

      socket.off('connect_error')
    }

  }, [state])




  //console.log('state', state)

  return {
    socket,
    state,
    setState,
    hash,
    setIsPaused,
    i,
    requestInitialState,
  }
}

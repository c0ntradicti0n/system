import { useSocket } from '../query/useSocket'
import {useEffect} from "react";

if (!sessionStorage.tabId) {
    sessionStorage.tabId = Date.now() + Math.random().toString(36).substr(2, 9);
}
const tabId = sessionStorage.tabId;

export const SocketTest = () => {
  const { state, hash, requestInitialState, i } = useSocket(tabId)
    useEffect(() => {
        console.log('SocketTest', state, hash, i)
        requestInitialState(hash)
    }, []);

  return (
    <div>
      {i} {hash} {JSON.stringify(state)}
      <button onClick={() => requestInitialState()}>hello</button>
    </div>
  )
}

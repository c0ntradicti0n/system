import { useSocket } from '../query/useSocket'

export const SocketTest = () => {
  const { state, hash, i, meta, progress, status, text, params } = useSocket(
    '1cb10b667a508d3585b760a6ef9e8567b38af5421469a767678007a8a525d911',
  )

  return (
    <div style={{ overflowY: 'scroll !important' }}>
      <h1>SocketTest</h1>
      <pre style={{ overflowY: 'scroll', height: '60vh' }}>
        {i}
        <br />
        {hash}
        <br />
        {JSON.stringify(meta, null, 2)}
        <br />
        {JSON.stringify(params, null, 2)}
        <br />
        {JSON.stringify(progress, null, 2)}
        <br />
        {JSON.stringify(status, null, 2)}
        <br />
        {JSON.stringify(state, null, 2)}
        <br />

        {JSON.stringify(text)}
      </pre>
    </div>
  )
}

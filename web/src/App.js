import React, { useState } from 'react'
import { QueryClient, QueryClientProvider } from 'react-query'
import './App.css'
import Fractal from './Fractal.tsx'
const queryClient = new QueryClient()
function App() {
  const [content, setContent] = useState('')

  return (
    <QueryClientProvider client={queryClient}>
      <Fractal setContent={setContent} />
      <div
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          zIndex: 9999,
          fontSize: '2.5rem',
          wordWrap: 'break-word',
          width: '40%',
        }}
      >
        {content}
      </div>
    </QueryClientProvider>
  )
}

export default App

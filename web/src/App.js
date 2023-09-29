import React, { useState } from 'react'
import { QueryClient, QueryClientProvider } from 'react-query'
import './App.css'

import Fractal from './ui/Fractal.tsx'
import { Puzzle } from './ui/Puzzle'

import { useRoutes } from 'raviger'

const routes = {
  '/': () => <Fractal />,
  '/puzzle': () => <Puzzle />,
}

const queryClient = new QueryClient()
function App() {
  const routeResult = useRoutes(routes)

  return (
    <div className="App">
      <QueryClientProvider client={queryClient}>
        {routeResult || '404'}
      </QueryClientProvider>
    </div>
  )
}

export default App

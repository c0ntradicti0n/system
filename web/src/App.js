import React from 'react'
import { useRoutes } from 'raviger'
import { QueryClient, QueryClientProvider } from 'react-query'
import './app.css'

import { Puzzle } from './ui/editor/Puzzle'
import { Editor } from './ui/editor/Editor'

import { Viewer } from './ui/viewer/Viewer'

const routes = {
  '/': () => <Viewer />,
  '/puzzle': () => (
    <Puzzle
      data={{
        1: {
          1: {
            1: {
              '.': '111',
            },

            2: {
              '.': '112',
            },
            3: {
              '.': '113',
            },
            '.': '11',
          },
          2: {
            '.': '12',
          },
          3: {
            1: {
              '.': '131',
            },
            2: {
              '.': '132',
            },
            3: {
              '.': '133',
            },
            '.': '13',
          },
          '.': '1',
        },
        2: {
          1: {
            '.': '21',
          },
          2: {
            '.': '22',
          },
          3: {
            '.': '23',
          },
          '.': '2',
        },
        3: {
          1: {
            '.': '31',
          },
          2: {
            '.': '32',
          },
          3: {
            '.': '33',
          },
          '.': '3',
        },
      }}
    />
  ),
  '/editor': () => <Editor />,
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

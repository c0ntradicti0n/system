import React, { createContext, useContext, useState, useEffect, useMemo } from 'react'
import { useRoutes } from 'raviger'
import { QueryClient, QueryClientProvider } from 'react-query'
import './app.css'

import { Puzzle } from './ui/editor/Puzzle'
import { Editor } from './ui/editor/Editor'

import { Viewer } from './ui/viewer/Viewer'
import Explanation from './ui/explanation/Explanation'

const ThemeContext = createContext()

export function useTheme() {
  return useContext(ThemeContext)
}

const routes = {
  '/': () => <Viewer />,
  '/explanation': () => <Explanation />,
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
  const [lightMode, setLightMode] = useState(() => {
    return localStorage.getItem('lightMode') === 'true'
  })

  useEffect(() => {
    localStorage.setItem('lightMode', lightMode)
    document.documentElement.classList.toggle('light-mode', lightMode)
  }, [lightMode])

  const toggleLightMode = () => setLightMode(v => !v)

  const [lang, setLang] = useState(() => {
    return localStorage.getItem('lang') || 'en'
  })

  useEffect(() => {
    localStorage.setItem('lang', lang)
  }, [lang])

  const toggleLang = () => setLang(v => v === 'en' ? 'de' : 'en')
  const themeValue = useMemo(() => ({ lightMode, toggleLightMode, lang, toggleLang }), [lightMode, lang])

  const routeResult = useRoutes(routes)

  return (
    <div className="App">
      <QueryClientProvider client={queryClient}>
        <ThemeContext.Provider value={themeValue}>
          {routeResult || '404'}
        </ThemeContext.Provider>
      </QueryClientProvider>
    </div>
  )
}

export default App

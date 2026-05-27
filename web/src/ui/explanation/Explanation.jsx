import React from 'react'
import Fractal from '../viewer/Fractal'
import { useTheme } from '../../App'
import './explanation.css'

const EXPLANATION_DATA = {
  '1': {
    '1': { '.': 'These' },
    '2': { '.': 'Antithese' },
    '3': { '.': 'Synthese' },
    '.': 'These',
  },
  '2': {
    '1': { '.': 'These' },
    '2': { '.': 'Antithese' },
    '3': { '.': 'Synthese' },
    '.': 'Antithese',
  },
  '3': {
    '1': { '.': 'These' },
    '2': { '.': 'Antithese' },
    '3': { '.': 'Synthese' },
    '.': 'Synthese',
  },
  '.': '',
}

const Explanation = () => {
  const { lightMode } = useTheme()

  return (
    <div className="explanation-page">
      <div className="explanation-fractal">
        <Fractal
          PRESET_DATA={EXPLANATION_DATA}
          hideIndices={true}
          pastel={lightMode}
          fontMultiplier={lightMode ? 2 : 1}
        />
      </div>
      <div className="explanation-text">
        <h1>These / Antithese / Synthese</h1>
        <p>
          This fractal follows the Hegelian dialectic. Every node is a position
          (These) that generates its own contradiction (Antithese). From the
          tension between both, a new, more refined position emerges (Synthese).
        </p>
        <p>
          Navigate by zooming and clicking. The triangular shape shows how
          opposites produce something new — every step inward is a synthesis.
        </p>
      </div>
    </div>
  )
}

export default Explanation
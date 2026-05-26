import React from 'react'
import Fractal from '../viewer/Fractal'
import './explanation.css'

const EXPLANATION_DATA = {
  '1': { '.': 'These' },
  '2': { '.': 'Antithese' },
  '3': { '.': 'Synthese' },
  '.': '',
}

const Explanation = () => {
  return (
    <div className="explanation-page">
      <div className="explanation-fractal">
        <Fractal PRESET_DATA={EXPLANATION_DATA} hideIndices={true} />
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
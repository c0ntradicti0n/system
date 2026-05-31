import React, { useState, useEffect } from 'react'
import Triangle from './Triangle'
import { mergeDeep, lookupDeep } from '../../lib/nesting'
import './fractal.css'
import TriangleMini from './TriangleMini'

const FractalMini = ({ PRESET_DATA, size }) => {
  const [collectedData, setCollectedData] = useState({})
  const [visualData, setVisualData] = useState(null)

  useEffect(() => {
    if (PRESET_DATA) {
      const mergedData = mergeDeep({}, PRESET_DATA)
      setCollectedData(mergedData)
      setVisualData(lookupDeep('', mergedData))
    }
  }, [PRESET_DATA])

  return (
    <div className="fractal-thumbnail" style={{ width: size, height: size }}>
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
        }}
      >
        <TriangleMini
          id={''}
          fullId={''}
          data={visualData ?? collectedData}
          size={size}
          left={0}
          top={0}
          level={3}
          animate={false}
        />
      </div>
    </div>
  )
}

export default FractalMini

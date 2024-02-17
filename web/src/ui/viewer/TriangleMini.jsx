import React from 'react'
import { stringToColour } from '../../lib/color'
import calculateFontSize from '../../lib/FontSize'
import { MAX_LEVEL } from '../../config/const'
import { getLeftPosition, getTopPosition } from '../../lib/position'

const TriangleMini = ({
  id,
  fullId,
  data,
  size,
  left,
  top,
  level = MAX_LEVEL,
}) => {
  if (!data) return null

  const title = data?.['.']
  const { shortTitle, fontSize } = calculateFontSize(size, title, 0.7)

  return (
    <div
        key={fullId}
      style={{
        position: 'absolute',
        width: size,
        height: size,
        left: left,
        top: top,
      }}
    >
      <div
        className="triangle"
        style={{
          backgroundColor: fullId
            ? stringToColour(fullId.replace('/', ''), 1)
            : 'black',
          zIndex: 1000 - level,
        }}
      >
        {[1, 2, 3].map((subTriangleDir, index) =>
          data[subTriangleDir] ? (
            <TriangleMini
              fullId={`${fullId}/${subTriangleDir}`}
              id={subTriangleDir}
              data={data[subTriangleDir]}
              size={size / 2}
              left={getLeftPosition(index, size)}
              top={getTopPosition(index, size)}
              level={level + 1}
            />
          ) : null,
        )}

        <div
          className="triangle-content"
          style={{
            fontSize: fontSize,
            color: id ? 'black' : 'white',
            width: `${size / 3}px`,
          }}
        >
          {title && <div className="triangle-title">{shortTitle}</div>}
        </div>
      </div>
    </div>
  )
}

export default TriangleMini

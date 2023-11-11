import React from 'react'
import Fractal from '../viewer/Fractal'

export function TriangleView(props) {
  return <Fractal PRESET_DATA={props.state} />
}

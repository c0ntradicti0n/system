import React from 'react'
import { Puzzle } from '../Puzzle'

export function PuzzleView(props) {
  return <Puzzle data={props.state} applyPatch={props.applyPatch} />
}

import React, { useState } from 'react'
import { Puzzle } from './Puzzle'
import './puzzle-controls.css'

export function PuzzleView(props) {
  return <Puzzle data={props.state} action={props.action} props={props} />
}

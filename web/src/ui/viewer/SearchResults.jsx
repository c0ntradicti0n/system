import React, { useEffect, useRef } from 'react'
import Muuri from 'muuri'
import { Pin } from './Pin'
import './search.css'
const SearchResults = ({ labels, setHiddenId }) => {
  const gridRef = useRef(null)

  useEffect(() => {
    if (!gridRef.current) return
    const grid = new Muuri(gridRef.current, {
      items: '.search-item',
      dragEnabled: true,
      layout: { horizontal: true },
      rounding: true,
      width: 1000,
    })

    return () => grid.destroy()
  }, [gridRef, labels])

  if (!labels) return null

  return (
    <div ref={gridRef} className="grid">
      {labels.map((value) => (
        <Pin value={value} setHiddenId={setHiddenId} key={value.path} />
      ))}
    </div>
  )
}

export { SearchResults }

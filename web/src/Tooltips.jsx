import { postProcessTitle } from './position'
import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'

export const Tooltips = ({ data, id, isWindowWide }) => {
  const fetchTexts = async () => {
    const res = await fetch(
      `${process.env['REACT_APP_HOST']}/api/text/${id ?? ''}`,
    )
    if (!res.ok) {
      throw new Error('Network response was not ok')
    }
    const newData = await res.json()
    console.log('got', { newData })
    return newData
  }

  const { data: texts, error } = useQuery(['triangle', id], fetchTexts, {
    keepPreviousData: true,
  })

  console.log('texts', texts, error)

  return (
    <div
      style={{
        position: isWindowWide ? 'absolute' : 'static',
        pointerEvents: 'none',
        top: isWindowWide ? 'auto' : 0,
        left: isWindowWide ? 0 : 'auto',
        height: isWindowWide ? '100vh' : '30vh',
        width: isWindowWide ? '40vw' : 'vw',
        zIndex: 99999999,
        backgroundColor: '#000',
        display: 'flex',
        flexDirection: 'row',
        flexWrap: 'wrap', // New addition for wrapping elements
      }}
    >
      {['1', '2', '3', '_'].map((key) => {
        const header_value = key === '_' ? data[key] : data[key]
        const value = (texts ?? data)[key]
        const header =
          typeof header_value === 'object' ? header_value?.['.'] : header_value
        const content = typeof value === 'object' ? value?.['.'] : value
        return (
          <div
            key={key}
            style={{
              wordWrap: 'break-word',
              textAlign: 'left',
              width: '49%',
              fontSize: '14px',
            }}
          >
            <h4>
              {['1', '2', '3'].includes(key) ? key + '.' : '∇'}{' '}
              {postProcessTitle(header)}
            </h4>
            {content}
          </div>
        )
      })}
    </div>
  )
}

import React from 'react'
import { toJSON } from 'bibtex-parse-js'

const BibTeXViewer = ({ entry, setIsGood, isGood, short = false }) => {
  if (!entry) return null
  let parsedData
  try {
    parsedData = toJSON(entry)
  } catch (e) {
    console.log(e)
    console.log(entry)
    return <div>{entry}</div>
  }

  if (!parsedData || !parsedData.length) {
    return <div>{entry}</div>
  }
  // Assuming only one entry for simplicity
  const data = parsedData[0].entryTags

  if (data.title && !isGood) {
    setIsGood(true)
  }
  return (
    <div className="bibtex-entry" style={{ display: 'inline' }}>
      <strong>{data.title}</strong>{' '}
      {!short && (
        <>
          {' '}
          {data.author}
          <em>{data.journal}</em> {data.year}
          {data.url && <a href={data.url}> Link</a>}
        </>
      )}
    </div>
  )
}

export default BibTeXViewer

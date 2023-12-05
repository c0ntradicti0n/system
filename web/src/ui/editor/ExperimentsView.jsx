import React from 'react'
import { Button } from 'antd'
import { navigate } from 'raviger'
import BibTeXViewer from './BibtexViewer'
import { updateUrlHash } from '../../lib/read_link_params'

export function ExperimentsView(props) {
  const [isGood, setIsGood] = React.useState({})
  return (
    <>
      {props.mods?.map((mod) => (
        <div
          key={mod.hash}
          style={{
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            margin: '0.5em',
          }}
        >
          <Button
            style={{
              width: '70vw',
              backgroundColor: '#123',
              color: 'yellow',
              textAlign: 'left',
            }}
            onClick={() => {
              props.setHashId(mod.hash)
              props.setActiveTab('puzzle')
              updateUrlHash('activeTab', 'puzzle')
              updateUrlHash('hash', mod.hash)
            }}
          >
            {mod.meta ? (
              <BibTeXViewer
                entry={mod.meta}
                isGood={isGood[mod.hash]}
                setIsGood={() => {
                  setIsGood({ ...isGood, [mod.hash]: !isGood[mod.hash] })
                }}
              />
            ) : (
              mod.hash
            )}
          </Button>
          {!isGood[mod.hash] && (
            <Button
              style={{ width: '5vw', backgroundColor: '#123', color: 'yellow' }}
              onClick={() => props.deleteMod(mod.hash)}
            >
              🗑
            </Button>
          )}
          Reset
          {
            <Button
              style={{ width: '5vw', backgroundColor: '#123', color: 'yellow' }}
              onClick={() => props.resetMod(mod.hash)}
            >
              ↺
            </Button>
          }
        </div>
      ))}
    </>
  )
}

import React from 'react'
import { Button } from 'antd'
import { navigate } from 'raviger'
import BibTeXViewer from './BibtexViewer'

export function ExperimentsView(props) {
  const [isGood, setIsGood] = React.useState({})
  return (
    <>
      {props.mods?.map((mod) => (
        <div>
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
          <Button
            style={{ width: '70vw', backgroundColor: '#123', color: 'yellow' }}
            onClick={() => {
              navigate(`/editor#hash=${mod.hash}&activeTab=px`)
              window.location.reload()
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
        </div>
      ))}
    </>
  )
}

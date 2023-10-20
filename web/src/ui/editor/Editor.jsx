import React, { useState, useEffect } from 'react'
import io from 'socket.io-client'
import ShareModal from '../ShareModal'
import { parseHash } from '../../lib/read_link_params'
import TextArea from 'antd/es/input/TextArea'
import jsonPatch from 'fast-json-patch';
import {Button, Tabs} from "antd";
import {JsonView} from "./JsonView";
import {TreeView} from "./TreeView";
import {TriangleView} from "./TriangleView";
import {ExperimentsView} from "./ExperimentsView";

localStorage.debug = '*'
const socket = io('', { transports: ['websocket'] })
let inited = false

export const Editor = () => {
  const [state, setState] = useState(null)
  const [patch, setPatch] = useState(null)
  const [mods, setMods] = useState(null)
  const [activeTab, setActiveTab] = useState("json")

  const [_text, _setText] = useState('')

  const [text, setText] = useState('')
  const [hash, setHash] = useState(null)

  useEffect(() => {
    const params = parseHash(window.location.hash)
console.log(params)
    if (params.hash !== undefined && params.hash !== hash) {
      setHash(params.hash)
      socket.emit('set_init_hash', params.hash)
    }
        if (params.activeTab !== undefined && params.activeTab !== activeTab) {
      setActiveTab(params.activeTab)
    }
  }, [hash]) // Depend on initialPageLoad so that this useEffect runs only once

  useEffect(() => {
    // Listen for the hash from the server after sending the text
    socket.on('set_hash', (hash) => {
      setHash(hash)
        setPatch(null)
        setState(null)
    })
    // Listen for the hash from the server after sending the text
    socket.on('set_text', (text) => {
      console.log('set_text', text)
      setText(text)
      _setText(text)
                setPatch(null)
        setState(null)
    })
          socket.on('set_mods', (mods) => {
      console.log('set_mods', mods)
      setMods(mods)

    })


    socket.on('initial_state', (data) => {
        console.log('initial_state', hash)
      socket.emit('set_init_state', hash)
    })
    socket.on('set_state', (state) => {
      console.log('set_state', state)
      setState(state)
      if (hash) socket.emit('update_state', hash)
    })

    socket.on('state_patch', (patch) => {
      console.log('patch', patch, state)
      if (patch &&  state) {
          try {
              setPatch(patch)
              const newState = jsonPatch.applyPatch(state, patch).newDocument;
              setState(newState);
              console.log('patched')
            } catch (e) {
                console.log('patch error', e)
              socket.emit('set_init_state', hash)
          }
      }
      socket.emit('update_state', hash)
    })

    socket.on('connect_error', (error) => {
      console.error('Connect Error:', error)
    })

    socket.on('error', (error) => {
      console.error('Error:', error)
    })

    socket.on('connect', () => {
      console.log('Connected to server')
    })

    socket.on('disconnect', (reason) => {
      console.log('Disconnected:', reason)

      inited = false
    })
    inited = true

    return () => {
      socket.off('initial_state')
      socket.off('state_patch')
      socket.off('set_hash')
      socket.off('set_text')
    }
  }, [hash, state])

  useEffect(() => {
    console.log('useEffect', { hash, text })
    if (text && hash) socket.emit('set_init_state', hash)
  }, [hash, text])
    useEffect(() => {
     socket.emit('set_initial_mods')
    }, []);

  return (
    <div className="App" style={{overflowY: "scroll"}}>
        <ShareModal url={'/editor'} linkInfo={{hash, activeTab}}/>
        <TextArea
            showCount
            value={_text}
            maxLength={1000000}
            style={{
                height: 120,
                marginBottom: 24,
                backgroundColor: '#111 !important',
            }}
            onChange={(e) => {
                _setText(e.target.value)
            }}
            placeholder="Paste your paragraphed text here"
        />
        <Button
            onClick={() => {
                setText(_text)
                setHash(null)
                console.log('set_text', _text)
                socket.emit('set_init_text', _text)
            }}
        >
                        Send Text
        </Button>
            <Tabs
                activeKey={activeTab}
                onChange={(key) => {setActiveTab(key)}}
                tabPosition={"left"}
                width={"100%"}
                items={[
                              {
                        key: 'ex',
                        label: 'Experiments',
                        children:  <ExperimentsView  {...{mods}}/>,
                    },
                                        {
                        key: 'pz',
                        label: 'Puzzle',
                        children:  state && <TriangleView  {...{hash, text,patch,state}}/>,
                    },
                    {
                        key: '3',
                        label: 'Triangle',
                        children:  state && <TriangleView  {...{hash, text,patch,state}}/>,
                    },
                    {
                        key: 'tree',
                        label: 'Tree',
                        children: <TreeView  {...{hash, text,patch,state}}/>,
                    },
                    {
                        key: 'json',
                        label: 'JSON',
                        children:<JsonView {...{hash, text,patch,state}}/>
,
                    },
                ]}
            />



    </div>
  )
}

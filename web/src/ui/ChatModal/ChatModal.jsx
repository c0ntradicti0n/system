import React, { useState, useEffect, useRef } from 'react'
import { Modal, Button, Spin, Select, Input, Card, Collapse, Radio } from 'antd'
import { CloseOutlined } from '@ant-design/icons'
import '../viewer/chat-modal.css'
import TextArea from 'antd/es/input/TextArea'
import Meta from 'antd/es/card/Meta'
import Fractal from '../viewer/Fractal'
import FractalMini from '../viewer/FractalMini'
import { setValueInNestedObject } from '../../lib/nesting'
import useChatModal from "./state";

const { Option } = Select
const TOKEN_STORAGE_KEY = 'chatModalToken'
let loading = false

const ChatModal = () => {
  const state = useChatModal()
  const [messages, setMessages] = useState([])
  const [message, setMessage] = useState('')
  const [token, setToken] = useState(localStorage.getItem(TOKEN_STORAGE_KEY))
  const [task, setTask] = useState(localStorage.getItem('task'))
  const messageContainerRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [visualizationData, setVisualizationData] = useState(null)
  const [bestSolution, setBestSolution] = useState(0)

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  const showModal = () => {
    setMessages([])
    state.setVisible(true)
    console.log("NEW")
  }

  const handleCancel = (e) => {
    console.log("CANCEL!!!")
    if (isFullscreen) {
      return
    }
    e.preventDefault()
    e.stopPropagation()
    state.setVisible(false)
    console.log(state)
  }



  useEffect(() => {
    console.log('useEffect', state, task, token, loading)
    if (!state.visible || loading) return
    loading = true
    console.log({task, token})
    if (!(task&& token && state.path))
      return

    console.log(state.path)
    setMessages([{ loading: true }])

    fetch('/api/philo/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, token, path: state.path }),
    })
      .then((response) => response.json())
      .then((data) => {

        addMessage(data)
        loading = false
        scrollMessageContainerToBottom()
      })
      .catch((e) => console.error(e))
  }, [ state.visible, state.path, task, token])


  const handleSendMessage = () => {
    if (! state.visible) return
    loading = true

    addMessage({ user: message })
    addMessage({ loading: true })

    const instruction = messages.find((msg) => msg.instruction)?.instruction
    const prompt = messages.find((msg) => msg.prompt)?.prompt

    fetch('/api/philo/reply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, task, token, path: state.path, instruction, prompt }),
    })
      .then((response) => response.json())
      .then((data) => {
        addMessage(data)
        loading = false
        scrollMessageContainerToBottom()
      })
      .catch((e) => console.error(e))
  }

  const handleSubmitMessage = () => {
    if (! state.visible) return
    const message = messages[bestSolution]
    if (!message) return
    const data = message?.data

    fetch('/api/philo/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, task, token, path: state.path, prompt, data }),
    })
      .then((response) => response.json())
      .then((data) => {
        addMessage(data)
        loading = false
        scrollMessageContainerToBottom()
        if (!data.error) {
          state.setVisible(false)
        }
      })
      .catch((e) => console.error(e))
  }

  const addMessage = (data) => {
    if (messages.length > 0 && messages[messages.length - 1].loading) {
      messages.pop()
    }

    setMessages((prevMessages) => {
      prevMessages = prevMessages.filter((msg) => !msg.loading)
      return [...prevMessages, data]
    })
  }

  useEffect(() => {
    if (!state.visible) return
    if (token)
    localStorage.setItem(TOKEN_STORAGE_KEY, token)
    if (task)
    localStorage.setItem('task', task)
  }, [ state.visible, token, task])



  const scrollMessageContainerToBottom = () => {
    if (messageContainerRef.current) {
      messageContainerRef.current.scrollTop =
        messageContainerRef.current.scrollHeight
    }
  }
  console.log('messages', messages)

  return (
    <div style={{ width: '90%', display: 'inline' }}>
      <Button
        onClick={showModal}
        style={{
          color: 'lime',
          fontWeight: 'bold',
          fontSize: '1.5vw',
          height: '3vw',
          width: '90%',
          filter: 'invert(100%)',
        }}
      >
        Got a token? Edit via chat
      </Button>
      { state.visible && (
        <Modal
          title={
            <>
              Work on fractal at path{' '}
              <Input
                style={{ display: 'inline-flex' }}
                defaultValue={state.path}
                onPressEnter={(e) => {
                  setMessages([])
                  state.setPath(value)
                }}
                placeholder={'path in triangle'}
              />
            </>
          }
          visible={ state.visible}
          width="98%"
          height="90%"
          style={{ height: '90%', top: '0', left: '0', marginTop: '1vh' }}
          onCancel={handleCancel}
          footer={null}
          maskClosable={false}
          bodyStyle={{ padding: 0 }}
        >
          <div className="chat-container">
            <Card
              title="Chat"
              style={{
                height: '90vh',
                width: '99%',
                borderRight: '1px solid #e8e8e8',
                padding: 0,
              }}
              actions={[
                <>
                  task
                  <Select
                    value={task ?? "toc"}
                    style={{ width: 260, marginLeft: 8 }}
                    onChange={(value) => setTask(value)}
                  >
                    <Option value="toc">Work on table of contents</Option>
                    <Option value="text">Produce texts for the fractal</Option>
                  </Select>
                </>,
                <>
                  token
                  <Input
                    type="password"
                    placeholder="Enter token..."
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    style={{ marginLeft: 8, width: 50 }}
                  />
                </>,

                <Button type="primary" onClick={handleSendMessage}>
                  Teach it!
                </Button>,

                <Button type="primary" onClick={handleSubmitMessage}>
                  Submit Results of message no. {bestSolution}
                </Button>,
              ]}
            >
              <div
                className="chat-box"
                style={{ height: '60vh', overflow: 'scroll' }}
              >
                <Radio.Group
                  value={bestSolution}
                  onChange={(e) => setBestSolution(e.target.value)}
                >
                  <div className="message-container" ref={messageContainerRef}>
                    {
                      messages.map((msg, index) => (
                        <>
                          {msg?.error && (
                            <div className=" message message-error">
                              {msg?.error}
                            </div>
                          )}

                          {msg?.user && (
                            <div className=" message message-user">
                              {msg?.user}
                            </div>
                          )}
                          {msg.loading && <Spin size="small" />}

                          {msg?.data && (
                            <div className="message message-bot">
                              <Collapse
                                items={[
                                  msg?.prompt && {
                                    label: 'Prompt',
                                    children: (
                                      <pre style={{ whiteSpace: 'pre-wrap' }}>
                                        {msg.prompt}{' '}
                                      </pre>
                                    ),
                                    key: '1',
                                  },
                                  msg?.instruction && {
                                    label: 'Instruction',
                                    children: (
                                      <pre style={{ whiteSpace: 'pre-wrap' }}>
                                        {msg.instruction}
                                      </pre>
                                    ),
                                    key: '2',
                                  },
                                  msg?.reply && {
                                    label: 'Reply',
                                    children: (
                                      <pre style={{ whiteSpace: 'pre-wrap' }}>
                                        {msg.reply}
                                      </pre>
                                    ),
                                    key: '3',
                                  },
                                  msg?.data && {
                                    label: 'Data',
                                    children: ( msg?.data.length ?
                                    (
                                      <pre style={{ whiteSpace: 'pre-wrap' }}>
                                        {JSON.stringify(msg.data)}
                                      </pre>
                                    ) : null),
                                    key: '4',
                                  },
                                ].filter((x) => x)}
                                size="small"
                              />
                              {msg?.data ? (
                                <div
                                  onClick={() => {
                                    setVisualizationData([index, msg.data])
                                    toggleFullscreen()
                                  }}
                                  style={{
                                    alignSelf: 'flex-end',
                                    cursor: 'pointer',
                                  }}
                                >
                                  <div
                                    className={'message-display'}
                                    style={{
                                      display: 'flex',
                                      flexDirection: 'column-reverse',
                                      alignItems: 'center',
                                    }}
                                  >
                                    {(
                                      <Radio value={index}>Best</Radio>
                                    )}

                                    <FractalMini
                                      PRESET_DATA={msg.data}
                                      size={200}
                                    />
                                  </div>
                                </div>
                              ) : (
                                <div className="message-reply">
                                  <pre>{msg?.reply}</pre>
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      ))
                    }
                  </div>
                </Radio.Group>
              </div>
              <Meta
                description={
                  <TextArea
                    placeholder="Type a message..."
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onPressEnter={handleSendMessage}
                  />
                }
              />
            </Card>

            {isFullscreen && visualizationData && (
              <div className="fullscreen-overlay">
                <Fractal
                  PRESET_DATA={visualizationData[1]}
                  tooltips={false}
                  editData={(keys, value) => {
                    const lastKey = keys[keys.length - 1]
                    const keysWithDot = [...keys, ...(lastKey === "_" ?[] :["."])]
                    console.log("EDIT", lastKey, keys, value,keysWithDot)

                    const newObject = setValueInNestedObject(visualizationData[1] , keysWithDot, value)
                    console.log(newObject)
                    setVisualizationData([visualizationData[0], newObject])
                    messages[visualizationData[0]].data = newObject
                    setMessages([...messages])
                  }}
                />
                <button className="close-button" onClick={toggleFullscreen}>
<CloseOutlined />
                </button>
              </div>
            )}
          </div>
        </Modal>
      )}


    </div>
  )
}

export default ChatModal

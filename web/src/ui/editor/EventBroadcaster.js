import React, { useCallback } from 'react'

const EventBroadcaster = ({ children, className, predicate }) => {
  const handleEvent = useCallback(
    (e) => {
      console.log('EVENT', e)
      // Get the mouse position
      const mouseX = e.clientX
      const mouseY = e.clientY

      // Get all elements at the mouse position
      const elements = document.elementsFromPoint(mouseX, mouseY)
      console.log('ELEMENTS', elements)

      // Filter elements based on the predicate and class, then dispatch the event
      elements.forEach((element) => {
        if (element !== e.target && element.classList.contains(className)) {
          console.log('BROADCASTING EVENT', e, element)
          // Clone the event and dispatch it to the element
          const newEvent = new MouseEvent(e.type, {
            bubbles: true,
            cancelable: true,
            clientX: mouseX,
            clientY: mouseY,
            ...e, // You can spread the original event properties if needed
          })
          element.dispatchEvent(newEvent)
        }
      })
    },
    [className, predicate],
  )

  return (
    <div
      onMouseDown={handleEvent}
      onMouseUp={handleEvent}
      onMouseMove={handleEvent}
      onTouchStart={handleEvent}
      onTouchEnd={handleEvent}
      onTouchMove={handleEvent}
      onClick={handleEvent}
      style={{ width: '100vw', height: '100vh', zIndex: 9999999999999 }}
    >
      {children}
    </div>
  )
}

export default EventBroadcaster

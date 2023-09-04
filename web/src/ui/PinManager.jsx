import React, { useCallback, useEffect } from 'react'
import { PinComponent } from './PinComponent'
const PinManager = ({ pins = [], transformState, addLabel, removeLabel }) => {
  console.log('pins', pins)
  const adjustPinPosition = useCallback((pinId, sourceElementId) => {
    const pinElement = document.getElementById(pinId)
    const sourceElement = document.getElementById(sourceElementId)

    if (pinElement && sourceElement) {
      const rect = sourceElement.getBoundingClientRect()
      pinElement.style.left = rect.left + 'px'
      pinElement.style.top = rect.top + 'px'
    }
  })
  useEffect(() => {
    const registerListeners = (pinId, sourceElementId) => {
      // Adjust the pin position on different events
      const adjust = () => adjustPinPosition(pinId, sourceElementId)

      window.addEventListener('wheel', adjust)
      window.addEventListener('click', adjust)
      // For `react-zoom-pan-pinch`, you can also use the provided callback for transformations like `onTransformed`
      // Assuming you've set the transformState in your Fractal component, you can pass it down and use it here

      return () => {
        window.removeEventListener('wheel', adjust)
        window.removeEventListener('click', adjust)
      }
    }

    pins.forEach(({ pinId, sourceElementId }) => {
      if (!pinId || !sourceElementId) return
      registerListeners(pinId, sourceElementId)
    })

    // Cleanup event listeners on unmount
    return () => {
      pins.forEach(({ path }) => {
        const pinId = `pin-${path.replace(/\//g, '').replace('.', '')}`
        const sourceElementId = `triangle-${path
          .replace(/\//g, '')
          .replace('.', '')}`
        registerListeners(pinId, sourceElementId)
      })
    }
  }, [adjustPinPosition, pins])

  useEffect(() => {
    pins.forEach(({ path }) => {
      const pinId = `pin-${path.replace(/\//g, '').replace('.', '')}`
      const sourceElementId = `triangle-${path
        .replace(/\//g, '')
        .replace('.', '')}`
      adjustPinPosition(pinId, sourceElementId)
    })
  }, [adjustPinPosition, pins, transformState])
  return (
        <div>
            {pins.map(({ path, answer }) => (
                <PinComponent path={path} answer={answer}  addLabel={addLabel} removeLabel={removeLabel} />
            ))}
        </div>

  )
}

export { PinManager }

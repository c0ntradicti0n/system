import { useEffect } from 'react'
import Muuri from 'muuri'
import { idToSize, idToZIndex, positionToId } from '../ui/editor/Puzzle'
import { pointInTriangle } from './position'

const useMuuriGrid = (gridRef, options, size, props) => {
  useEffect(() => {
    try {
      if (!gridRef.current) return

      const grid = new Muuri(gridRef.current, options)
      grid.size = size
      grid.on('dragStart', (item, event) => {
        console.log('DRAG START', item, event)
        const element = item.getElement()
        const rect = element.getBoundingClientRect()

        // Triangle vertices based on the rectangle
        const ax = rect.left
        const ay = rect.bottom
        const bx = rect.right
        const by = rect.bottom
        const cx = rect.left + rect.width / 2
        const cy = rect.top

        // Get the mouse position
        const mouseX = event.clientX
        const mouseY = event.clientY

        // Check if the click is inside the triangle
        if (!pointInTriangle(mouseX, mouseY, ax, ay, bx, by, cx, cy)) {
          event.srcEvent.preventDefault()
          return
        }
      })
      grid.on('dragEnd', (item, event) => {
        console.log('DRAG RELEASE END', item, event)

        const element = item.getElement()
        const rect = element.getBoundingClientRect()
        const x = (rect.left + rect.right) / 2
        const y = (rect.top + rect.bottom) / 2

        const oldId = element.id
        const newId = positionToId(x, y)

        console.log('NEW ID', newId, x, y)

        const targetElement = document.getElementById(newId)
        console.log('TARGET ELEMENT', newId, targetElement)

        if (props.action && newId === oldId) {
          props.action(newId)
          return
        }

        // Find the triangle-text sub-divs
        const elementTriangleText = element.querySelector('.triangle-text')
        const targetTriangleText = targetElement.querySelector('.triangle-text')

        // Swap their font sizes
        if (elementTriangleText && targetTriangleText) {
          const tempFontSize = elementTriangleText.style.fontSize
          elementTriangleText.style.fontSize = targetTriangleText.style.fontSize
          targetTriangleText.style.fontSize = tempFontSize
        }

        targetElement.setAttribute('id', oldId)
        element.setAttribute('id', newId)

        targetElement.style.width = idToSize(oldId)
        targetElement.style.height = idToSize(oldId)
        element.style.width = idToSize(newId)
        element.style.height = idToSize(newId)

        const oldZIndex = targetElement.style.zIndex

        targetElement.style.zIndex = element.style.zIndex
        element.style.zIndex = oldZIndex

        props.socket.timeout(3000).emit(
          'save_params',

          {
            ...props.params,
            actions: [
              ...(props?.params?.actions ?? []),
              {
                timestamp: Date.now(),
                action: 'swap',
                label: element.textContent.slice(0, 20),
                source: oldId,
                target: newId,
              },
            ],
          },
          props.hash,
        )
        console.log('SAVE PARAMS', props.params)

        grid.refreshItems().layout()
      })

      return () => {
        // Remove event listener for dragReleaseEnd event
        grid.off('dragReleaseEnd')

        grid.off('dragStart')
        grid.off('dragEnd')
        try {
          grid.destroy()
        } catch (e) {
          console.log(e)
        }
      }
    } catch (e) {
      console.log(e)
    }
  }, [gridRef, options, size, props.params, props.socket, props.hash])
}

export default useMuuriGrid

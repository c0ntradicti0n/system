import { useEffect } from 'react'
import Muuri from 'muuri'
import { idToSize, idToZIndex, positionToId } from '../ui/editor/Puzzle'

const useMuuriGrid = (gridRef, options, size, props) => {
  useEffect(() => {
    try {
      if (!gridRef.current) return

      const grid = new Muuri(gridRef.current, options)
      grid.size = size

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
        targetElement.setAttribute('id', oldId)
        element.setAttribute('id', newId)

        targetElement.style.width = idToSize(oldId)
        targetElement.style.height = idToSize(oldId)
        element.style.width = idToSize(newId)
        element.style.height = idToSize(newId)

        targetElement.style.zIndex = idToZIndex(oldId)
        element.style.zIndex = idToZIndex(newId)

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

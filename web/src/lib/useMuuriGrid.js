// useMuuriGrid.js
import { useEffect } from 'react'
import Muuri from 'muuri'

const useMuuriGrid = (gridRef, options, addInstance, removeInstance, size) => {
  useEffect(() => {
    if (!gridRef.current) return

    const grid = new Muuri(gridRef.current, options)
    grid.size = size

    addInstance(grid)

    // Add event listener for dragReleaseEnd event
    grid.on('dragReleaseEnd', (item) => {
      // Get the grid of the item
      const itemGrid = item.getGrid()
      // Calculate the new height
      const newHeight = itemGrid.size

      console.log({ item, newHeight, gridHeight: itemGrid._height, itemGrid })
      // Set the new height to the item's element
      item.getElement().style.height = `${newHeight}px`
      item.getElement().style.width = `${newHeight}px`
      itemGrid.refreshItems()
    })

    return () => {
      // Remove event listener for dragReleaseEnd event
      grid.off('dragReleaseEnd')
      removeInstance(grid)
      grid.destroy()
    }
  }, [gridRef, options, addInstance, removeInstance])
}

export default useMuuriGrid

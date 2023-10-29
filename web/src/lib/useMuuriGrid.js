// useMuuriGrid.js
import { useEffect } from 'react'
import Muuri from 'muuri'
// Function to check if a grid is a "full" triangle
function isFullTriangle(grid) {
  // Assuming a full triangle grid contains 3 items
  console.log('IS FULL TRIANGLE', grid)
  if (!grid) return false
  return grid.getItems().length >= 3
}
const useMuuriGrid = (gridRef, options, addInstance, removeInstance, size) => {
  useEffect(() => {
    try {
      if (!gridRef.current) return

      const grid = new Muuri(gridRef.current, options)
      grid.size = size

      addInstance(grid)

      // Add event listener for dragReleaseEnd event
      grid.on('dragReleaseEnd', (item) => {
        const oldGrid = item.getGrid()
        const newGrid = item._drag._grid

        // If the item was dragged to a different grid
        if (oldGrid !== newGrid) {
          // Check if the newGrid is a "full" triangle
          if (isFullTriangle(newGrid)) {
            // If it's a full triangle, move the item back to the old grid
            oldGrid.add(item)
            // You may also want to refresh the grid to update the item positions

            // Calculate the new height
            const newHeight = newGrid.size

            console.log({
              item,
              newHeight,
              gridHeight: newGrid._height,
              newGrid,
            })
            // Set the new height to the item's element
            item.getElement().style.height = `${newHeight}px`
            item.getElement().style.width = `${newHeight}px`
            // Refresh the old grid to update the item positions
            if (oldGrid) oldGrid.refreshItems().layout()
          } else {
            // Otherwise, allow the item to be added to the new grid
            // You may also want to refresh the new grid to update the item positions
            if (newGrid) newGrid.refreshItems().layout()
          }
        }
      })

      return () => {
        // Remove event listener for dragReleaseEnd event
        grid.off('dragReleaseEnd')
        removeInstance(grid)
        try {
          grid.destroy()
        } catch (e) {
          console.log(e)
        }
      }
    } catch (e) {
      console.log(e)
    }
  }, [gridRef, options, addInstance, removeInstance])
}

export default useMuuriGrid

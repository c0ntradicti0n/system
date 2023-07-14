export const hoverObjects = []
export const addHoverObject = (id) => {
  hoverObjects.push(id)
}
export const removeHoverObject = (id) => {
  const index = hoverObjects.indexOf(id)
  if (index > -1) {
    hoverObjects.splice(index, 1)
  }
}

export const hoverObjects = new Set();

export const addHoverObject = (id) => {
  if (!hoverObjects.has(id)) {
    hoverObjects.add(id);
  }
}

export const removeHoverObject = (id) => {
  if (hoverObjects.has(id)) {
    hoverObjects.delete(id);
  }
}
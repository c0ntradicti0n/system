export function getLeftPosition(index, parentSize) {
  switch (index) {
    case 0:
      return 0
    case 1:
      return parentSize / 2
    case 2:
      return parentSize / 4
    default:
      console.log('Invalid index for subtriangle', index)
  }
}

export function getTopPosition(index, parentSize) {
  switch (index) {
    case 0:
      return parentSize / 2
    case 1:
      return parentSize / 2
    case 2:
      return 0
    default:
      console.log('Invalid index for subtriangle', index)
  }
}

export function isElementInViewportAndBigAndNoChildren(el, data) {
  var rect = el.getBoundingClientRect()

  if (!rect) return false
  return (
    rect.top >=
      -(window.innerHeight || document.documentElement.clientHeight) * 0.15 &&
    rect.left >=
      -(window.innerHeight || document.documentElement.clientHeight) * 0.15 &&
    rect.bottom <=
      (window.innerHeight || document.documentElement.clientHeight) * 1.15 &&
    rect.right <=
      (window.innerWidth || document.documentElement.clientWidth) * 1.15 &&
    rect.height > window.innerHeight * 0.3
    //typeof data !== 'object'
  )
}

export function postProcessTitle(title) {
  if (!title || typeof title == 'object') return null
  return title?.replace('.md', '')?.replace('_.', '')
}

export const pointInTriangle = (px, py, ax, ay, bx, by, cx, cy) => {
  const v0 = [cx - ax, cy - ay]
  const v1 = [bx - ax, by - ay]
  const v2 = [px - ax, py - ay]

  const dot00 = v0[0] * v0[0] + v0[1] * v0[1]
  const dot01 = v0[0] * v1[0] + v0[1] * v1[1]
  const dot02 = v0[0] * v2[0] + v0[1] * v2[1]
  const dot11 = v1[0] * v1[0] + v1[1] * v1[1]
  const dot12 = v1[0] * v2[0] + v1[1] * v2[1]

  const invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
  const u = (dot11 * dot02 - dot01 * dot12) * invDenom
  const v = (dot00 * dot12 - dot01 * dot02) * invDenom

  return u >= 0 && v >= 0 && u + v < 1
}

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

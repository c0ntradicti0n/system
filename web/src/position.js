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

export const stringToColour = (str, alpha = 1) => {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 7) - hash)
  }

  let color = ''
  for (let i = 0; i < 3; i++) {
    const value = (hash >> (i * 8)) & 0xff
    color += ('00' + value.toString(18)).substr(-2)
  }

  return (
    'rgba(' +
    parseInt(color.substr(0, 2), 16) +
    ',' +
    parseInt(color.substr(2, 2), 16) +
    ',' +
    parseInt(color.substr(4, 2), 16) +
    ',' +
    alpha +
    ')'
  )
}

export function isElementInViewportAndBigAndNoChildren(el, data) {
  var rect = el.getBoundingClientRect()

  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <=
      (window.innerHeight ||
        document.documentElement.clientHeight) &&
    rect.right <=
      (window.innerWidth ||
        document.documentElement.clientWidth) &&
    rect.height > window.innerHeight * 0.4 &&
    typeof data !== 'object'
  )
}

export function postProcessTitle(title) {
  return title?.replace('.md', '').replace('_.', '')
}

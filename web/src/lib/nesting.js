export const mergeDeep = (target, source) => {
  const isObject = (obj) => obj && typeof obj === 'object'

  if (!isObject(target) || !isObject(source)) {
    return source
  }

  Object.keys(source).forEach((key) => {
    const targetValue = target[key]
    const sourceValue = source[key]

    if (!sourceValue) return

    if (Array.isArray(targetValue) && Array.isArray(sourceValue)) {
      target[key] = targetValue.concat(sourceValue)
    } else if (isObject(targetValue) && isObject(sourceValue)) {
      target[key] = mergeDeep(targetValue, sourceValue)
    } else {
      target[key] = sourceValue
    }
  })

  return target
}

export function lookupDeep(key, data) {
  return splitKey(key).reduce((xs, x) => (xs && xs[x] ? xs[x] : null), data)
}

export function splitKey(key) {
  return key.split('/').filter((x) => x)
}
export function shiftIn(firstPath, secondPath, options) {
  // Remove leading slashes from paths
  if (firstPath.startsWith('/')) {
    firstPath = firstPath.slice(1)
  }

  if (secondPath.startsWith('/')) {
    secondPath = secondPath.slice(1)
  }

  // Split paths into their component parts
  const firstParts = firstPath.split('/')
  const secondParts = secondPath.split('/')

  // Check direction of shift
  if (options && options.left === true) {
    // Left shift
    if (!firstPath) {
      return [secondParts[0], secondParts.slice(1).join('/')]
    } else {
      let shiftIndex = firstParts.length
      return [
        secondParts.slice(0, shiftIndex + 1).join('/'),
        secondParts.slice(shiftIndex + 1).join('/'),
      ]
    }
  } else {
    // Right shift
    let shiftIndex = firstParts.length - 1
    if (!firstPath || !secondPath.startsWith(firstPath)) {
      shiftIndex-- // Exclude the last path of firstPath
    }
    shiftIndex = shiftIndex < 0 ? 0 : shiftIndex // Ensure shiftIndex is not negative
    return [
      secondParts.slice(0, shiftIndex).join('/'),
      secondParts.slice(shiftIndex).join('/'),
    ]
  }
}

export function shiftHorizontal(inputPath, direction) {
  let currentPath = inputPath.replace(/\//g, '')
  if (direction === 'right') {
    if (currentPath === '') return '1'

    let index = currentPath.length - 1
    while (index >= 0 && currentPath[index] === '3') {
      index--
    }

    if (index < 0) {
      currentPath = currentPath + '1'
    } else {
      const char = currentPath[index]
      const incrementedChar = (parseInt(char, 10) + 1).toString()
      currentPath =
        currentPath.substring(0, index) +
        incrementedChar +
        currentPath.substring(index + 1)
    }
  } else if (direction === 'left') {
    if (currentPath === '') return ''

    const lastChar = currentPath[currentPath.length - 1]
    if (lastChar === '1') {
      currentPath = currentPath.substring(0, currentPath.length - 1)
    } else if (lastChar === '2') {
      currentPath = currentPath.substring(0, currentPath.length - 1) + '1'
    } else if (lastChar === '3') {
      currentPath = currentPath.substring(0, currentPath.length - 1) + '2'
    }
  } else if (direction === 'zoom-in') {
    currentPath += '1'
  }

  return currentPath
}

export function shiftVertical(inputPath, direction) {
  if (direction === 'lower') {
    return inputPath + '1'
  } else if (direction === 'higher' && inputPath !== '') {
    return inputPath.slice(0, -1)
  }

  return inputPath
}

export function slashIt(currentPath) {
  const str = currentPath.replace(/\/+/g, '')
  return str.split('').join('/')
}
export function removeMultipleSlashes(str) {
  return str.replace(/\/+/g, '/')
}

export function shiftSideways(currentPath, direction) {
  const res = shiftHorizontal(currentPath, direction)
  return res
}

export function _shift(currentPath, direction) {
  if (['left', 'right'].includes(direction)) {
    return shiftSideways(currentPath, direction)
  }
  if (['lower', 'higher'].includes(direction)) {
    return shiftVertical(currentPath, direction)
  }
}

export function shift(currentPath, direction) {
  return slashIt(_shift(currentPath, direction))
}

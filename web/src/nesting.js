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

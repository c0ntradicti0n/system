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

export function lookupDeep(keys, data) {
  return keys.reduce((xs, x) => (xs && xs[x] ? xs[x] : null), data)
}

export function splitKey(key) {
  return key.split('/').filter((x) => x)
}

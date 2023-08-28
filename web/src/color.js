function filterString(str) {
  return str
    .split('')
    .filter((char) => ['1', '2', '3'].includes(char))
    .join('')
}

export const stringToColour = (str, alpha = 1) => {
  console.log('color', str)
  if (!str) return '#fff'
  str = filterString(str)
  const colors = {
    1: { r: 255, g: 255, b: 0 }, // yellow
    2: { r: 255, g: 0, b: 0 }, // red
    3: { r: 0, g: 0, b: 255 }, // blue
  }

  let r = 0,
    g = 0,
    b = 0

  // Loop through the string and add up the colors
  for (let s of str) {
    r += colors[s].r
    g += colors[s].g
    b += colors[s].b
  }

  // Average the colors based on the length of the string
  r = Math.min(255, r / str.length)
  g = Math.min(255, g / str.length)
  b = Math.min(255, b / str.length)

  return `rgba(${r},${g},${b},${alpha})`
}

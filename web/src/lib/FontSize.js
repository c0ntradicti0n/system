import { postProcessTitle } from './position'

function calculateFontSize(size, title, factor = 1) {
  // Calculate base font size
  const devicePixelRatio = window.devicePixelRatio || 1

  let baseFontSize = size / 30 / Math.log1p(devicePixelRatio)
  const shortTitle = (postProcessTitle(title ?? '') ?? '')
    .split(' ')
    .slice(0, 100)
    .join(' ')
  // Check the combined text length
  const combinedTextLength = shortTitle.length

  // Adjust font size if the combined text length is more than 100 chars
  if (combinedTextLength > 50) {
    baseFontSize *= 0.5 // you can adjust this factor to your needs
  }

  const fontSize = baseFontSize * factor
  return { shortTitle, fontSize }
}
export default calculateFontSize

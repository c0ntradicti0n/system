export const CONTROL_AREAS = {
  leftTriangle: {
    type: 'triangle',

    vertices: [
      { x: 0, y: (window.innerHeight * 2) / 3 }, // Bottom left
      { x: 0, y: 0 }, // Top left
      { x: window.innerWidth / 3, y: 0 }, // Middle top
    ],
  },
  rightTriagnle: {
    type: 'triangle',

    vertices: [
      { x: window.innerWidth * 0.66, y: 0 }, // Middle top
      { x: window.innerWidth, y: 0 }, // Top right
      { x: window.innerWidth, y: (window.innerHeight * 2) / 3 }, // Bottom right
    ],
  },
}

export const TEST_AREAS = {
  leftTriangle: {
    type: 'triangle',

    vertices: [
      { x: 0, y: (window.innerHeight * 2) / 3 }, // Bottom left
      { x: 0, y: 0 }, // Top left
      { x: window.innerWidth / 3, y: 0 }, // Middle top
    ],
  },
  rightTriangle: {
    type: 'triangle',

    vertices: [
      { x: window.innerWidth * 0.66, y: 0 }, // Middle top
      { x: window.innerWidth, y: 0 }, // Top right
      { x: window.innerWidth, y: (window.innerHeight * 2) / 3 }, // Bottom right
    ],
  },

  circle: {
    type: 'circle',
    cx: window.innerWidth / 2,
    cy: window.innerHeight / 2,
    r: window.innerWidth / 7,
  },
  circle2: {
    type: 'circle',
    cx: window.innerWidth / 3,
    cy: (window.innerHeight * 2) / 3,
    r: window.innerWidth / 9,
  },
  rectangle: {
    type: 'rectangle',

    vertices: [
      { x: window.innerWidth * 0.9, y: window.innerHeight / 2 },
      { x: window.innerWidth * 0.9, y: window.innerHeight },
      { x: window.innerWidth, y: window.innerHeight },
      { x: window.innerWidth, y: window.innerHeight / 2 },
    ],
  },
}

export const RIGHT_BOTTOM_CORNER = {
  rectangle: {
    type: 'rectangle',

    vertices: [
      { x: window.innerWidth * 0.7, y: (window.innerHeight * 3) / 4 },
      { x: window.innerWidth * 0.7, y: window.innerHeight },
      { x: window.innerWidth, y: window.innerHeight },
      { x: window.innerWidth, y: (window.innerHeight * 3) / 4 },
    ],
  },
}

export const RIGHT_BIG_TRIANGLE = {
  leftTriangle: {
    type: 'triangle',

    vertices: [
      { x: 0, y: window.innerHeight * 2 }, // Bottom left
      { x: 0, y: 0 }, // Top left
      { x: window.innerWidth / 4, y: 0 }, // Middle top
    ],
  },
  triangle: {
    type: 'triangle',

    vertices: [
      { x: window.innerWidth * 0.4, y: 0 },
      { x: window.innerWidth * 0.8, y: window.innerHeight },
      { x: window.innerWidth * 0.8, y: 0 },
    ],
  },
}

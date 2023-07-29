import { render, screen } from '@testing-library/react'
import App from '../App'
import { mergeDeep } from '../nesting'

test('merge obj', () => {
  const x = mergeDeep({ a: 1, b: 2 }, { a: 3, c: 4 })
  expect(x).toEqual({ a: 3, b: 2, c: 4 })
})

test('merge obj 2 ', () => {
  const x = mergeDeep({ a: { 2: 4 }, b: 2 }, { a: { g: 5 }, c: 4 })
  expect(x).toEqual({ a: { 2: 4, g: 5 }, b: 2, c: 4 })
})

import { shiftIn } from '../nesting'

test('shifting left from subset', () => {
  expect(shiftIn('1/2/3', '1/2/3/4/5/6', { left: true })).toEqual([
    '1/2/3/4',
    '5/6',
  ])
})

test('shifting left from empty string', () => {
  expect(shiftIn('', '1/2/3/4/5/6', { left: true })).toEqual(['1', '2/3/4/5/6'])
})

test('shifting left from complete path', () => {
  expect(shiftIn('1/2/3/4/5', '1/2/3/4/5/6', { left: true })).toEqual([
    '1/2/3/4/5/6',
    '',
  ])
})

test('shifting right from subset', () => {
  expect(shiftIn('1/2/3', '1/2/3/4/5/6', { left: false })).toEqual([
    '1/2',
    '3/4/5/6',
  ])
})

test('shifting right from empty string', () => {
  expect(shiftIn('', '1/2/3/4/5/6', { left: false })).toEqual([
    '',
    '1/2/3/4/5/6',
  ])
})

test('shifting right from complete path', () => {
  expect(shiftIn('1/2/3/4/5', '1/2/3/4/5/6', { left: false })).toEqual([
    '1/2/3/4',
    '5/6',
  ])
})

test('shifting right from path lengh', () => {
  expect(shiftIn('1', '1', { left: false })).toEqual(['', '1'])
})

test('shifting left from same paths with slash', () => {
  expect(shiftIn('/1/2', '/1/2', { left: true })).toEqual(['1/2', ''])
})

test('shifting right from same path with slash', () => {
  expect(shiftIn('/1/2', '/1/2', { left: false })).toEqual(['1', '2'])
})

test('shifting right wrong paths', () => {
  expect(shiftIn('3/2/1/1', '3/2/1', { left: false })).toEqual(['3/2', '1'])
})

import { shiftIn, _shift } from '../nesting'

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

describe('shift tests', () => {
  test('shifts right correctly', () => {
    expect(_shift('', 'right')).toBe('1')
    expect(_shift('1', 'right')).toBe('2')
    expect(_shift('2', 'right')).toBe('3')
    expect(_shift('3', 'right')).toBe('31')
    expect(_shift('31', 'right')).toBe('32')
    expect(_shift('33', 'right')).toBe('331')
  })

  test('_shifts left correctly', () => {
    expect(_shift('', 'left')).toBe('')
    expect(_shift('1', 'left')).toBe('')
    expect(_shift('2', 'left')).toBe('1')
    expect(_shift('3', 'left')).toBe('2')
    expect(_shift('31', 'left')).toBe('3')
    expect(_shift('32', 'left')).toBe('31')
    expect(_shift('33', 'left')).toBe('32')
    expect(_shift('331', 'left')).toBe('33')
  })

  test('_shift left and right are inverse operations', () => {
    const paths = ['', '1', '2', '3', '31', '32', '33', '331']
    for (const path of paths) {
      expect(_shift(_shift(path, 'right'), 'left')).toBe(path)
    }
  })

  test('_shifts lower correctly', () => {
    expect(_shift('', 'lower')).toBe('1')
    expect(_shift('1', 'lower')).toBe('11')
    expect(_shift('2', 'lower')).toBe('21')
    expect(_shift('3', 'lower')).toBe('31')
  })

  test('_shifts higher correctly', () => {
    expect(_shift('', 'higher')).toBe('')
    expect(_shift('1', 'higher')).toBe('')
    expect(_shift('11', 'higher')).toBe('1')
    expect(_shift('21', 'higher')).toBe('2')
    expect(_shift('31', 'higher')).toBe('3')
  })

  test('_shift lower and higher are inverse operations', () => {
    const paths = ['', '1', '2', '3', '11', '21', '31', '111']
    for (const path of paths) {
      expect(_shift(_shift(path, 'lower'), 'higher')).toBe(path)
    }
  })
})

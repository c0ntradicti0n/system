// Helper functions
import { lookupDeep, shift } from './nesting'

export function beamDataTo(
  newHiddenId,
  collectedData,
  hoverId,
  newRestId,
  detailId,
  transformComponentRef,
  setHiddenId,
  setVisualData,
) {
  const newData = lookupDeep(newHiddenId, collectedData)
  setHiddenId(newHiddenId || '')
  transformComponentRef.current.setTransform(0, 0, 1, 0, 0)
  setVisualData(newData)
}

export function go(params) {
  const newHiddenId = shift(params.hiddenId, params.direction)
  params.setDetailId(newHiddenId || '')
  beamDataTo(
    newHiddenId,
    params.collectedData,
    params.hoverId,
    '',
    params.detailId,
    params.transformComponentRef,
    params.setHiddenId,
    params.setVisualData,
  )
}

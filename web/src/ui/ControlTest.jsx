import { ControlContainer } from './ControlContainer'
import '../controls.css'
import { TEST_AREAS } from '../config/areas'

export const ControlContainerTest = () => {
  return (
    <ControlContainer areas={TEST_AREAS} cssPrefix="test" debug>
      {[...Array(200).keys()].map((i) => (
        <div key={i} style={{ backgroundColor: 'lime' }}>
          {' '}
          {i}{' '}
        </div>
      ))}
    </ControlContainer>
  )
}

import { ControlContainer } from './ControlContainer'
import './controls.css'
import { TEST_AREAS } from '../config/areas'

export const ControlContainerTest = () => {
  return (
    <ControlContainer areas={TEST_AREAS} cssPrefix="test" debug>
      {[...Array(650).keys()].map((i) => (
        <div key={i} style={{ border: '1px solit white' }}>
          {i}
        </div>
      ))}
    </ControlContainer>
  )
}

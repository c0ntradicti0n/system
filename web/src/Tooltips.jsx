import { getLeftPosition, getTopPosition, postProcessTitle } from './position'

export const Tooltips = ({ data,  id}) => {
  console.log("RENDERING TOOLTIPS", id)
  return (
    <div
      style={{
        position: 'static',
        pointerEvents: 'none',
        top: 0,
        left: 0,
        zIndex: 999999 ,
      }}
    >
      <div
        style={{
          pointerEvents: 'none',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 99999999,
        }}
      >
        <div style={{ flex: '1 0 auto', backgroundColor: '#000' }}>
          {postProcessTitle(data['.'])}
        </div>

        <div
          style={{
            pointerEvents: 'none',
            display: 'flex',
            flexDirection: 'row',
            flex: '2 0 auto',
          }}
        >
          <div
            style={{
              pointerEvents: 'none',
              display: 'flex',
              flexDirection: 'column',
              flex: '3 0 auto',
              backgroundColor: '#000',
            }}
          >
            {['1', '2', '3'].map((key) => {
              const value = data[key]
              const content = typeof value === 'object' ? value?.['.'] : value
              return (
                <div
                  key={key}
                  style={{
                    pointerEvents: 'none',
                    flex: '1 0 auto',
                    padding: '10px',
                  }}
                >
                  {key}.{postProcessTitle(content)}{' '}
                </div>
              )
            })}
          </div>

          <div
            style={{
              pointerEvents: 'none',
              flex: '3 3 auto',
              backgroundColor: '#000',
              padding: '10px',
            }}
          >
            {postProcessTitle(data['_'])}
          </div>
        </div>
      </div>
    </div>
  )
}

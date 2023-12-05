export const UnicodeIcon = ({ symbol, className, style }) => {
  return (
    <span className={className} style={{ fontFamily: 'Arial', ...style }}>
      {symbol}
    </span>
  )
}

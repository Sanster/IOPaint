import React, { ReactNode } from 'react'

interface Props {
  children: ReactNode
  className?: string
  style?: React.CSSProperties
}

const Flex: React.FC<Props> = props => {
  const { children, className, style } = props

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        ...style,
      }}
      className={className}
    >
      {children}
    </div>
  )
}

export default Flex

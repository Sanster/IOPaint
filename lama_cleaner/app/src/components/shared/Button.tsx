import React, { ReactNode } from 'react'
import Tooltip from './Tooltip'

interface ButtonProps {
  border?: boolean
  disabled?: boolean
  children?: ReactNode
  className?: string
  icon?: ReactNode
  toolTip?: string
  onKeyDown?: () => void
  onClick?: () => void
  onDown?: (ev: PointerEvent) => void
  onUp?: (ev: PointerEvent) => void
  onMouseEnter?: () => void
  onMouseLeave?: () => void
  style?: React.CSSProperties
}

const Button: React.FC<ButtonProps> = props => {
  const {
    children,
    border,
    className,
    disabled,
    icon,
    toolTip,
    onKeyDown,
    onClick,
    onDown,
    onUp,
    onMouseEnter,
    onMouseLeave,
    style,
  } = props

  const blurOnClick = (e: React.MouseEvent<HTMLDivElement>) => {
    e.currentTarget.blur()
    onClick?.()
  }

  const renderButton = () => {
    return (
      <div
        role="button"
        style={style}
        onKeyDown={onKeyDown}
        onClick={blurOnClick}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        onPointerDown={(ev: React.PointerEvent<HTMLDivElement>) => {
          onDown?.(ev.nativeEvent)
        }}
        onPointerUp={(ev: React.PointerEvent<HTMLDivElement>) => {
          onUp?.(ev.nativeEvent)
        }}
        tabIndex={-1}
        className={[
          'btn-primary',
          children ? 'btn-primary-content' : '',
          disabled === true ? 'btn-primary-disabled' : '',
          className,
          border ? `btn-border` : '',
        ].join(' ')}
      >
        {icon}
        {children}
      </div>
    )
  }

  if (toolTip) {
    return <Tooltip content={toolTip}>{renderButton()}</Tooltip>
  }
  return renderButton()
}

Button.defaultProps = {
  disabled: false,
  border: false,
}

export default Button

import React, { ReactNode } from 'react'

interface ButtonProps {
  border?: boolean
  disabled?: boolean
  children?: ReactNode
  className?: string
  icon?: ReactNode
  toolTip?: string
  tooltipPosition?: string
  onKeyDown?: () => void
  onClick?: () => void
  onDown?: (ev: PointerEvent) => void
  onUp?: (ev: PointerEvent) => void
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
    tooltipPosition,
    onKeyDown,
    onClick,
    onDown,
    onUp,
    style,
  } = props

  const blurOnClick = (e: React.MouseEvent<HTMLDivElement>) => {
    e.currentTarget.blur()
    onClick?.()
  }

  return (
    <div
      role="button"
      data-tooltip={toolTip}
      style={style}
      onKeyDown={onKeyDown}
      onClick={blurOnClick}
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
        toolTip ? 'info-tooltip' : '',
        tooltipPosition ? `info-tooltip-${tooltipPosition}` : '',
        className,
        border ? `btn-border` : '',
      ].join(' ')}
    >
      {icon}
      {children ? <span>{children}</span> : null}
    </div>
  )
}

Button.defaultProps = {
  disabled: false,
  border: false,
}

export default Button

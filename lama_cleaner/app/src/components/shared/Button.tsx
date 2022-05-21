import React, { ReactNode } from 'react'

interface ButtonProps {
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
      ].join(' ')}
    >
      {icon}
      {children ? <span>{children}</span> : null}
    </div>
  )
}

Button.defaultProps = {
  disabled: false,
}

export default Button

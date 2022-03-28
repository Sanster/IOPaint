import React, { ReactNode } from 'react'

interface ButtonProps {
  children?: ReactNode
  className?: string
  icon?: ReactNode
  disabled?: boolean
  onKeyDown?: () => void
  onClick?: () => void
  onDown?: (ev: PointerEvent) => void
  onUp?: (ev: PointerEvent) => void
}

export default function Button(props: ButtonProps) {
  const {
    children,
    className,
    disabled,
    icon,
    onKeyDown,
    onClick,
    onDown,
    onUp,
  } = props

  const blurOnClick = (e: React.MouseEvent<HTMLDivElement>) => {
    e.currentTarget.blur()
    onClick?.()
  }

  return (
    <div
      role="button"
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
        disabled ? 'btn-primary-disabled' : '',
        className,
      ].join(' ')}
    >
      {icon}
      {children ? <span>{children}</span> : null}
    </div>
  )
}

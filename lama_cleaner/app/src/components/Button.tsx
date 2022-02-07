import React, { ReactNode, useState } from 'react'

interface ButtonProps {
  children?: ReactNode
  className?: string
  icon?: ReactNode
  primary?: boolean
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
    primary,
    onKeyDown,
    onClick,
    onDown,
    onUp,
  } = props
  const [active, setActive] = useState(false)
  let background = ''
  if (primary && !disabled) {
    background = 'bg-primary hover:bg-black hover:text-white'
  }
  if (active) {
    background = 'bg-black text-white'
  }
  if (!primary && !active) {
    background = 'hover:bg-primary'
  }

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
        setActive(true)
        onDown?.(ev.nativeEvent)
      }}
      onPointerUp={(ev: React.PointerEvent<HTMLDivElement>) => {
        setActive(false)
        onUp?.(ev.nativeEvent)
      }}
      tabIndex={-1}
      className={[
        'inline-flex py-3 px-3 rounded-md cursor-pointer',
        children ? 'space-x-3' : '',
        background,
        disabled ? 'pointer-events-none opacity-50' : '',
        className,
      ].join(' ')}
    >
      {icon}
      <span className="whitespace-nowrap select-none">{children}</span>
    </div>
  )
}

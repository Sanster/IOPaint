import React from 'react'
import * as SwitchPrimitive from '@radix-ui/react-switch'

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  React.ComponentProps<typeof SwitchPrimitive.Root>
>((props, forwardedRef) => {
  const { className, ...itemProps } = props

  return (
    <SwitchPrimitive.Root
      {...itemProps}
      ref={forwardedRef}
      className={`switch-root ${className}`}
      onKeyDown={e => e.preventDefault()}
    />
  )
})

const SwitchThumb = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Thumb>,
  React.ComponentProps<typeof SwitchPrimitive.Thumb>
>((props, forwardedRef) => {
  const { className, ...itemProps } = props

  return (
    <SwitchPrimitive.Thumb
      {...itemProps}
      ref={forwardedRef}
      className={`switch-thumb ${className}`}
    />
  )
})

export { Switch, SwitchThumb }

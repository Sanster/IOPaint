import React, { ReactNode } from 'react'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { TooltipProps } from '@radix-ui/react-tooltip'

interface MyTooltipProps extends TooltipProps {
  content: string | ReactNode
  children: ReactNode
}

const Tooltip = (props: MyTooltipProps) => {
  const { content, children } = props

  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Provider>
        <TooltipPrimitive.Trigger className="tooltip-trigger" asChild>
          {children}
        </TooltipPrimitive.Trigger>

        <TooltipPrimitive.Content className="tooltip-content">
          {content}
          <TooltipPrimitive.Arrow className="tooltip-arrow" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Provider>
    </TooltipPrimitive.Root>
  )
}

export default Tooltip

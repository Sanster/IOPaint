import * as React from "react"

import { cn } from "@/lib/utils"
import { useStore } from "@/lib/states"

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    const updateAppState = useStore((state) => state.updateAppState)

    const handleOnFocus = () => {
      updateAppState({ disableShortCuts: true })
    }

    const handleOnBlur = () => {
      updateAppState({ disableShortCuts: false })
    }

    return (
      <textarea
        className={cn(
          "flex min-h-[60px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
          "overflow-auto",
          className
        )}
        tabIndex={-1}
        ref={ref}
        onFocus={handleOnFocus}
        onBlur={handleOnBlur}
        {...props}
      />
    )
  }
)
Textarea.displayName = "Textarea"

export { Textarea }

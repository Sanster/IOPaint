import * as React from "react"

import { cn } from "@/lib/utils"
import { useStore } from "@/lib/states"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    const updateAppState = useStore((state) => state.updateAppState)

    const handleOnFocus = () => {
      updateAppState({ disableShortCuts: true })
    }

    const handleOnBlur = () => {
      updateAppState({ disableShortCuts: false })
    }

    return (
      <input
        type={type}
        className={cn(
          "flex h-8 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        autoComplete="off"
        tabIndex={-1}
        onFocus={handleOnFocus}
        onBlur={handleOnBlur}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export interface NumberInputProps extends InputProps {
  numberValue: number
  allowFloat: boolean
  onNumberValueChange: (value: number) => void
}

const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ numberValue, allowFloat, onNumberValueChange, ...rest }, ref) => {
    const [value, setValue] = React.useState<string>(numberValue.toString())

    React.useEffect(() => {
      if (value !== numberValue.toString() + ".") {
        setValue(numberValue.toString())
      }
    }, [numberValue])

    const onInput = (evt: React.FormEvent<HTMLInputElement>) => {
      const target = evt.target as HTMLInputElement
      let val = target.value
      if (allowFloat) {
        val = val.replace(/[^0-9.]/g, "").replace(/(\..*?)\..*/g, "$1")
        if (val.length === 0) {
          onNumberValueChange(0)
          return
        }
        // val = parseFloat(val).toFixed(2)
        onNumberValueChange(parseFloat(val))
      } else {
        val = val.replace(/\D/g, "")
        if (val.length === 0) {
          onNumberValueChange(0)
          return
        }
        onNumberValueChange(parseInt(val, 10))
      }
      setValue(val)
    }

    return <Input ref={ref} value={value} onInput={onInput} {...rest} />
  }
)

export { Input, NumberInput }

import React, { FormEvent, InputHTMLAttributes } from 'react'

interface NumberInputProps extends InputHTMLAttributes<HTMLInputElement> {
  value: string
  onValue?: (val: string) => void
}

const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  (props: NumberInputProps, forwardedRef) => {
    const { value, onValue, ...itemProps } = props

    const handleOnInput = (evt: FormEvent<HTMLInputElement>) => {
      const target = evt.target as HTMLInputElement
      const val = target.value.replace(/\D/g, '')
      onValue?.(val)
    }

    return (
      <input
        value={value}
        onInput={handleOnInput}
        className="number-input"
        {...itemProps}
        ref={forwardedRef}
        type="text"
      />
    )
  }
)

export default NumberInput

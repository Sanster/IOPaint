import React, {
  FormEvent,
  InputHTMLAttributes,
  useEffect,
  useState,
} from 'react'
import TextInput from './Input'

interface NumberInputProps extends InputHTMLAttributes<HTMLInputElement> {
  value: string
  allowFloat?: boolean
  onValue?: (val: string) => void
}

const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  (props: NumberInputProps, forwardedRef) => {
    const { value, allowFloat, onValue, ...itemProps } = props
    const [innerValue, setInnerValue] = useState(value)

    useEffect(() => {
      setInnerValue(value)
    }, [value])

    const handleOnInput = (evt: FormEvent<HTMLInputElement>) => {
      const target = evt.target as HTMLInputElement
      let val = target.value
      if (allowFloat) {
        val = val.replace(/[^0-9.]/g, '').replace(/(\..*?)\..*/g, '$1')
        onValue?.(val)
      } else {
        val = val.replace(/\D/g, '')
        onValue?.(val)
      }
      setInnerValue(val)
    }

    return (
      <TextInput
        value={innerValue}
        onInput={handleOnInput}
        className="number-input"
        {...itemProps}
        ref={forwardedRef}
      />
    )
  }
)

NumberInput.defaultProps = {
  allowFloat: false,
}

export default NumberInput

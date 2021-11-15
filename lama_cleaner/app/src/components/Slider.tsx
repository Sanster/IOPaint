import React from 'react'

type SliderProps = {
  label?: any
  value?: number
  min?: number
  max?: number
  onChange: (value: number) => void
}

export default function Slider(props: SliderProps) {
  const { value, onChange, label, min, max } = props

  const step = ((max || 100) - (min || 0)) / 100

  return (
    <div className="inline-flex items-center space-x-4 text-black">
      <span>{label}</span>
      <input
        className={[
          'appearance-none rounded-lg h-4',
          'bg-primary',
          'w-24 md:w-auto',
        ].join(' ')}
        type="range"
        step={step}
        min={min}
        max={max}
        value={value}
        onChange={ev => {
          ev.preventDefault()
          ev.stopPropagation()
          onChange(parseInt(ev.currentTarget.value, 10))
        }}
      />
    </div>
  )
}

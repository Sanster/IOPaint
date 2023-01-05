import React from 'react'

type SliderProps = {
  label?: any
  value?: number
  min?: number
  max?: number
  onChange: (value: number) => void
  onClick?: () => void
  width?: number
}

export default function Slider(props: SliderProps) {
  const { value, onChange, onClick, label, min, max, width } = props
  const styles: any = {}
  if (width !== undefined) {
    styles.width = width
  }

  const step = ((max || 100) - (min || 0)) / 100

  const onMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
    e.currentTarget?.blur()
  }

  return (
    <div className="editor-brush-slider">
      <span>{label}</span>
      <input
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
        onClick={onClick}
        style={styles}
        onMouseUp={onMouseUp}
      />
    </div>
  )
}

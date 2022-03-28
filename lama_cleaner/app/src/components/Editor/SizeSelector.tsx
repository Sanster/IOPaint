import React, { FocusEvent, useCallback, useRef } from 'react'

const sizes = ['720', '1080', '2000', 'Original']

type SizeSelectorProps = {
  value: string
  originalWidth: number
  originalHeight: number
  onChange: (value: string) => void
}

export default function SizeSelector(props: SizeSelectorProps) {
  const { value, originalHeight, originalWidth, onChange } = props
  const selectRef = useRef()

  const getSizeShowName = (size: string) => {
    if (size === 'Original') {
      return `${originalWidth}x${originalHeight}`
    }
    const length: number = parseInt(size, 10)
    const longSide: number =
      originalWidth > originalHeight ? originalWidth : originalHeight
    const scale = length / longSide

    if (originalWidth > originalHeight) {
      const newHeight = Math.ceil(scale * originalHeight)
      return `${size}x${newHeight}`
    }
    const newWidth = Math.ceil(scale * originalWidth)
    return `${newWidth}x${size}`
  }

  const onButtonFocus = (e: FocusEvent<any>) => {
    e.currentTarget.blur()
  }

  const getValidSizes = useCallback((): string[] => {
    const longSide: number =
      originalWidth > originalHeight ? originalWidth : originalHeight

    const validSizes = []
    for (let i = 0; i < sizes.length; i += 1) {
      const s = sizes[i]
      if (s === 'Original') {
        validSizes.push(s)
      } else if (parseInt(s, 10) <= longSide) {
        validSizes.push(s)
      }
    }
    return validSizes
  }, [originalHeight, originalWidth])

  const getValidSize = useCallback(() => {
    if (getValidSizes().indexOf(value) === -1) {
      return getValidSizes()[0]
    }
    return value
  }, [value, getValidSizes])

  const sizeChangeHandler = (e: any) => {
    onChange(e.target.value)
    e.target.blur()
  }

  return (
    <div className="editor-size-selector">
      <p>Size:</p>
      <select value={getValidSize()} onChange={sizeChangeHandler}>
        {getValidSizes().map(size => (
          <option key={size} value={size}>
            {getSizeShowName(size)}
          </option>
        ))}
      </select>
    </div>
  )
}

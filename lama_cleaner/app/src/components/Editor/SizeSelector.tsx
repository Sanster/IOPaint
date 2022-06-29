import React, { useCallback, useEffect, useState } from 'react'
import Selector from '../shared/Selector'

const sizes = ['720', '1080', '2000', 'Original']

type SizeSelectorProps = {
  originalWidth: number
  originalHeight: number
  onChange: (value: number) => void
}

export default function SizeSelector(props: SizeSelectorProps) {
  const { originalHeight, originalWidth, onChange } = props
  const [activeSize, setActiveSize] = useState<string>('Original')
  const longSide: number = Math.max(originalWidth, originalHeight)

  const getSizeShowName = useCallback(
    (size: string) => {
      if (size === 'Original') {
        return `${originalWidth}x${originalHeight}`
      }
      const scale = parseInt(size, 10) / longSide
      if (originalWidth > originalHeight) {
        const newHeight = Math.ceil(originalHeight * scale)
        return `${size}x${newHeight}`
      }
      const newWidth = Math.ceil(originalWidth * scale)
      return `${newWidth}x${size}`
    },
    [originalWidth, originalHeight, longSide]
  )

  const getValidSizes = useCallback(() => {
    const validSizes: string[] = []
    for (let i = 0; i < sizes.length; i += 1) {
      if (sizes[i] === 'Original') {
        validSizes.push(getSizeShowName(sizes[i]))
      }
      if (parseInt(sizes[i], 10) < longSide) {
        validSizes.push(getSizeShowName(sizes[i]))
      }
    }
    return validSizes
  }, [longSide, getSizeShowName])

  const sizeChangeHandler = (value: string) => {
    const currentRes = value.split('x')
    if (originalWidth > originalHeight) {
      setActiveSize(currentRes[0])
      onChange(parseInt(currentRes[0], 10))
    } else {
      setActiveSize(currentRes[1])
      onChange(parseInt(currentRes[1], 10))
    }
  }

  return (
    <Selector
      width={100}
      autoFocusAfterClose={false}
      value={getSizeShowName(activeSize.toString())}
      options={getValidSizes()}
      onChange={sizeChangeHandler}
      chevronDirection="up"
    />
  )
}

import React, { useCallback, useRef, useState } from 'react'
import ChevronDoubleDownIcon from '@heroicons/react/solid/ChevronDoubleDownIcon'
import { useClickAway } from 'react-use'

const sizes = ['720', '1080', '2000', 'Original']

type SizeSelectorProps = {
  originalWidth: number
  originalHeight: number
  onChange: (value: number) => void
}

export default function SizeSelector(props: SizeSelectorProps) {
  const { originalHeight, originalWidth, onChange } = props
  const [showOptions, setShowOptions] = useState<boolean>(false)
  const sizeSelectorRef = useRef(null)
  const [activeSize, setActiveSize] = useState<string>('Original')
  const longSide: number = Math.max(originalWidth, originalHeight)

  const getValidSizes = useCallback(() => {
    const validSizes: string[] = []
    for (let i = 0; i < sizes.length; i += 1) {
      if (sizes[i] === 'Original') {
        validSizes.push(sizes[i])
      }
      if (parseInt(sizes[i], 10) < longSide) {
        validSizes.push(sizes[i])
      }
    }
    return validSizes
  }, [longSide])

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

  const showOptionsHandler = () => {
    setShowOptions(currentShowOptionsState => !currentShowOptionsState)
  }

  useClickAway(sizeSelectorRef, () => {
    setShowOptions(false)
  })

  const sizeChangeHandler = (e: any) => {
    const currentRes = e.target.textContent.split('x')
    if (originalWidth > originalHeight) {
      setActiveSize(currentRes[0])
      onChange(currentRes[0])
    } else {
      setActiveSize(currentRes[1])
      onChange(currentRes[1])
    }
    setShowOptions(!showOptions)
  }

  return (
    <div className="editor-size-selector" ref={sizeSelectorRef}>
      <p>Size:</p>
      <div
        className="editor-size-selector-main"
        role="button"
        tabIndex={0}
        onClick={showOptionsHandler}
        aria-hidden="true"
      >
        <p>{getSizeShowName(activeSize.toString())}</p>
        <div className="editor-size-selector-chevron">
          <ChevronDoubleDownIcon />
        </div>
      </div>

      {showOptions && (
        <div className="editor-size-options">
          {getValidSizes().map(size => (
            <div
              className="editor-size-option"
              role="button"
              tabIndex={0}
              key={size}
              onClick={sizeChangeHandler}
              aria-hidden="true"
            >
              {getSizeShowName(size)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

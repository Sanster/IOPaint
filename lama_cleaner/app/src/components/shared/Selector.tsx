import React, { MutableRefObject, useCallback, useRef, useState } from 'react'
import { useClickAway, useKeyPressEvent } from 'react-use'
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/outline'

type SelectorChevronDirection = 'up' | 'down'

type SelectorProps = {
  minWidth?: number
  chevronDirection?: SelectorChevronDirection
  value: string
  options: string[]
  onChange: (value: string) => void
}

const selectorDefaultProps = {
  minWidth: 128,
  chevronDirection: 'down',
}

function Selector(props: SelectorProps) {
  const { minWidth, chevronDirection, value, options, onChange } = props
  const [showOptions, setShowOptions] = useState<boolean>(false)
  const selectorRef = useRef<HTMLDivElement | null>(null)

  const showOptionsHandler = () => {
    // console.log(selectorRef.current?.focus)
    // selectorRef?.current?.focus()
    setShowOptions(currentShowOptionsState => !currentShowOptionsState)
  }

  useClickAway(selectorRef, () => {
    setShowOptions(false)
  })

  // TODO: how to prevent Modal close?
  // useKeyPressEvent('Escape', (e: KeyboardEvent) => {
  //   if (showOptions === true) {
  //     console.log(`selector ${e}`)
  //     e.preventDefault()
  //     e.stopPropagation()
  //     setShowOptions(false)
  //   }
  // })

  const onOptionClick = (e: any, newIndex: number) => {
    const currentRes = e.target.textContent.split('x')
    onChange(currentRes[0])
    setShowOptions(false)
  }

  return (
    <div className="selector" ref={selectorRef} style={{ minWidth }}>
      <div
        className="selector-main"
        role="button"
        onClick={showOptionsHandler}
        aria-hidden="true"
      >
        <p>{value}</p>
        <div className="selector-icon">
          {chevronDirection === 'up' ? <ChevronUpIcon /> : <ChevronDownIcon />}
        </div>
      </div>

      {showOptions && (
        <div className="selector-options">
          {options.map((val, _index) => (
            <div
              className="selector-option"
              role="button"
              tabIndex={0}
              key={val}
              onClick={e => onOptionClick(e, _index)}
              aria-hidden="true"
            >
              {val}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

Selector.defaultProps = selectorDefaultProps
export default Selector

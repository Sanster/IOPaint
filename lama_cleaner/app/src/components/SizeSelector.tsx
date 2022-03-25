import React, { FocusEvent, useCallback } from 'react'
import { Listbox } from '@headlessui/react'
import { CheckIcon, SelectorIcon } from '@heroicons/react/solid'

const sizes = ['720', '1080', '2000', 'Original']

type SizeSelectorProps = {
  value: string
  originalWidth: number
  originalHeight: number
  onChange: (value: string) => void
}

export default function SizeSelector(props: SizeSelectorProps) {
  const { value, originalHeight, originalWidth, onChange } = props

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

  return (
    <div className="w-32">
      <Listbox value={getValidSize()} onChange={onChange}>
        <div className="relative">
          <Listbox.Options
            style={{ top: `-${getValidSizes().length * 40 + 5}px` }}
            className="absolute mb-1 w-full overflow-auto text-base bg-opacity-10 bg-black backdrop-blur rounded-md max-h-60 outline-none sm:text-sm"
          >
            {getValidSizes().map(size => (
              <Listbox.Option
                key={size}
                className={({ active }) =>
                  `${active ? 'bg-black bg-opacity-10' : 'text-gray-900'}
                   cursor-default select-none relative py-2 pl-4 pr-4`
                }
                value={size}
              >
                {({ selected, active }) => (
                  <>
                    <span
                      className={`${
                        selected ? 'font-medium' : 'font-normal'
                      } block truncate`}
                    >
                      {getSizeShowName(size)}
                    </span>
                    {/* {selected ? (
                      <span
                        className={`${
                          active ? 'text-amber-600' : 'text-amber-600'
                        }
                                absolute inset-y-0 left-0 flex items-center pl-3`}
                      >
                        <CheckIcon className="w-5 h-5" aria-hidden="true" />
                      </span>
                    ) : null} */}
                  </>
                )}
              </Listbox.Option>
            ))}
          </Listbox.Options>
          <Listbox.Button
            onFocus={onButtonFocus}
            className="relative w-full inline-flex w-full px-4 py-2 text-sm font-medium bg-black rounded-md bg-opacity-10  focus:outline-none "
          >
            <span className="block truncate">
              {getSizeShowName(getValidSize())}
            </span>
            <span className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
              <SelectorIcon
                className="w-5 h-5 text-gray-400"
                aria-hidden="true"
              />
            </span>
          </Listbox.Button>
        </div>
      </Listbox>
    </div>
  )
}

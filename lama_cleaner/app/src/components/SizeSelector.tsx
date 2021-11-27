import React from 'react'
import { RadioGroup } from '@headlessui/react'

const sizes = [
  ['1080', '1080'],
  ['2000', '2k'],
  ['Original', 'Original'],
]

type SizeSelectorProps = {
  value?: string
  originalSize: string
  onChange: (value: string) => void
}

export default function SizeSelector(props: SizeSelectorProps) {
  const { value, originalSize, onChange } = props

  return (
    <RadioGroup
      className="my-4 flex items-center space-x-2"
      value={value}
      onChange={onChange}
    >
      <RadioGroup.Label>Resize</RadioGroup.Label>
      {sizes.map(size => (
        <RadioGroup.Option key={size[0]} value={size[0]}>
          {({ checked }) => (
            <div
              className={[
                checked ? 'bg-gray-200' : 'border-opacity-10',
                'border-3 px-2 py-2 rounded-md',
              ].join(' ')}
            >
              <div className="flex items-center space-x-4">
                <div
                  className={[
                    'rounded-full w-5 h-5 border-4 ',
                    checked
                      ? 'border-primary bg-black'
                      : 'border-black border-opacity-10',
                  ].join(' ')}
                />
                {size[0] === 'Original' ? (
                  <span>{`${size[1]}(${originalSize})`}</span>
                ) : (
                  <span>{size[1]}</span>
                )}
              </div>
            </div>
          )}
        </RadioGroup.Option>
      ))}
    </RadioGroup>
  )
}

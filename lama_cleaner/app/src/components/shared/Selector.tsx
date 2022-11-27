import React, { useRef } from 'react'
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline'
import * as Select from '@radix-ui/react-select'

type SelectorChevronDirection = 'up' | 'down'

interface Props {
  width?: number
  value: string
  options: string[]
  chevronDirection?: SelectorChevronDirection
  autoFocusAfterClose?: boolean
  onChange: (value: string) => void
  disabled?: boolean
}

const Selector = (props: Props) => {
  const {
    width,
    value,
    chevronDirection,
    options,
    autoFocusAfterClose,
    onChange,
    disabled,
  } = props

  const contentRef = useRef<HTMLButtonElement>(null)

  const onOpenChange = (open: boolean) => {
    if (!open) {
      if (!autoFocusAfterClose) {
        // 如果使用 Select.Content 的 onCloseAutoFocus 来取消 focus（防止空格继续打开这个 select）
        // 会导致其它快捷键失效，原因未知
        window.setTimeout(() => {
          contentRef?.current?.blur()
        }, 100)
      }
    }
  }

  return (
    <Select.Root
      value={value}
      onValueChange={onChange}
      onOpenChange={onOpenChange}
    >
      <Select.Trigger
        className="select-trigger"
        style={{ width }}
        ref={contentRef}
        onKeyDown={e => e.preventDefault()}
        disabled={disabled}
      >
        <Select.Value />
        <Select.Icon>
          {chevronDirection === 'up' ? <ChevronUpIcon /> : <ChevronDownIcon />}
        </Select.Icon>
      </Select.Trigger>

      <Select.Content className="select-content">
        <Select.Viewport className="select-viewport">
          {options.map(val => (
            <Select.Item value={val} className="select-item" key={val}>
              <Select.ItemText>{val}</Select.ItemText>
              <Select.ItemIndicator className="select-item-indicator">
                <CheckIcon />
              </Select.ItemIndicator>
            </Select.Item>
          ))}
        </Select.Viewport>
      </Select.Content>
    </Select.Root>
  )
}

const selectorDefaultProps = {
  chevronDirection: 'down',
  autoFocusAfterClose: true,
  disabled: false,
}

Selector.defaultProps = selectorDefaultProps

export default Selector

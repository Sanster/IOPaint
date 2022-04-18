import React from 'react'
import NumberInput from '../shared/NumberInput'
import SettingBlock from './SettingBlock'

interface NumberInputSettingProps {
  title: string
  value: string
  suffix?: string
  onValue: (val: string) => void
}

function NumberInputSetting(props: NumberInputSettingProps) {
  const { title, value, suffix, onValue } = props

  return (
    <SettingBlock
      className="sub-setting-block"
      title={title}
      input={
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <NumberInput
            style={{ width: '80px' }}
            value={`${value}`}
            onValue={onValue}
          />
          {suffix && <span>{suffix}</span>}
        </div>
      }
    />
  )
}

export default NumberInputSetting

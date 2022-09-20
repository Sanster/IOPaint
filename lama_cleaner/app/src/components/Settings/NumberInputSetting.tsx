import React from 'react'
import NumberInput from '../shared/NumberInput'
import SettingBlock from './SettingBlock'

interface NumberInputSettingProps {
  title: string
  allowFloat?: boolean
  desc?: string
  value: string
  suffix?: string
  width?: number
  widthUnit?: string
  disable?: boolean
  onValue: (val: string) => void
}

function NumberInputSetting(props: NumberInputSettingProps) {
  const {
    title,
    allowFloat,
    desc,
    value,
    suffix,
    onValue,
    width,
    widthUnit,
    disable,
  } = props

  return (
    <SettingBlock
      className="sub-setting-block"
      title={title}
      desc={desc}
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
            allowFloat={allowFloat}
            style={{ width: `${width}${widthUnit}` }}
            value={value}
            disabled={disable}
            onValue={onValue}
          />
          {suffix && <span>{suffix}</span>}
        </div>
      }
    />
  )
}

NumberInputSetting.defaultProps = {
  allowFloat: false,
  width: 80,
  widthUnit: 'px',
  disable: false,
}

export default NumberInputSetting

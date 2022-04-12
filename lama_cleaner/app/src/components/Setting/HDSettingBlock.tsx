import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import NumberInput from '../shared/NumberInput'
import Selector from '../shared/Selector'
import SettingBlock from './SettingBlock'

export enum HDStrategy {
  ORIGINAL = 'Original',
  REISIZE = 'Resize',
  CROP = 'Crop',
}

interface PixelSizeInputProps {
  title: string
  value: string
  onValue: (val: string) => void
}

function PixelSizeInputSetting(props: PixelSizeInputProps) {
  const { title, value, onValue } = props

  return (
    <SettingBlock
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
          <span>pixel</span>
        </div>
      }
    />
  )
}

function HDSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onStrategyChange = (value: HDStrategy) => {
    setSettingState(old => {
      return { ...old, hdStrategy: value }
    })
  }

  const onResizeLimitChange = (value: string) => {
    setSettingState(old => {
      return { ...old, hdStrategyResizeLimit: value }
    })
  }

  const onCropTriggerSizeChange = (value: string) => {
    setSettingState(old => {
      return { ...old, hdStrategyCropTrigerSize: value }
    })
  }

  const renderOriginalOptionDesc = () => {
    return (
      <div>
        Use the original resolution of the picture, suitable for picture size
        below 2K, of course you can try it on higher resolution pictures
      </div>
    )
  }

  const renderResizeOptionDesc = () => {
    return (
      <div>
        <div>
          Resize the longer side of the image to a specific size(keep ratio),
          then do inpainting on the entire resized image.
        </div>
        <PixelSizeInputSetting
          title="Size limit"
          value={`${setting.hdStrategyResizeLimit}`}
          onValue={onResizeLimitChange}
        />
      </div>
    )
  }

  const renderCropOptionDesc = () => {
    return (
      <div>
        <div>
          Crop masking area from the original image to do inpainting, and paste
          the result back. Mainly for performance and memory reasons on high
          resolution image.
        </div>
        <PixelSizeInputSetting
          title="Trigger size"
          value={`${setting.hdStrategyCropTrigerSize}`}
          onValue={onCropTriggerSizeChange}
        />
      </div>
    )
  }

  const renderHDStrategyOptionDesc = (): ReactNode => {
    switch (setting.hdStrategy) {
      case HDStrategy.ORIGINAL:
        return renderOriginalOptionDesc()
      case HDStrategy.CROP:
        return renderCropOptionDesc()
      case HDStrategy.REISIZE:
        return renderResizeOptionDesc()
      default:
        return renderOriginalOptionDesc()
    }
  }

  return (
    <SettingBlock
      title="High Resolution Strategy"
      input={
        <Selector
          options={Object.values(HDStrategy)}
          onChange={val => onStrategyChange(val as HDStrategy)}
        />
      }
      optionDesc={renderHDStrategyOptionDesc()}
    />
  )
}

export default HDSettingBlock

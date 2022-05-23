import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import Selector from '../shared/Selector'
import NumberInputSetting from './NumberInputSetting'
import SettingBlock from './SettingBlock'

export enum HDStrategy {
  ORIGINAL = 'Original',
  RESIZE = 'Resize',
  CROP = 'Crop',
}

function HDSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onStrategyChange = (value: HDStrategy) => {
    setSettingState(old => {
      return { ...old, hdStrategy: value }
    })
  }

  const onResizeLimitChange = (value: string) => {
    const val = value.length === 0 ? 0 : parseInt(value, 10)
    setSettingState(old => {
      return { ...old, hdStrategyResizeLimit: val }
    })
  }

  const onCropTriggerSizeChange = (value: string) => {
    const val = value.length === 0 ? 0 : parseInt(value, 10)
    setSettingState(old => {
      return { ...old, hdStrategyCropTrigerSize: val }
    })
  }

  const onCropMarginChange = (value: string) => {
    const val = value.length === 0 ? 0 : parseInt(value, 10)
    setSettingState(old => {
      return { ...old, hdStrategyCropMargin: val }
    })
  }

  const renderOriginalOptionDesc = () => {
    return (
      <div>
        Use the original resolution of the picture, suitable for picture size
        below 2K. Try{' '}
        <div
          tabIndex={0}
          role="button"
          className="inline-tip"
          onClick={() => onStrategyChange(HDStrategy.RESIZE)}
        >
          Resize Strategy
        </div>{' '}
        if you do not get good results on high resolution images.
      </div>
    )
  }

  const renderResizeOptionDesc = () => {
    return (
      <div>
        <div>
          Resize the longer side of the image to a specific size(keep ratio),
          then do inpainting on the resized image.
        </div>
        <NumberInputSetting
          title="Size limit"
          value={`${setting.hdStrategyResizeLimit}`}
          suffix="pixel"
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
        <NumberInputSetting
          title="Trigger size"
          value={`${setting.hdStrategyCropTrigerSize}`}
          suffix="pixel"
          onValue={onCropTriggerSizeChange}
        />
        <NumberInputSetting
          title="Crop margin"
          value={`${setting.hdStrategyCropMargin}`}
          suffix="pixel"
          onValue={onCropMarginChange}
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
      case HDStrategy.RESIZE:
        return renderResizeOptionDesc()
      default:
        return renderOriginalOptionDesc()
    }
  }

  return (
    <SettingBlock
      className="hd-setting-block"
      title="High Resolution Strategy"
      input={
        <Selector
          width={80}
          value={setting.hdStrategy as string}
          options={Object.values(HDStrategy)}
          onChange={val => onStrategyChange(val as HDStrategy)}
        />
      }
      optionDesc={renderHDStrategyOptionDesc()}
    />
  )
}

export default HDSettingBlock

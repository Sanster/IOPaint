import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { hdSettingsState, settingState } from '../../store/Atoms'
import Selector from '../shared/Selector'
import NumberInputSetting from './NumberInputSetting'
import SettingBlock from './SettingBlock'

export enum HDStrategy {
  ORIGINAL = 'Original',
  RESIZE = 'Resize',
  CROP = 'Crop',
}

export enum LDMSampler {
  ddim = 'ddim',
  plms = 'plms',
}

function HDSettingBlock() {
  const [hdSettings, setHDSettings] = useRecoilState(hdSettingsState)
  if (!hdSettings?.enabled) {
    return <></>
  }

  const onStrategyChange = (value: HDStrategy) => {
    setHDSettings({ hdStrategy: value })
  }

  const onResizeLimitChange = (value: string) => {
    const val = value.length === 0 ? 0 : parseInt(value, 10)
    setHDSettings({ hdStrategyResizeLimit: val })
  }

  const onCropTriggerSizeChange = (value: string) => {
    const val = value.length === 0 ? 0 : parseInt(value, 10)
    setHDSettings({ hdStrategyCropTrigerSize: val })
  }

  const onCropMarginChange = (value: string) => {
    const val = value.length === 0 ? 0 : parseInt(value, 10)
    setHDSettings({ hdStrategyCropMargin: val })
  }

  const renderOriginalOptionDesc = () => {
    return (
      <div>
        Use original picture, suitable for picture size below 2K. Try{' '}
        <div
          tabIndex={0}
          role="button"
          className="inline-tip"
          onClick={() => onStrategyChange(HDStrategy.RESIZE)}
        >
          Resize
        </div>
        {' or '}
        <div
          tabIndex={0}
          role="button"
          className="inline-tip"
          onClick={() => onStrategyChange(HDStrategy.CROP)}
        >
          Crop
        </div>{' '}
        if you didn&apos;t get good results or have GPU memory issue.
      </div>
    )
  }

  const renderResizeOptionDesc = () => {
    return (
      <>
        <div>
          Resize the longer side of the image to a specific size, then do
          inpainting on the resized image.
        </div>
        <NumberInputSetting
          title="Size limit"
          value={`${hdSettings.hdStrategyResizeLimit}`}
          suffix="pixel"
          onValue={onResizeLimitChange}
        />
      </>
    )
  }

  const renderCropOptionDesc = () => {
    return (
      <>
        <div>Crop masking area from the original image to do inpainting.</div>
        <NumberInputSetting
          title="Trigger size"
          value={`${hdSettings.hdStrategyCropTrigerSize}`}
          suffix="pixel"
          onValue={onCropTriggerSizeChange}
        />
        <NumberInputSetting
          title="Crop margin"
          value={`${hdSettings.hdStrategyCropMargin}`}
          suffix="pixel"
          onValue={onCropMarginChange}
        />
      </>
    )
  }

  const renderHDStrategyOptionDesc = (): ReactNode => {
    switch (hdSettings.hdStrategy) {
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
      title="Strategy"
      input={
        <Selector
          width={80}
          value={hdSettings.hdStrategy as string}
          options={Object.values(HDStrategy)}
          onChange={val => onStrategyChange(val as HDStrategy)}
        />
      }
      optionDesc={renderHDStrategyOptionDesc()}
    />
  )
}

export default HDSettingBlock

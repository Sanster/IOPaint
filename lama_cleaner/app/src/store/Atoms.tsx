import { atom } from 'recoil'
import { HDStrategy } from '../components/Setting/HDSettingBlock'
import { AIModel } from '../components/Setting/ModelSettingBlock'

export const fileState = atom<File | undefined>({
  key: 'fileState',
  default: undefined,
})

export const shortcutsState = atom<boolean>({
  key: 'shortcutsState',
  default: false,
})

export interface Setting {
  show: boolean
  saveImageBesideOrigin: boolean
  model: AIModel
  hdStrategy: HDStrategy
  hdStrategyResizeLimit: number
  hdStrategyCropTrigerSize: number
  hdStrategyCropMargin: number
}

export const settingState = atom<Setting>({
  key: 'settingsState',
  default: {
    show: false,
    saveImageBesideOrigin: false,
    model: AIModel.LAMA,
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 2048,
    hdStrategyCropTrigerSize: 2048,
    hdStrategyCropMargin: 128,
  },
})

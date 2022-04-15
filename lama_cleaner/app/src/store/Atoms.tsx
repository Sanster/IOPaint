import { atom } from 'recoil'
import { HDStrategy } from '../components/Settings/HDSettingBlock'
import { AIModel } from '../components/Settings/ModelSettingBlock'

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

  // For LaMa
  hdStrategy: HDStrategy
  hdStrategyResizeLimit: number
  hdStrategyCropTrigerSize: number
  hdStrategyCropMargin: number

  // For LDM
  ldmSteps: number
}

export const settingState = atom<Setting>({
  key: 'settingsState',
  default: {
    show: false,
    saveImageBesideOrigin: false,
    model: AIModel.LAMA,
    hdStrategy: HDStrategy.RESIZE,
    hdStrategyResizeLimit: 2048,
    hdStrategyCropTrigerSize: 2048,
    hdStrategyCropMargin: 128,
    ldmSteps: 50,
  },
})

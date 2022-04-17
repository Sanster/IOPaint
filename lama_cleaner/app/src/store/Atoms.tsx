import { atom } from 'recoil'
import { HDStrategy } from '../components/Settings/HDSettingBlock'
import { AIModel } from '../components/Settings/ModelSettingBlock'
import { ToastState } from '../components/shared/Toast'

export const fileState = atom<File | undefined>({
  key: 'fileState',
  default: undefined,
})

interface ToastAtomState {
  open: boolean
  desc: string
  state: ToastState
  duration: number
}

export const toastState = atom<ToastAtomState>({
  key: 'toastState',
  default: {
    open: false,
    desc: '',
    state: 'default',
    duration: 3000,
  },
})

export const shortcutsState = atom<boolean>({
  key: 'shortcutsState',
  default: false,
})

export interface Settings {
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

export const settingStateDefault = {
  show: false,
  saveImageBesideOrigin: false,
  model: AIModel.LAMA,
  ldmSteps: 50,
  hdStrategy: HDStrategy.RESIZE,
  hdStrategyResizeLimit: 2048,
  hdStrategyCropTrigerSize: 2048,
  hdStrategyCropMargin: 128,
}

export const settingState = atom<Settings>({
  key: 'settingsState',
  default: settingStateDefault,
})

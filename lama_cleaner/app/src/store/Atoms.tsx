import { atom } from 'recoil'
import { HDStrategy } from '../components/Setting/HDSettingBlock'

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
  hdStrategy: HDStrategy
  hdStrategyResizeLimit: string
  hdStrategyCropTrigerSize: string
}

export const settingState = atom<Setting>({
  key: 'settingsState',
  default: {
    show: false,
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: '2048',
    hdStrategyCropTrigerSize: '2048',
  },
})

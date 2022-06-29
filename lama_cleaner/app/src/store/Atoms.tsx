import { atom } from 'recoil'
import { HDStrategy, LDMSampler } from '../components/Settings/HDSettingBlock'
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
  graduallyInpainting: boolean
  runInpaintingManually: boolean
  model: AIModel

  // For LaMa
  hdStrategy: HDStrategy
  hdStrategyResizeLimit: number
  hdStrategyCropTrigerSize: number
  hdStrategyCropMargin: number

  // For LDM
  ldmSteps: number
  ldmSampler: LDMSampler
}

export const settingStateDefault = {
  show: false,
  graduallyInpainting: true,
  runInpaintingManually: false,
  model: AIModel.LAMA,
  ldmSteps: 50,
  ldmSampler: LDMSampler.plms,
  hdStrategy: HDStrategy.RESIZE,
  hdStrategyResizeLimit: 2048,
  hdStrategyCropTrigerSize: 2048,
  hdStrategyCropMargin: 128,
}

const localStorageEffect =
  (key: string) =>
  ({ setSelf, onSet }: any) => {
    const savedValue = localStorage.getItem(key)
    if (savedValue != null) {
      const storageSettings = JSON.parse(savedValue)
      storageSettings.show = false
      setSelf({ ...settingStateDefault, ...storageSettings })
    }

    onSet((newValue: Settings, _: string, isReset: boolean) =>
      isReset
        ? localStorage.removeItem(key)
        : localStorage.setItem(key, JSON.stringify(newValue))
    )
  }

// Each atom can reference an array of these atom effect functions which are called in priority order when the atom is initialized
// https://recoiljs.org/docs/guides/atom-effects/#local-storage-persistence
export const settingState = atom<Settings>({
  key: 'settingsState',
  default: settingStateDefault,
  effects: [localStorageEffect('settingsState')],
})

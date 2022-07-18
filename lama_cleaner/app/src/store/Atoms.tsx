import { atom, selector } from 'recoil'
import _ from 'lodash'
import { HDStrategy, LDMSampler } from '../components/Settings/HDSettingBlock'
import { ToastState } from '../components/shared/Toast'

export enum AIModel {
  LAMA = 'lama',
  LDM = 'ldm',
  ZITS = 'zits',
}

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

export interface HDSettings {
  hdStrategy: HDStrategy
  hdStrategyResizeLimit: number
  hdStrategyCropTrigerSize: number
  hdStrategyCropMargin: number
}

type ModelsHDSettings = { [key in AIModel]: HDSettings }

export interface Settings {
  show: boolean
  downloadMask: boolean
  graduallyInpainting: boolean
  runInpaintingManually: boolean
  model: AIModel
  hdSettings: ModelsHDSettings

  // For LDM
  ldmSteps: number
  ldmSampler: LDMSampler

  // For ZITS
  zitsWireframe: boolean
}

const defaultHDSettings: ModelsHDSettings = {
  [AIModel.LAMA]: {
    hdStrategy: HDStrategy.RESIZE,
    hdStrategyResizeLimit: 2048,
    hdStrategyCropTrigerSize: 2048,
    hdStrategyCropMargin: 128,
  },
  [AIModel.LDM]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 1080,
    hdStrategyCropTrigerSize: 1080,
    hdStrategyCropMargin: 128,
  },
  [AIModel.ZITS]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 1024,
    hdStrategyCropTrigerSize: 1024,
    hdStrategyCropMargin: 128,
  },
}

export const settingStateDefault: Settings = {
  show: false,
  downloadMask: false,
  graduallyInpainting: true,
  runInpaintingManually: false,
  model: AIModel.LAMA,
  hdSettings: defaultHDSettings,

  ldmSteps: 25,
  ldmSampler: LDMSampler.plms,

  zitsWireframe: true,
}

const localStorageEffect =
  (key: string) =>
  ({ setSelf, onSet }: any) => {
    const savedValue = localStorage.getItem(key)
    if (savedValue != null) {
      const storageSettings = JSON.parse(savedValue)
      storageSettings.show = false

      const restored = _.merge(
        _.cloneDeep(settingStateDefault),
        storageSettings
      )
      setSelf(restored)
    }

    onSet((newValue: Settings, val: string, isReset: boolean) =>
      isReset
        ? localStorage.removeItem(key)
        : localStorage.setItem(key, JSON.stringify(newValue))
    )
  }

const ROOT_STATE_KEY = 'settingsState2'
// Each atom can reference an array of these atom effect functions which are called in priority order when the atom is initialized
// https://recoiljs.org/docs/guides/atom-effects/#local-storage-persistence
export const settingState = atom<Settings>({
  key: ROOT_STATE_KEY,
  default: settingStateDefault,
  effects: [localStorageEffect(ROOT_STATE_KEY)],
})

export const hdSettingsState = selector({
  key: 'hdSettings',
  get: ({ get }) => {
    const settings = get(settingState)
    return settings.hdSettings[settings.model]
  },
  set: ({ get, set }, newValue: any) => {
    const settings = get(settingState)
    const hdSettings = settings.hdSettings[settings.model]
    const newHDSettings = { ...hdSettings, ...newValue }

    set(settingState, {
      ...settings,
      hdSettings: { ...settings.hdSettings, [settings.model]: newHDSettings },
    })
  },
})

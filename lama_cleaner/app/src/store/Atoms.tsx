import { atom, selector } from 'recoil'
import _ from 'lodash'
import { HDStrategy, LDMSampler } from '../components/Settings/HDSettingBlock'
import { ToastState } from '../components/shared/Toast'

export enum AIModel {
  LAMA = 'lama',
  LDM = 'ldm',
  ZITS = 'zits',
  MAT = 'mat',
  FCF = 'fcf',
  SD15 = 'sd1.5',
  CV2 = 'cv2',
}

export const fileState = atom<File | undefined>({
  key: 'fileState',
  default: undefined,
})

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

interface AppState {
  disableShortCuts: boolean
  isInpainting: boolean
}

export const appState = atom<AppState>({
  key: 'appState',
  default: {
    disableShortCuts: false,
    isInpainting: false,
  },
})

export const propmtState = atom<string>({
  key: 'promptState',
  default: '',
})

export const isInpaintingState = selector({
  key: 'isInpainting',
  get: ({ get }) => {
    const app = get(appState)
    return app.isInpainting
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isInpainting: newValue })
  },
})

export const croperState = atom<Rect>({
  key: 'croperState',
  default: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
})

export const croperX = selector({
  key: 'croperX',
  get: ({ get }) => get(croperState).x,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, x: newValue })
  },
})

export const croperY = selector({
  key: 'croperY',
  get: ({ get }) => get(croperState).y,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, y: newValue })
  },
})

export const croperHeight = selector({
  key: 'croperHeight',
  get: ({ get }) => get(croperState).height,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, height: newValue })
  },
})

export const croperWidth = selector({
  key: 'croperWidth',
  get: ({ get }) => get(croperState).width,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, width: newValue })
  },
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
  enabled: boolean
}

type ModelsHDSettings = { [key in AIModel]: HDSettings }

export enum CV2Flag {
  INPAINT_NS = 'INPAINT_NS',
  INPAINT_TELEA = 'INPAINT_TELEA',
}

export interface Settings {
  show: boolean
  showCroper: boolean
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

  // For SD
  sdMaskBlur: number
  sdMode: SDMode
  sdStrength: number
  sdSteps: number
  sdGuidanceScale: number
  sdSampler: SDSampler
  sdSeed: number
  sdSeedFixed: boolean // true: use sdSeed, false: random generate seed on backend
  sdNumSamples: number

  // For OpenCV2
  cv2Radius: number
  cv2Flag: CV2Flag
}

const defaultHDSettings: ModelsHDSettings = {
  [AIModel.LAMA]: {
    hdStrategy: HDStrategy.RESIZE,
    hdStrategyResizeLimit: 2048,
    hdStrategyCropTrigerSize: 2048,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
  [AIModel.LDM]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 1080,
    hdStrategyCropTrigerSize: 1080,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
  [AIModel.ZITS]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 1024,
    hdStrategyCropTrigerSize: 1024,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
  [AIModel.MAT]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 1024,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
  [AIModel.FCF]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 512,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
  [AIModel.SD15]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
  [AIModel.CV2]: {
    hdStrategy: HDStrategy.RESIZE,
    hdStrategyResizeLimit: 1080,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
}

export enum SDSampler {
  ddim = 'ddim',
  pndm = 'pndm',
  klms = 'k_lms',
}

export enum SDMode {
  text2img = 'text2img',
  img2img = 'img2img',
  inpainting = 'inpainting',
}

export const settingStateDefault: Settings = {
  show: false,
  showCroper: false,
  downloadMask: false,
  graduallyInpainting: true,
  runInpaintingManually: false,
  model: AIModel.LAMA,
  hdSettings: defaultHDSettings,

  ldmSteps: 25,
  ldmSampler: LDMSampler.plms,

  zitsWireframe: true,

  // SD
  sdMaskBlur: 5,
  sdMode: SDMode.inpainting,
  sdStrength: 0.75,
  sdSteps: 50,
  sdGuidanceScale: 7.5,
  sdSampler: SDSampler.pndm,
  sdSeed: 42,
  sdSeedFixed: true,
  sdNumSamples: 1,

  // CV2
  cv2Radius: 5,
  cv2Flag: CV2Flag.INPAINT_NS,
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

const ROOT_STATE_KEY = 'settingsState4'
// Each atom can reference an array of these atom effect functions which are called in priority order when the atom is initialized
// https://recoiljs.org/docs/guides/atom-effects/#local-storage-persistence
export const settingState = atom<Settings>({
  key: ROOT_STATE_KEY,
  default: settingStateDefault,
  effects: [localStorageEffect(ROOT_STATE_KEY)],
})

export const seedState = selector({
  key: 'seed',
  get: ({ get }) => {
    const settings = get(settingState)
    return settings.sdSeed
  },
  set: ({ get, set }, newValue: any) => {
    const settings = get(settingState)
    set(settingState, { ...settings, sdSeed: newValue })
  },
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

export const isSDState = selector({
  key: 'isSD',
  get: ({ get }) => {
    const settings = get(settingState)
    return settings.model === AIModel.SD15
  },
})

export const runManuallyState = selector({
  key: 'runManuallyState',
  get: ({ get }) => {
    const settings = get(settingState)
    const isSD = get(isSDState)
    return settings.runInpaintingManually || isSD
  },
})

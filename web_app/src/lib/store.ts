import { atom, selector } from "recoil"
import _ from "lodash"
import { CV2Flag, HDStrategy, LDMSampler, ModelsHDSettings } from "./types"

export enum AIModel {
  LAMA = "lama",
  LDM = "ldm",
  ZITS = "zits",
  MAT = "mat",
  FCF = "fcf",
  SD15 = "sd1.5",
  ANYTHING4 = "anything4",
  REALISTIC_VISION_1_4 = "realisticVision1.4",
  SD2 = "sd2",
  CV2 = "cv2",
  Mange = "manga",
  PAINT_BY_EXAMPLE = "paint_by_example",
  PIX2PIX = "instruct_pix2pix",
  KANDINSKY22 = "kandinsky2.2",
}

export enum ControlNetMethod {
  canny = "canny",
  inpaint = "inpaint",
  openpose = "openpose",
  depth = "depth",
}

export const ControlNetMethodMap: any = {
  canny: "control_v11p_sd15_canny",
  inpaint: "control_v11p_sd15_inpaint",
  openpose: "control_v11p_sd15_openpose",
  depth: "control_v11f1p_sd15_depth",
}

export const ControlNetMethodMap2: any = {
  control_v11p_sd15_canny: "canny",
  control_v11p_sd15_inpaint: "inpaint",
  control_v11p_sd15_openpose: "openpose",
  control_v11f1p_sd15_depth: "depth",
}

export const maskState = atom<File | undefined>({
  key: "maskState",
  default: undefined,
})

export const paintByExampleImageState = atom<File | undefined>({
  key: "paintByExampleImageState",
  default: undefined,
})

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

interface AppState {
  isDisableModelSwitch: boolean
  isEnableAutoSaving: boolean
  isInteractiveSeg: boolean
  isInteractiveSegRunning: boolean
  interactiveSegClicks: number[][]
  enableFileManager: boolean
  isControlNet: boolean
  controlNetMethod: string
  plugins: string[]
  isPluginRunning: boolean
}

export const appState = atom<AppState>({
  key: "appState",
  default: {
    isDisableModelSwitch: false,
    isEnableAutoSaving: false,
    isInteractiveSeg: false,
    isInteractiveSegRunning: false,
    interactiveSegClicks: [],
    enableFileManager: false,
    isControlNet: false,
    controlNetMethod: ControlNetMethod.canny,
    plugins: [],
    isPluginRunning: false,
  },
})

export const negativePropmtState = atom<string>({
  key: "negativePromptState",
  default: "",
})

export const isPluginRunningState = selector({
  key: "isPluginRunningState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isPluginRunning
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isPluginRunning: newValue })
  },
})

export const serverConfigState = selector({
  key: "serverConfigState",
  get: ({ get }) => {
    const app = get(appState)
    return {
      isControlNet: app.isControlNet,
      controlNetMethod: app.controlNetMethod,
      isDisableModelSwitchState: app.isDisableModelSwitch,
      isEnableAutoSaving: app.isEnableAutoSaving,
      enableFileManager: app.enableFileManager,
      plugins: app.plugins,
    }
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    const methodShortName = ControlNetMethodMap2[newValue.controlNetMethod]
    set(appState, { ...app, ...newValue, controlnetMethod: methodShortName })

    const setting = get(settingState)
    set(settingState, {
      ...setting,
      controlnetMethod: methodShortName,
    })
  },
})

export const enableFileManagerState = selector({
  key: "enableFileManagerState",
  get: ({ get }) => {
    const app = get(appState)
    return app.enableFileManager
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, enableFileManager: newValue })
  },
})

export const isInteractiveSegState = selector({
  key: "isInteractiveSegState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isInteractiveSeg
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isInteractiveSeg: newValue })
  },
})

export const isInteractiveSegRunningState = selector({
  key: "isInteractiveSegRunningState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isInteractiveSegRunning
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isInteractiveSegRunning: newValue })
  },
})

export const isProcessingState = selector({
  key: "isProcessingState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isInteractiveSegRunning || app.isPluginRunning
  },
})

export const interactiveSegClicksState = selector({
  key: "interactiveSegClicksState",
  get: ({ get }) => {
    const app = get(appState)
    return app.interactiveSegClicks
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, interactiveSegClicks: newValue })
  },
})

export const isDisableModelSwitchState = selector({
  key: "isDisableModelSwitchState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isDisableModelSwitch
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isDisableModelSwitch: newValue })
  },
})

export const isControlNetState = selector({
  key: "isControlNetState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isControlNet
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isControlNet: newValue })
  },
})

export const isEnableAutoSavingState = selector({
  key: "isEnableAutoSavingState",
  get: ({ get }) => {
    const app = get(appState)
    return app.isEnableAutoSaving
  },
  set: ({ get, set }, newValue: any) => {
    const app = get(appState)
    set(appState, { ...app, isEnableAutoSaving: newValue })
  },
})

export const croperState = atom<Rect>({
  key: "croperState",
  default: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
})

export const croperX = selector({
  key: "croperX",
  get: ({ get }) => get(croperState).x,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, x: newValue })
  },
})

export const croperY = selector({
  key: "croperY",
  get: ({ get }) => get(croperState).y,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, y: newValue })
  },
})

export const croperHeight = selector({
  key: "croperHeight",
  get: ({ get }) => get(croperState).height,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, height: newValue })
  },
})

export const croperWidth = selector({
  key: "croperWidth",
  get: ({ get }) => get(croperState).width,
  set: ({ get, set }, newValue: any) => {
    const rect = get(croperState)
    set(croperState, { ...rect, width: newValue })
  },
})

export const extenderState = atom<Rect>({
  key: "extenderState",
  default: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
})

export const extenderX = selector({
  key: "extenderX",
  get: ({ get }) => get(extenderState).x,
  set: ({ get, set }, newValue: any) => {
    const rect = get(extenderState)
    set(extenderState, { ...rect, x: newValue })
  },
})

export const extenderY = selector({
  key: "extenderY",
  get: ({ get }) => get(extenderState).y,
  set: ({ get, set }, newValue: any) => {
    const rect = get(extenderState)
    set(extenderState, { ...rect, y: newValue })
  },
})

export const extenderHeight = selector({
  key: "extenderHeight",
  get: ({ get }) => get(extenderState).height,
  set: ({ get, set }, newValue: any) => {
    const rect = get(extenderState)
    set(extenderState, { ...rect, height: newValue })
  },
})

export const extenderWidth = selector({
  key: "extenderWidth",
  get: ({ get }) => get(extenderState).width,
  set: ({ get, set }, newValue: any) => {
    const rect = get(extenderState)
    set(extenderState, { ...rect, width: newValue })
  },
})

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
  sdStrength: number
  sdSteps: number
  sdGuidanceScale: number
  sdSampler: SDSampler
  sdSeed: number
  sdSeedFixed: boolean // true: use sdSeed, false: random generate seed on backend
  sdNumSamples: number
  sdMatchHistograms: boolean
  sdScale: number

  // For OpenCV2
  cv2Radius: number
  cv2Flag: CV2Flag

  // Paint by Example
  paintByExampleSteps: number
  paintByExampleGuidanceScale: number
  paintByExampleSeed: number
  paintByExampleSeedFixed: boolean
  paintByExampleMaskBlur: number
  paintByExampleMatchHistograms: boolean

  // InstructPix2Pix
  p2pSteps: number
  p2pImageGuidanceScale: number
  p2pGuidanceScale: number

  // ControlNet
  controlnetConditioningScale: number
  controlnetMethod: string
}

const defaultHDSettings: ModelsHDSettings = {
  [AIModel.LAMA]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 2048,
    hdStrategyCropTrigerSize: 800,
    hdStrategyCropMargin: 196,
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
    enabled: false,
  },
  [AIModel.ANYTHING4]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
  [AIModel.REALISTIC_VISION_1_4]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
  [AIModel.SD2]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
  [AIModel.PAINT_BY_EXAMPLE]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
  [AIModel.PIX2PIX]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
  [AIModel.Mange]: {
    hdStrategy: HDStrategy.CROP,
    hdStrategyResizeLimit: 1280,
    hdStrategyCropTrigerSize: 1024,
    hdStrategyCropMargin: 196,
    enabled: true,
  },
  [AIModel.CV2]: {
    hdStrategy: HDStrategy.RESIZE,
    hdStrategyResizeLimit: 1080,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: true,
  },
  [AIModel.KANDINSKY22]: {
    hdStrategy: HDStrategy.ORIGINAL,
    hdStrategyResizeLimit: 768,
    hdStrategyCropTrigerSize: 512,
    hdStrategyCropMargin: 128,
    enabled: false,
  },
}

export enum SDSampler {
  ddim = "ddim",
  pndm = "pndm",
  klms = "k_lms",
  kEuler = "k_euler",
  kEulerA = "k_euler_a",
  dpmPlusPlus = "dpm++",
  uni_pc = "uni_pc",
}

export enum SDMode {
  text2img = "text2img",
  img2img = "img2img",
  inpainting = "inpainting",
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
  sdStrength: 0.75,
  sdSteps: 50,
  sdGuidanceScale: 7.5,
  sdSampler: SDSampler.uni_pc,
  sdSeed: 42,
  sdSeedFixed: false,
  sdNumSamples: 1,
  sdMatchHistograms: false,
  sdScale: 100,

  // CV2
  cv2Radius: 5,
  cv2Flag: CV2Flag.INPAINT_NS,

  // Paint by Example
  paintByExampleSteps: 50,
  paintByExampleGuidanceScale: 7.5,
  paintByExampleSeed: 42,
  paintByExampleMaskBlur: 5,
  paintByExampleSeedFixed: false,
  paintByExampleMatchHistograms: false,

  // InstructPix2Pix
  p2pSteps: 50,
  p2pImageGuidanceScale: 1.5,
  p2pGuidanceScale: 7.5,

  // ControlNet
  controlnetConditioningScale: 0.4,
  controlnetMethod: ControlNetMethod.canny,
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

const ROOT_STATE_KEY = "settingsState4"
// Each atom can reference an array of these atom effect functions which are called in priority order when the atom is initialized
// https://recoiljs.org/docs/guides/atom-effects/#local-storage-persistence
export const settingState = atom<Settings>({
  key: ROOT_STATE_KEY,
  default: settingStateDefault,
  effects: [localStorageEffect(ROOT_STATE_KEY)],
})

export const seedState = selector({
  key: "seed",
  get: ({ get }) => {
    const settings = get(settingState)
    switch (settings.model) {
      case AIModel.PAINT_BY_EXAMPLE:
        return settings.paintByExampleSeedFixed
          ? settings.paintByExampleSeed
          : -1
      default:
        return settings.sdSeedFixed ? settings.sdSeed : -1
    }
  },
  set: ({ get, set }, newValue: any) => {
    const settings = get(settingState)
    switch (settings.model) {
      case AIModel.PAINT_BY_EXAMPLE:
        if (!settings.paintByExampleSeedFixed) {
          set(settingState, { ...settings, paintByExampleSeed: newValue })
        }
        break
      default:
        if (!settings.sdSeedFixed) {
          set(settingState, { ...settings, sdSeed: newValue })
        }
    }
  },
})

export const hdSettingsState = selector({
  key: "hdSettings",
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
  key: "isSD",
  get: ({ get }) => {
    const settings = get(settingState)
    return (
      settings.model === AIModel.SD15 ||
      settings.model === AIModel.SD2 ||
      settings.model === AIModel.ANYTHING4 ||
      settings.model === AIModel.REALISTIC_VISION_1_4 ||
      settings.model === AIModel.KANDINSKY22
    )
  },
})

export const isPaintByExampleState = selector({
  key: "isPaintByExampleState",
  get: ({ get }) => {
    const settings = get(settingState)
    return settings.model === AIModel.PAINT_BY_EXAMPLE
  },
})

export const isPix2PixState = selector({
  key: "isPix2PixState",
  get: ({ get }) => {
    const settings = get(settingState)
    return settings.model === AIModel.PIX2PIX
  },
})

export const runManuallyState = selector({
  key: "runManuallyState",
  get: ({ get }) => {
    const settings = get(settingState)
    const isSD = get(isSDState)
    const isPaintByExample = get(isPaintByExampleState)
    const isPix2Pix = get(isPix2PixState)
    return (
      settings.runInpaintingManually || isSD || isPaintByExample || isPix2Pix
    )
  },
})

export const isDiffusionModelsState = selector({
  key: "isDiffusionModelsState",
  get: ({ get }) => {
    const isSD = get(isSDState)
    const isPaintByExample = get(isPaintByExampleState)
    const isPix2Pix = get(isPix2PixState)
    return isSD || isPaintByExample || isPix2Pix
  },
})

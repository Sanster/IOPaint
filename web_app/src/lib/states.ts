import { create } from "zustand"
import { persist } from "zustand/middleware"
import { immer } from "zustand/middleware/immer"
import {
  CV2Flag,
  FreeuConfig,
  LDMSampler,
  ModelInfo,
  SDSampler,
  SortBy,
  SortOrder,
} from "./types"
import { DEFAULT_BRUSH_SIZE } from "./const"

type FileManagerState = {
  sortBy: SortBy
  sortOrder: SortOrder
  layout: "rows" | "masonry"
  searchText: string
  inputDirectory: string
  outputDirectory: string
}

type CropperState = {
  x: number
  y: number
  width: number
  height: number
}

export type Settings = {
  model: ModelInfo
  enableDownloadMask: boolean
  enableManualInpainting: boolean
  enableUploadMask: boolean
  showCroper: boolean

  // For LDM
  ldmSteps: number
  ldmSampler: LDMSampler

  // For ZITS
  zitsWireframe: boolean

  // For OpenCV2
  cv2Radius: number
  cv2Flag: CV2Flag

  // For Diffusion moel
  prompt: string
  negativePrompt: string
  seed: number
  seedFixed: boolean

  // For SD
  sdMaskBlur: number
  sdStrength: number
  sdSteps: number
  sdGuidanceScale: number
  sdSampler: SDSampler
  sdMatchHistograms: boolean
  sdScale: number

  // Paint by Example
  paintByExampleSteps: number
  paintByExampleGuidanceScale: number
  paintByExampleMaskBlur: number
  paintByExampleMatchHistograms: boolean

  // InstructPix2Pix
  p2pSteps: number
  p2pImageGuidanceScale: number
  p2pGuidanceScale: number

  // ControlNet
  controlnetConditioningScale: number
  controlnetMethod: string

  enableLCMLora: boolean
  enableFreeu: boolean
  freeuConfig: FreeuConfig
}

type ServerConfig = {
  plugins: string[]
  enableFileManager: boolean
  enableAutoSaving: boolean
}

type InteractiveSegState = {
  isInteractiveSeg: boolean
  isInteractiveSegRunning: boolean
  clicks: number[][]
}

type AppState = {
  file: File | null
  customMask: File | null
  imageHeight: number
  imageWidth: number
  brushSize: number
  brushSizeScale: number
  isInpainting: boolean
  isPluginRunning: boolean

  interactiveSegState: InteractiveSegState
  fileManagerState: FileManagerState
  cropperState: CropperState
  serverConfig: ServerConfig

  settings: Settings
}

type AppAction = {
  setFile: (file: File) => void
  setCustomFile: (file: File) => void
  setIsInpainting: (newValue: boolean) => void
  setIsPluginRunning: (newValue: boolean) => void
  setBrushSize: (newValue: number) => void
  setImageSize: (width: number, height: number) => void

  setCropperX: (newValue: number) => void
  setCropperY: (newValue: number) => void
  setCropperWidth: (newValue: number) => void
  setCropperHeight: (newValue: number) => void

  setServerConfig: (newValue: ServerConfig) => void
  setSeed: (newValue: number) => void
  updateSettings: (newSettings: Partial<Settings>) => void
  updateFileManagerState: (newState: Partial<FileManagerState>) => void
  updateInteractiveSegState: (newState: Partial<InteractiveSegState>) => void
  resetInteractiveSegState: () => void
  showPromptInput: () => boolean
  showSidePanel: () => boolean
}

const defaultValues: AppState = {
  file: null,
  customMask: null,
  imageHeight: 0,
  imageWidth: 0,
  brushSize: DEFAULT_BRUSH_SIZE,
  brushSizeScale: 1,
  isInpainting: false,
  isPluginRunning: false,

  interactiveSegState: {
    isInteractiveSeg: false,
    isInteractiveSegRunning: false,
    clicks: [],
  },

  cropperState: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
  fileManagerState: {
    sortBy: SortBy.CTIME,
    sortOrder: SortOrder.DESCENDING,
    layout: "masonry",
    searchText: "",
    inputDirectory: "",
    outputDirectory: "",
  },
  serverConfig: {
    plugins: [],
    enableFileManager: false,
    enableAutoSaving: false,
  },
  settings: {
    model: {
      name: "lama",
      path: "lama",
      model_type: "inpaint",
      support_controlnet: false,
      controlnets: [],
      support_freeu: false,
      support_lcm_lora: false,
      is_single_file_diffusers: false,
    },
    showCroper: false,
    enableDownloadMask: false,
    enableManualInpainting: false,
    enableUploadMask: false,
    ldmSteps: 30,
    ldmSampler: LDMSampler.ddim,
    zitsWireframe: true,
    cv2Radius: 5,
    cv2Flag: CV2Flag.INPAINT_NS,
    prompt: "",
    negativePrompt: "",
    seed: 42,
    seedFixed: false,
    sdMaskBlur: 5,
    sdStrength: 1.0,
    sdSteps: 50,
    sdGuidanceScale: 7.5,
    sdSampler: SDSampler.uni_pc,
    sdMatchHistograms: false,
    sdScale: 100,
    paintByExampleSteps: 50,
    paintByExampleGuidanceScale: 7.5,
    paintByExampleMaskBlur: 5,
    paintByExampleMatchHistograms: false,
    p2pSteps: 50,
    p2pImageGuidanceScale: 1.5,
    p2pGuidanceScale: 7.5,
    controlnetConditioningScale: 0.4,
    controlnetMethod: "lllyasviel/control_v11p_sd15_canny",
    enableLCMLora: false,
    enableFreeu: false,
    freeuConfig: { s1: 0.9, s2: 0.2, b1: 1.2, b2: 1.4 },
  },
}

export const useStore = create<AppState & AppAction>()(
  immer(
    persist(
      (set, get) => ({
        ...defaultValues,

        showPromptInput: (): boolean => {
          const model_type = get().settings.model.model_type
          return ["diffusers_sd", "diffusers_sd_inpaint"].includes(model_type)
        },

        showSidePanel: (): boolean => {
          const model_type = get().settings.model.model_type
          return ["diffusers_sd", "diffusers_sd_inpaint"].includes(model_type)
        },

        setServerConfig: (newValue: ServerConfig) => {
          set((state: AppState) => {
            state.serverConfig = newValue
          })
        },

        updateSettings: (newSettings: Partial<Settings>) => {
          set((state: AppState) => {
            state.settings = {
              ...state.settings,
              ...newSettings,
            }
          })
        },

        updateFileManagerState: (newState: Partial<FileManagerState>) => {
          set((state: AppState) => {
            state.fileManagerState = {
              ...state.fileManagerState,
              ...newState,
            }
          })
        },

        updateInteractiveSegState: (newState: Partial<InteractiveSegState>) => {
          set((state: AppState) => {
            state.interactiveSegState = {
              ...state.interactiveSegState,
              ...newState,
            }
          })
        },
        resetInteractiveSegState: () => {
          set((state: AppState) => {
            state.interactiveSegState = defaultValues.interactiveSegState
          })
        },

        setIsInpainting: (newValue: boolean) =>
          set((state: AppState) => {
            state.isInpainting = newValue
          }),

        setIsPluginRunning: (newValue: boolean) =>
          set((state: AppState) => {
            state.isPluginRunning = newValue
          }),

        setFile: (file: File) =>
          set((state: AppState) => {
            // TODO: 清空各种状态
            state.file = file
          }),

        setCustomFile: (file: File) =>
          set((state: AppState) => {
            state.customMask = file
          }),

        setBrushSize: (newValue: number) =>
          set((state: AppState) => {
            state.brushSize = newValue
          }),

        setImageSize: (width: number, height: number) => {
          // 根据图片尺寸调整 brushSize 的 scale
          set((state: AppState) => {
            state.imageWidth = width
            state.imageHeight = height
            state.brushSizeScale = Math.max(Math.min(width, height), 512) / 512
          })
        },

        setCropperX: (newValue: number) =>
          set((state: AppState) => {
            state.cropperState.x = newValue
          }),

        setCropperY: (newValue: number) =>
          set((state: AppState) => {
            state.cropperState.y = newValue
          }),

        setCropperWidth: (newValue: number) =>
          set((state: AppState) => {
            state.cropperState.width = newValue
          }),

        setCropperHeight: (newValue: number) =>
          set((state: AppState) => {
            state.cropperState.height = newValue
          }),

        setSeed: (newValue: number) =>
          set((state: AppState) => {
            state.settings.seed = newValue
          }),
      }),
      {
        name: "ZUSTAND_STATE", // name of the item in the storage (must be unique)
        version: 0,
        partialize: (state) =>
          Object.fromEntries(
            Object.entries(state).filter(([key]) =>
              ["fileManagerState", "prompt", "settings"].includes(key)
            )
          ),
      }
    )
  )
)

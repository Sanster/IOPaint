import { persist } from "zustand/middleware"
import { shallow } from "zustand/shallow"
import { immer } from "zustand/middleware/immer"
import { castDraft } from "immer"
import { nanoid } from "nanoid"
import { createWithEqualityFn } from "zustand/traditional"
import {
  CV2Flag,
  FreeuConfig,
  LDMSampler,
  Line,
  LineGroup,
  ModelInfo,
  PluginParams,
  Point,
  SDSampler,
  Size,
  SortBy,
  SortOrder,
} from "./types"
import {
  DEFAULT_BRUSH_SIZE,
  INSTRUCT_PIX2PIX,
  MODEL_TYPE_INPAINT,
  MODEL_TYPE_OTHER,
  PAINT_BY_EXAMPLE,
} from "./const"
import { dataURItoBlob, generateMask, loadImage, srcToFile } from "./utils"
import inpaint, { runPlugin } from "./api"
import { toast, useToast } from "@/components/ui/use-toast"

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
  enableControlNet: boolean
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
  interactiveSegMask: HTMLImageElement | null
  tmpInteractiveSegMask: HTMLImageElement | null
  prevInteractiveSegMask: HTMLImageElement | null
  clicks: number[][]
}

type EditorState = {
  baseBrushSize: number
  brushSizeScale: number
  renders: HTMLImageElement[]
  paintByExampleImage: File | null
  lineGroups: LineGroup[]
  lastLineGroup: LineGroup
  curLineGroup: LineGroup
  extraMasks: HTMLImageElement[]
  // redo 相关
  redoRenders: HTMLImageElement[]
  redoCurLines: Line[]
  redoLineGroups: LineGroup[]
}

type AppState = {
  file: File | null
  customMask: File | null
  imageHeight: number
  imageWidth: number
  isInpainting: boolean
  isPluginRunning: boolean
  windowSize: Size
  editorState: EditorState
  disableShortCuts: boolean

  interactiveSegState: InteractiveSegState
  fileManagerState: FileManagerState
  cropperState: CropperState
  serverConfig: ServerConfig

  settings: Settings
}

type AppAction = {
  updateAppState: (newState: Partial<AppState>) => void
  setFile: (file: File) => void
  setCustomFile: (file: File) => void
  setIsInpainting: (newValue: boolean) => void
  setIsPluginRunning: (newValue: boolean) => void
  getIsProcessing: () => boolean
  setBaseBrushSize: (newValue: number) => void
  getBrushSize: () => number
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
  handleInteractiveSegAccept: () => void
  showPromptInput: () => boolean
  showSidePanel: () => boolean

  runInpainting: () => Promise<void>
  runRenderablePlugin: (
    pluginName: string,
    params?: PluginParams
  ) => Promise<void>

  // EditorState
  getCurrentTargetFile: () => Promise<File>
  updateEditorState: (newState: Partial<EditorState>) => void
  runMannually: () => boolean
  handleCanvasMouseDown: (point: Point) => void
  handleCanvasMouseMove: (point: Point) => void
  cleanCurLineGroup: () => void
  resetRedoState: () => void
  undo: () => void
  redo: () => void
  undoDisabled: () => boolean
  redoDisabled: () => boolean
}

const defaultValues: AppState = {
  file: null,
  customMask: null,
  imageHeight: 0,
  imageWidth: 0,
  isInpainting: false,
  isPluginRunning: false,
  disableShortCuts: false,

  windowSize: {
    height: 600,
    width: 800,
  },
  editorState: {
    baseBrushSize: DEFAULT_BRUSH_SIZE,
    brushSizeScale: 1,
    renders: [],
    paintByExampleImage: null,
    extraMasks: [],
    lineGroups: [],
    lastLineGroup: [],
    curLineGroup: [],
    redoRenders: [],
    redoCurLines: [],
    redoLineGroups: [],
  },

  interactiveSegState: {
    isInteractiveSeg: false,
    interactiveSegMask: null,
    tmpInteractiveSegMask: null,
    prevInteractiveSegMask: null,
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
      need_prompt: false,
    },
    enableControlNet: false,
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

export const useStore = createWithEqualityFn<AppState & AppAction>()(
  persist(
    immer((set, get) => ({
      ...defaultValues,

      getCurrentTargetFile: async (): Promise<File> => {
        const file = get().file! // 一定是在 file 加载了以后才可能调用这个函数
        const renders = get().editorState.renders

        let targetFile = file
        if (renders.length > 0) {
          const lastRender = renders[renders.length - 1]
          targetFile = await srcToFile(
            lastRender.currentSrc,
            file.name,
            file.type
          )
        }
        return targetFile
      },

      runInpainting: async () => {
        const {
          isInpainting,
          file,
          imageWidth,
          imageHeight,
          settings,
          cropperState,
        } = get()
        if (isInpainting) {
          return
        }

        if (file === null) {
          return
        }
        const {
          lastLineGroup,
          curLineGroup,
          lineGroups,
          renders,
          paintByExampleImage,
        } = get().editorState

        const { interactiveSegMask, prevInteractiveSegMask } =
          get().interactiveSegState

        const useLastLineGroup =
          curLineGroup.length === 0 && interactiveSegMask === null

        const maskImage = useLastLineGroup
          ? prevInteractiveSegMask
          : interactiveSegMask

        // useLastLineGroup 的影响
        // 1. 使用上一次的 mask
        // 2. 结果替换当前 render
        let maskLineGroup: LineGroup = []
        if (useLastLineGroup === true) {
          if (lastLineGroup.length === 0 && maskImage === null) {
            toast({
              variant: "destructive",
              description: "Please draw mask on picture",
            })
            return
          }
          maskLineGroup = lastLineGroup
        } else {
          if (curLineGroup.length === 0 && maskImage === null) {
            toast({
              variant: "destructive",
              description: "Please draw mask on picture",
            })
            return
          }
          maskLineGroup = curLineGroup
        }

        const newLineGroups = [...lineGroups, maskLineGroup]

        set((state) => {
          state.isInpainting = true
        })

        let targetFile = file
        if (useLastLineGroup === true) {
          // renders.length == 1 还是用原来的
          if (renders.length > 1) {
            const lastRender = renders[renders.length - 2]
            targetFile = await srcToFile(
              lastRender.currentSrc,
              file.name,
              file.type
            )
          }
        } else if (renders.length > 0) {
          const lastRender = renders[renders.length - 1]
          targetFile = await srcToFile(
            lastRender.currentSrc,
            file.name,
            file.type
          )
        }

        const maskCanvas = generateMask(
          imageWidth,
          imageHeight,
          [maskLineGroup],
          maskImage ? [maskImage] : []
        )

        try {
          const res = await inpaint(
            targetFile,
            settings,
            cropperState,
            dataURItoBlob(maskCanvas.toDataURL()),
            paintByExampleImage
          )

          if (!res) {
            throw new Error("Something went wrong on server side.")
          }

          const { blob, seed } = res
          if (seed) {
            set((state) => (state.settings.seed = parseInt(seed, 10)))
          }
          const newRender = new Image()
          await loadImage(newRender, blob)
          if (useLastLineGroup === true) {
            const prevRenders = renders.slice(0, -1)
            const newRenders = [...prevRenders, newRender]
            get().updateEditorState({
              renders: newRenders,
              lineGroups: newLineGroups,
              lastLineGroup: curLineGroup,
              curLineGroup: [],
            })
          } else {
            const newRenders = [...renders, newRender]
            get().updateEditorState({
              renders: newRenders,
              lineGroups: newLineGroups,
              lastLineGroup: curLineGroup,
              curLineGroup: [],
            })
          }
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: e.message ? e.message : e.toString(),
          })
        }

        get().resetRedoState()
        set((state) => {
          state.isInpainting = false
        })

        const newInteractiveSegState = {
          ...defaultValues.interactiveSegState,
          prevInteractiveSegMask: useLastLineGroup ? null : maskImage,
        }

        set((state) => {
          state.interactiveSegState = castDraft(newInteractiveSegState)
        })
      },

      runRenderablePlugin: async (
        pluginName: string,
        params: PluginParams = { upscale: 1 }
      ) => {
        const { renders, lineGroups } = get().editorState
        set((state) => {
          state.isInpainting = true
        })

        try {
          const start = new Date()
          const targetFile = await get().getCurrentTargetFile()
          const res = await runPlugin(pluginName, targetFile, params.upscale)
          if (!res) {
            throw new Error("Something went wrong on server side.")
          }
          const { blob } = res
          const newRender = new Image()
          await loadImage(newRender, blob)
          get().setImageSize(newRender.height, newRender.width)
          const newRenders = [...renders, newRender]
          const newLineGroups = [...lineGroups, []]
          get().updateEditorState({
            renders: newRenders,
            lineGroups: newLineGroups,
          })
          const end = new Date()
          const time = end.getTime() - start.getTime()
          toast({
            description: `Run ${pluginName} successfully in ${time / 1000}s`,
          })
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: e.message ? e.message : e.toString(),
          })
        }
        set((state) => {
          state.isInpainting = false
        })
      },

      // Edirot State //
      updateEditorState: (newState: Partial<EditorState>) => {
        set((state) => {
          state.editorState = castDraft({ ...state.editorState, ...newState })
        })
      },

      cleanCurLineGroup: () => {
        get().updateEditorState({ curLineGroup: [] })
      },

      handleCanvasMouseDown: (point: Point) => {
        let lineGroup: LineGroup = []
        const state = get()
        if (state.runMannually()) {
          lineGroup = [...state.editorState.curLineGroup]
        }
        lineGroup.push({ size: state.getBrushSize(), pts: [point] })
        set((state) => {
          state.editorState.curLineGroup = lineGroup
        })
      },

      handleCanvasMouseMove: (point: Point) => {
        set((state) => {
          const curLineGroup = state.editorState.curLineGroup
          if (curLineGroup.length) {
            curLineGroup[curLineGroup.length - 1].pts.push(point)
          }
        })
      },

      runMannually: (): boolean => {
        const state = get()
        return (
          state.settings.enableManualInpainting ||
          state.settings.model.model_type !== MODEL_TYPE_INPAINT
        )
      },

      getIsProcessing: (): boolean => {
        return get().isInpainting || get().isPluginRunning
      },

      // undo/redo

      undoDisabled: (): boolean => {
        const editorState = get().editorState
        if (editorState.renders.length > 0) {
          return false
        }
        if (get().runMannually()) {
          if (editorState.curLineGroup.length === 0) {
            return true
          }
        } else if (editorState.renders.length === 0) {
          return true
        }
        return false
      },

      undo: () => {
        if (
          get().runMannually() &&
          get().editorState.curLineGroup.length !== 0
        ) {
          // undoStroke
          set((state) => {
            const editorState = state.editorState
            if (editorState.curLineGroup.length === 0) {
              return
            }
            editorState.lastLineGroup = []
            const lastLine = editorState.curLineGroup.pop()!
            editorState.redoCurLines.push(lastLine)
          })
        } else {
          set((state) => {
            const editorState = state.editorState
            if (
              editorState.renders.length === 0 ||
              editorState.lineGroups.length === 0
            ) {
              return
            }
            const lastLineGroup = editorState.lineGroups.pop()!
            editorState.redoLineGroups.push(lastLineGroup)
            editorState.redoCurLines = []
            editorState.curLineGroup = []

            const lastRender = editorState.renders.pop()!
            editorState.redoRenders.push(lastRender)
          })
        }
      },

      redoDisabled: (): boolean => {
        const editorState = get().editorState
        if (editorState.redoRenders.length > 0) {
          return false
        }
        if (get().runMannually()) {
          if (editorState.redoCurLines.length === 0) {
            return true
          }
        } else if (editorState.redoRenders.length === 0) {
          return true
        }
        return false
      },

      redo: () => {
        if (
          get().runMannually() &&
          get().editorState.redoCurLines.length !== 0
        ) {
          set((state) => {
            const editorState = state.editorState
            if (editorState.redoCurLines.length === 0) {
              return
            }
            const line = editorState.redoCurLines.pop()!
            editorState.curLineGroup.push(line)
          })
        } else {
          set((state) => {
            const editorState = state.editorState
            if (
              editorState.redoRenders.length === 0 ||
              editorState.redoLineGroups.length === 0
            ) {
              return
            }
            const lastLineGroup = editorState.redoLineGroups.pop()!
            editorState.lineGroups.push(lastLineGroup)
            editorState.curLineGroup = []

            const lastRender = editorState.redoRenders.pop()!
            editorState.renders.push(lastRender)
          })
        }
      },

      resetRedoState: () => {
        set((state) => {
          state.editorState.redoCurLines = []
          state.editorState.redoLineGroups = []
          state.editorState.redoRenders = []
        })
      },

      //****//

      updateAppState: (newState: Partial<AppState>) => {
        set(() => newState)
      },

      getBrushSize: (): number => {
        return (
          get().editorState.baseBrushSize * get().editorState.brushSizeScale
        )
      },

      showPromptInput: (): boolean => {
        const model = get().settings.model
        return (
          model.model_type !== MODEL_TYPE_INPAINT &&
          model.name !== PAINT_BY_EXAMPLE
        )
      },

      showSidePanel: (): boolean => {
        const model = get().settings.model
        return model.model_type !== MODEL_TYPE_INPAINT
      },

      setServerConfig: (newValue: ServerConfig) => {
        set((state) => {
          state.serverConfig = newValue
        })
      },

      updateSettings: (newSettings: Partial<Settings>) => {
        set((state) => {
          state.settings = {
            ...state.settings,
            ...newSettings,
          }
        })
      },

      updateFileManagerState: (newState: Partial<FileManagerState>) => {
        set((state) => {
          state.fileManagerState = {
            ...state.fileManagerState,
            ...newState,
          }
        })
      },

      updateInteractiveSegState: (newState: Partial<InteractiveSegState>) => {
        set((state) => {
          return {
            ...state,
            interactiveSegState: {
              ...state.interactiveSegState,
              ...newState,
            },
          }
        })
      },

      resetInteractiveSegState: () => {
        get().updateInteractiveSegState(defaultValues.interactiveSegState)
      },

      handleInteractiveSegAccept: () => {
        set((state) => {
          return {
            ...state,
            interactiveSegState: {
              ...defaultValues.interactiveSegState,
              interactiveSegMask:
                state.interactiveSegState.tmpInteractiveSegMask,
            },
          }
        })
      },

      setIsInpainting: (newValue: boolean) =>
        set((state) => {
          state.isInpainting = newValue
        }),

      setIsPluginRunning: (newValue: boolean) =>
        set((state) => {
          state.isPluginRunning = newValue
        }),

      setFile: (file: File) =>
        set((state) => {
          state.file = file
          state.interactiveSegState = castDraft(
            defaultValues.interactiveSegState
          )
          state.editorState = castDraft(defaultValues.editorState)
          state.cropperState = defaultValues.cropperState
        }),

      setCustomFile: (file: File) =>
        set((state) => {
          state.customMask = file
        }),

      setBaseBrushSize: (newValue: number) =>
        set((state) => {
          state.editorState.baseBrushSize = newValue
        }),

      setImageSize: (width: number, height: number) => {
        // 根据图片尺寸调整 brushSize 的 scale
        set((state) => {
          state.imageWidth = width
          state.imageHeight = height
          state.editorState.brushSizeScale =
            Math.max(Math.min(width, height), 512) / 512
        })
      },

      setCropperX: (newValue: number) =>
        set((state) => {
          state.cropperState.x = newValue
        }),

      setCropperY: (newValue: number) =>
        set((state) => {
          state.cropperState.y = newValue
        }),

      setCropperWidth: (newValue: number) =>
        set((state) => {
          state.cropperState.width = newValue
        }),

      setCropperHeight: (newValue: number) =>
        set((state) => {
          state.cropperState.height = newValue
        }),

      setSeed: (newValue: number) =>
        set((state) => {
          state.settings.seed = newValue
        }),
    })),
    {
      name: "ZUSTAND_STATE", // name of the item in the storage (must be unique)
      version: 0,
      partialize: (state) =>
        Object.fromEntries(
          Object.entries(state).filter(([key]) =>
            ["fileManagerState", "settings"].includes(key)
          )
        ),
    }
  ),
  shallow
)

// export const useStore = <U>(selector: (state: AppState & AppAction) => U) => {
//   return createWithEqualityFn(selector, shallow)
// }

// export const useStore = createWithEqualityFn(useBaseStore, shallow)

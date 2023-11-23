import { create, StoreApi, UseBoundStore } from "zustand"
import { persist } from "zustand/middleware"
import { immer } from "zustand/middleware/immer"
import { SortBy, SortOrder } from "./types"
import { DEFAULT_BRUSH_SIZE } from "./const"

type FileManagerState = {
  sortBy: SortBy
  sortOrder: SortOrder
  layout: "rows" | "masonry"
  searchText: string
}

type AppState = {
  file: File | null
  imageHeight: number
  imageWidth: number
  brushSize: number
  brushSizeScale: number

  isInpainting: boolean
  isInteractiveSeg: boolean // 是否正处于 sam 状态
  isInteractiveSegRunning: boolean
  interactiveSegClicks: number[][]

  prompt: string

  fileManagerState: FileManagerState
}

type AppAction = {
  setFile: (file: File) => void
  setIsInpainting: (newValue: boolean) => void
  setBrushSize: (newValue: number) => void
  setImageSize: (width: number, height: number) => void
  setFileManagerSortBy: (newValue: SortBy) => void
  setFileManagerSortOrder: (newValue: SortOrder) => void
  setFileManagerLayout: (
    newValue: AppState["fileManagerState"]["layout"]
  ) => void
  setFileManagerSearchText: (newValue: string) => void
  setPrompt: (newValue: string) => void
}

export const useStore = create<AppState & AppAction>()(
  immer(
    persist(
      (set, get) => ({
        file: null,
        imageHeight: 0,
        imageWidth: 0,
        brushSize: DEFAULT_BRUSH_SIZE,
        brushSizeScale: 1,
        isInpainting: false,
        isInteractiveSeg: false,
        isInteractiveSegRunning: false,
        interactiveSegClicks: [],
        prompt: "",
        fileManagerState: {
          sortBy: SortBy.CTIME,
          sortOrder: SortOrder.DESCENDING,
          layout: "masonry",
          searchText: "",
        },
        setIsInpainting: (newValue: boolean) =>
          set((state: AppState) => {
            state.isInpainting = newValue
          }),
        setFile: (file: File) =>
          set((state: AppState) => {
            // TODO: 清空各种状态
            state.file = file
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
        setPrompt: (newValue: string) =>
          set((state: AppState) => {
            state.prompt = newValue
          }),
        setFileManagerSortBy: (newValue: SortBy) =>
          set((state: AppState) => {
            state.fileManagerState.sortBy = newValue
          }),
        setFileManagerSortOrder: (newValue: SortOrder) =>
          set((state: AppState) => {
            state.fileManagerState.sortOrder = newValue
          }),
        setFileManagerLayout: (newValue: "rows" | "masonry") =>
          set((state: AppState) => {
            state.fileManagerState.layout = newValue
          }),
        setFileManagerSearchText: (newValue: string) =>
          set((state: AppState) => {
            state.fileManagerState.searchText = newValue
          }),
      }),
      {
        name: "ZUSTAND_STATE", // name of the item in the storage (must be unique)
        partialize: (state) =>
          Object.fromEntries(
            Object.entries(state).filter(([key]) =>
              ["fileManagerState", "prompt"].includes(key)
            )
          ),
      }
    )
  )
)

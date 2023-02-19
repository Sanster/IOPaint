import mitt from 'mitt'

export const EVENT_PROMPT = 'prompt'

export const EVENT_CUSTOM_MASK = 'custom_mask'
export interface CustomMaskEventData {
  mask: File
}

export const EVENT_PAINT_BY_EXAMPLE = 'paint_by_example'
export interface PaintByExampleEventData {
  image: File
}

export const RERUN_LAST_MASK = 'rerun_last_mask'

const emitter = mitt()

export default emitter

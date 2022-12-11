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

const emitter = mitt()

export default emitter

import mitt from 'mitt'

export const EVENT_PROMPT = 'prompt'
export const EVENT_CUSTOM_MASK = 'custom_mask'
export interface CustomMaskEventData {
  mask: File
}

const emitter = mitt()

export default emitter

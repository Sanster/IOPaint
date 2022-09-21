import mitt from 'mitt'

export const EVENT_PROMPT = 'prompt'
export const EVENT_RERUN = 'rerun_last_mask'

const emitter = mitt()

export default emitter

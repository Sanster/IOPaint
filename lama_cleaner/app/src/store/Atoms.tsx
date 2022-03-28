import { atom } from 'recoil'

export const fileState = atom<File | undefined>({
  key: 'fileState',
  default: undefined,
})

export const shortcutsState = atom<boolean>({
  key: 'shortcutsState',
  default: false,
})

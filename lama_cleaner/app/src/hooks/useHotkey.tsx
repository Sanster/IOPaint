import { Options, useHotkeys } from 'react-hotkeys-hook'
import { useRecoilValue } from 'recoil'
import { appState } from '../store/Atoms'

const useHotKey = (
  keys: string,
  callback: any,
  options?: Options,
  deps?: any[]
) => {
  const app = useRecoilValue(appState)

  const ref = useHotkeys(
    keys,
    callback,
    { ...options, enabled: !app.disableShortCuts },
    deps
  )
  return ref
}

export default useHotKey

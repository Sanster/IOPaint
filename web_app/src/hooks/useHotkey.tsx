import { useStore } from "@/lib/states"
import { useHotkeys } from "react-hotkeys-hook"

const useHotKey = (keys: string, callback: any, deps?: any[]) => {
  const disableShortCuts = useStore((state) => state.disableShortCuts)

  const ref = useHotkeys(keys, callback, { enabled: !disableShortCuts }, deps)
  return ref
}

export default useHotKey

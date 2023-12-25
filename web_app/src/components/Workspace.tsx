import { useEffect } from "react"
import Editor from "./Editor"
import { currentModel } from "@/lib/api"
import { useStore } from "@/lib/states"
import ImageSize from "./ImageSize"
import Plugins from "./Plugins"
import { InteractiveSeg } from "./InteractiveSeg"
import SidePanel from "./SidePanel"

const Workspace = () => {
  const [file, updateSettings] = useStore((state) => [
    state.file,
    state.updateSettings,
  ])

  useEffect(() => {
    currentModel()
      .then((res) => res.json())
      .then((model) => {
        updateSettings({ model })
      })
  }, [])

  return (
    <>
      <div className="flex gap-3 absolute top-[68px] left-[24px] items-center">
        <Plugins />
        <ImageSize />
      </div>
      <InteractiveSeg />
      <SidePanel />
      {file ? <Editor file={file} /> : <></>}
    </>
  )
}

export default Workspace

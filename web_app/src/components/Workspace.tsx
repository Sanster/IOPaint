import { useEffect } from "react"
import Editor from "./Editor"
import {
  AIModel,
  isPaintByExampleState,
  isPix2PixState,
  isSDState,
} from "@/lib/store"
import { currentModel } from "@/lib/api"
import { useStore } from "@/lib/states"
import ImageSize from "./ImageSize"
import Plugins from "./Plugins"
import { InteractiveSeg } from "./InteractiveSeg"
import SidePanel from "./SidePanel"
// import SidePanel from "./SidePanel/SidePanel"
// import PESidePanel from "./SidePanel/PESidePanel"
// import P2PSidePanel from "./SidePanel/P2PSidePanel"
// import Plugins from "./Plugins/Plugins"
// import Flex from "./shared/Layout"
// import ImageSize from "./ImageSize/ImageSize"

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
      {/* {isSD ? <SidePanel /> : <></>}
      {isPaintByExample ? <PESidePanel /> : <></>}
      {isPix2Pix ? <P2PSidePanel /> : <></>}
      {/* <SettingModal onClose={onSettingClose} /> */}
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

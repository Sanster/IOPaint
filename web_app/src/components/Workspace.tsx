import { useEffect } from "react"
import { useRecoilState, useRecoilValue, useSetRecoilState } from "recoil"
import Editor from "./Editor"
// import SettingModal from "./Settings/SettingsModal"
import {
  AIModel,
  isPaintByExampleState,
  isPix2PixState,
  isSDState,
  settingState,
} from "@/lib/store"
import { currentModel, modelDownloaded, switchModel } from "@/lib/api"
import { useStore } from "@/lib/states"
import ImageSize from "./ImageSize"
import Plugins from "./Plugins"
// import SidePanel from "./SidePanel/SidePanel"
// import PESidePanel from "./SidePanel/PESidePanel"
// import P2PSidePanel from "./SidePanel/P2PSidePanel"
// import Plugins from "./Plugins/Plugins"
// import Flex from "./shared/Layout"
// import ImageSize from "./ImageSize/ImageSize"

const Workspace = () => {
  const file = useStore((state) => state.file)
  const [settings, setSettingState] = useRecoilState(settingState)
  const isSD = useRecoilValue(isSDState)
  const isPaintByExample = useRecoilValue(isPaintByExampleState)
  const isPix2Pix = useRecoilValue(isPix2PixState)

  const onSettingClose = async () => {
    const curModel = await currentModel().then((res) => res.text())
    if (curModel === settings.model) {
      return
    }
    const downloaded = await modelDownloaded(settings.model).then((res) =>
      res.text()
    )

    const { model } = settings

    let loadingMessage = `Switching to ${model} model`
    let loadingDuration = 3000
    if (downloaded === "False") {
      loadingMessage = `Downloading ${model} model, this may take a while`
      loadingDuration = 9999999999
    }

    // TODO 修改成 Modal
    // setToastState({
    //   open: true,
    //   desc: loadingMessage,
    //   state: "loading",
    //   duration: loadingDuration,
    // })

    switchModel(model)
      .then((res) => {
        if (res.ok) {
          // setToastState({
          //   open: true,
          //   desc: `Switch to ${model} model success`,
          //   state: "success",
          //   duration: 3000,
          // })
        } else {
          throw new Error("Server error")
        }
      })
      .catch(() => {
        // setToastState({
        //   open: true,
        //   desc: `Switch to ${model} model failed`,
        //   state: "error",
        //   duration: 3000,
        // })
        setSettingState((old) => {
          return { ...old, model: curModel as AIModel }
        })
      })
  }

  useEffect(() => {
    currentModel()
      .then((res) => res.text())
      .then((model) => {
        setSettingState((old) => {
          return { ...old, model: model as AIModel }
        })
      })
  }, [setSettingState])

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
      {file ? <Editor file={file} /> : <></>}
    </>
  )
}

export default Workspace

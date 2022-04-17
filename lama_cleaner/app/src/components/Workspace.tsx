import React, { useEffect } from 'react'
import { useRecoilState } from 'recoil'
import Editor from './Editor/Editor'
import ShortcutsModal from './Shortcuts/ShortcutsModal'
import SettingModal from './Settings/SettingsModal'
import Toast from './shared/Toast'
import { Settings, settingState, toastState } from '../store/Atoms'
import {
  currentModel,
  modelDownloaded,
  switchModel,
} from '../adapters/inpainting'
import { AIModel } from './Settings/ModelSettingBlock'

interface WorkspaceProps {
  file: File
}

const Workspace = ({ file }: WorkspaceProps) => {
  const [settings, setSettingState] = useRecoilState(settingState)
  const [toastVal, setToastState] = useRecoilState(toastState)

  const onSettingClose = async () => {
    const curModel = await currentModel().then(res => res.text())
    if (curModel === settings.model) {
      return
    }
    const downloaded = await modelDownloaded(settings.model).then(res =>
      res.text()
    )

    const { model } = settings

    let loadingMessage = `Switching to ${model} model`
    let loadingDuration = 3000
    if (downloaded === 'False') {
      loadingMessage = `Downloading ${model} model, this may take a while`
      loadingDuration = 9999999999
    }

    setToastState({
      open: true,
      desc: loadingMessage,
      state: 'loading',
      duration: loadingDuration,
    })

    switchModel(model)
      .then(res => {
        if (res.ok) {
          setToastState({
            open: true,
            desc: `Switch to ${model} model success`,
            state: 'success',
            duration: 3000,
          })
        } else {
          throw new Error('Server error')
        }
      })
      .catch(() => {
        setToastState({
          open: true,
          desc: `Switch to ${model} model failed`,
          state: 'error',
          duration: 3000,
        })
        setSettingState(old => {
          return { ...old, model: curModel as AIModel }
        })
      })
  }

  useEffect(() => {
    currentModel()
      .then(res => res.text())
      .then(model => {
        setSettingState(old => {
          return { ...old, model: model as AIModel }
        })
      })
  }, [])

  return (
    <>
      <Editor file={file} />
      <SettingModal onClose={onSettingClose} />
      <ShortcutsModal />
      <Toast
        {...toastVal}
        onOpenChange={(open: boolean) => {
          setToastState(old => {
            return { ...old, open }
          })
        }}
      />
    </>
  )
}

export default Workspace

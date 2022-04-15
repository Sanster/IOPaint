import React from 'react'

import { useRecoilState } from 'recoil'
import { switchModel } from '../../adapters/inpainting'
import { settingState } from '../../store/Atoms'
import Modal from '../shared/Modal'
import HDSettingBlock from './HDSettingBlock'
import ModelSettingBlock from './ModelSettingBlock'

export default function SettingModal() {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onClose = () => {
    setSettingState(old => {
      return { ...old, show: false }
    })

    switchModel(setting.model)
  }

  return (
    <Modal
      onClose={onClose}
      title="Settings"
      className="modal-setting"
      show={setting.show}
    >
      {/* It's not possible because this poses a security risk */}
      {/* https://stackoverflow.com/questions/34870711/download-a-file-at-different-location-using-html5 */}
      {/* <SavePathSettingBlock /> */}
      <ModelSettingBlock />
      <HDSettingBlock />
    </Modal>
  )
}

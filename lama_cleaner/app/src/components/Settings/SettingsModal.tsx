import React from 'react'

import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import Modal from '../shared/Modal'
import HDSettingBlock from './HDSettingBlock'
import ModelSettingBlock from './ModelSettingBlock'

interface SettingModalProps {
  onClose: () => void
}
export default function SettingModal(props: SettingModalProps) {
  const { onClose } = props
  const [setting, setSettingState] = useRecoilState(settingState)

  const handleOnClose = () => {
    setSettingState(old => {
      return { ...old, show: false }
    })
    onClose()
  }

  return (
    <Modal
      onClose={handleOnClose}
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

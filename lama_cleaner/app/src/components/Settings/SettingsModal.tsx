import React from 'react'

import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import Modal from '../shared/Modal'
import ManualRunInpaintingSettingBlock from './ManualRunInpaintingSettingBlock'
import HDSettingBlock from './HDSettingBlock'
import ModelSettingBlock from './ModelSettingBlock'
import GraduallyInpaintingSettingBlock from './GraduallyInpaintingSettingBlock'
import DownloadMaskSettingBlock from './DownloadMaskSettingBlock'

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
      <ManualRunInpaintingSettingBlock />
      <GraduallyInpaintingSettingBlock />
      <DownloadMaskSettingBlock />
      <ModelSettingBlock />
      <HDSettingBlock />
    </Modal>
  )
}

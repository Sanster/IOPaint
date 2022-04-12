import React from 'react'

import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import Modal from '../shared/Modal'
import HDSettingBlock from './HDSettingBlock'
import SavePathSettingBlock from './SavePathSettingBlock'

export default function SettingModal() {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onClose = () => {
    setSettingState(old => {
      return { ...old, show: false }
    })
  }

  return (
    <Modal
      onClose={onClose}
      title="Settings"
      className="modal-setting"
      show={setting.show}
    >
      <SavePathSettingBlock />
      <HDSettingBlock />
    </Modal>
  )
}

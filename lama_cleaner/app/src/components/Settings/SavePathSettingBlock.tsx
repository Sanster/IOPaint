import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import { Switch, SwitchThumb } from '../shared/Switch'
import SettingBlock from './SettingBlock'

function SavePathSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onCheckChange = (checked: boolean) => {
    setSettingState(old => {
      return { ...old, saveImageBesideOrigin: checked }
    })
  }

  return (
    <SettingBlock
      title="Download image beside origin image"
      input={
        <Switch
          checked={setting.saveImageBesideOrigin}
          onCheckedChange={onCheckChange}
        >
          <SwitchThumb />
        </Switch>
      }
    />
  )
}

export default SavePathSettingBlock

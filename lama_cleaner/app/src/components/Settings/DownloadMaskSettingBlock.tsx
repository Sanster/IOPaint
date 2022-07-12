import React from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import { Switch, SwitchThumb } from '../shared/Switch'
import SettingBlock from './SettingBlock'

const DownloadMaskSettingBlock: React.FC = () => {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onCheckChange = (checked: boolean) => {
    setSettingState(old => {
      return { ...old, downloadMask: checked }
    })
  }

  return (
    <SettingBlock
      title="Download Mask"
      desc="Download inpainting result and mask"
      input={
        <Switch checked={setting.downloadMask} onCheckedChange={onCheckChange}>
          <SwitchThumb />
        </Switch>
      }
    />
  )
}

export default DownloadMaskSettingBlock

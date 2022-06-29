import React from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import { Switch, SwitchThumb } from '../shared/Switch'
import SettingBlock from './SettingBlock'

const GraduallyInpaintingSettingBlock: React.FC = () => {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onCheckChange = (checked: boolean) => {
    setSettingState(old => {
      return { ...old, graduallyInpainting: checked }
    })
  }

  return (
    <SettingBlock
      title="Gradually Inpainting"
      desc="If checked, perform inpainting on the last result, otherwise, always run the model on the initial image."
      input={
        <Switch
          checked={setting.graduallyInpainting}
          onCheckedChange={onCheckChange}
        >
          <SwitchThumb />
        </Switch>
      }
    />
  )
}

export default GraduallyInpaintingSettingBlock

import React from 'react'
import { useRecoilState } from 'recoil'
import { Cog6ToothIcon } from '@heroicons/react/24/outline'
import { settingState } from '../../store/Atoms'
import Button from '../shared/Button'

const SettingIcon = () => {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onClick = () => {
    setSettingState({ ...setting, show: !setting.show })
  }

  return (
    <div>
      <Button
        onClick={onClick}
        toolTip="Settings"
        style={{ border: 0 }}
        icon={<Cog6ToothIcon />}
      />
    </div>
  )
}

export default SettingIcon

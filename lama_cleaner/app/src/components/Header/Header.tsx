import { ArrowLeftIcon } from '@heroicons/react/outline'
import React from 'react'
import { useRecoilState } from 'recoil'
import { fileState } from '../../store/Atoms'
import Button from '../shared/Button'
import Shortcuts from '../Shortcuts/Shortcuts'
import useResolution from '../../hooks/useResolution'
import { ThemeChanger } from './ThemeChanger'
import SettingIcon from '../Settings/SettingIcon'

const Header = () => {
  const [file, setFile] = useRecoilState(fileState)
  const resolution = useResolution()

  const renderHeader = () => {
    return (
      <header>
        <div style={{ visibility: file ? 'visible' : 'hidden' }}>
          <Button
            icon={<ArrowLeftIcon />}
            onClick={() => {
              setFile(undefined)
            }}
            style={{ border: 0 }}
          >
            {resolution === 'desktop' ? 'Start New' : undefined}
          </Button>
        </div>

        <div className="header-icons-wrapper">
          <ThemeChanger />
          {file && (
            <div className="header-icons">
              <Shortcuts />
              <SettingIcon />
            </div>
          )}
        </div>
      </header>
    )
  }
  return renderHeader()
}

export default Header

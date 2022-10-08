import { ArrowLeftIcon, UploadIcon } from '@heroicons/react/outline'
import React, { useState } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import { fileState, isSDState } from '../../store/Atoms'
import Button from '../shared/Button'
import Shortcuts from '../Shortcuts/Shortcuts'
import useResolution from '../../hooks/useResolution'
import { ThemeChanger } from './ThemeChanger'
import SettingIcon from '../Settings/SettingIcon'
import PromptInput from './PromptInput'

const Header = () => {
  const [file, setFile] = useRecoilState(fileState)
  const resolution = useResolution()
  const [uploadElemId] = useState(`file-upload-${Math.random().toString()}`)
  const isSD = useRecoilValue(isSDState)

  const renderHeader = () => {
    return (
      <header>
        <div>
          <label htmlFor={uploadElemId}>
            <Button icon={<UploadIcon />} style={{ border: 0 }}>
              <input
                style={{ display: 'none' }}
                id={uploadElemId}
                name={uploadElemId}
                type="file"
                onChange={ev => {
                  const newFile = ev.currentTarget.files?.[0]
                  if (newFile) {
                    setFile(newFile)
                  }
                }}
                accept="image/png, image/jpeg"
              />
              {resolution === 'desktop' ? 'Upload New' : undefined}
            </Button>
          </label>
        </div>

        {isSD && file ? <PromptInput /> : <></>}

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

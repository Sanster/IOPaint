import { ArrowLeftIcon, UploadIcon } from '@heroicons/react/outline'
import React, { useState } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import { fileState, isInpaintingState, isSDState } from '../../store/Atoms'
import Button from '../shared/Button'
import Shortcuts from '../Shortcuts/Shortcuts'
import { ThemeChanger } from './ThemeChanger'
import SettingIcon from '../Settings/SettingIcon'
import PromptInput from './PromptInput'
import CoffeeIcon from '../CoffeeIcon/CoffeeIcon'
import emitter, { EVENT_CUSTOM_MASK } from '../../event'

const Header = () => {
  const isInpainting = useRecoilValue(isInpaintingState)
  const [file, setFile] = useRecoilState(fileState)
  const [uploadElemId] = useState(`file-upload-${Math.random().toString()}`)
  const [maskUploadElemId] = useState(`mask-upload-${Math.random().toString()}`)
  const isSD = useRecoilValue(isSDState)

  const renderHeader = () => {
    return (
      <header>
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 8,
          }}
        >
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
              Image
            </Button>
          </label>

          <label
            htmlFor={maskUploadElemId}
            style={{ visibility: file ? 'visible' : 'hidden' }}
          >
            <Button style={{ border: 0 }} disabled={isInpainting}>
              <input
                style={{ display: 'none' }}
                id={maskUploadElemId}
                name={maskUploadElemId}
                type="file"
                onClick={e => {
                  const element = e.target as HTMLInputElement
                  element.value = ''
                }}
                onChange={ev => {
                  const newFile = ev.currentTarget.files?.[0]
                  if (newFile) {
                    // TODO: check mask size
                    console.info('Send custom mask')
                    emitter.emit(EVENT_CUSTOM_MASK, { mask: newFile })
                  }
                }}
                accept="image/png, image/jpeg"
              />
              Mask
            </Button>
          </label>
        </div>

        {isSD && file ? <PromptInput /> : <></>}

        <div className="header-icons-wrapper">
          <CoffeeIcon />
          <ThemeChanger />
          <div className="header-icons">
            <Shortcuts />
            <SettingIcon />
          </div>
        </div>
      </header>
    )
  }
  return renderHeader()
}

export default Header

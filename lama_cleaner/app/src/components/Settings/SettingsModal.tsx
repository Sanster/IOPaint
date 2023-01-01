import React, { FormEvent } from 'react'

import { useRecoilState, useRecoilValue } from 'recoil'
import { FolderOpenIcon } from '@heroicons/react/24/outline'
import * as Tabs from '@radix-ui/react-tabs'
import {
  isPaintByExampleState,
  isSDState,
  settingState,
} from '../../store/Atoms'
import Modal from '../shared/Modal'
import ManualRunInpaintingSettingBlock from './ManualRunInpaintingSettingBlock'
import HDSettingBlock from './HDSettingBlock'
import ModelSettingBlock from './ModelSettingBlock'
import DownloadMaskSettingBlock from './DownloadMaskSettingBlock'
import useHotKey from '../../hooks/useHotkey'
import SettingBlock from './SettingBlock'
import { Switch, SwitchThumb } from '../shared/Switch'
import Button from '../shared/Button'
import TextInput from '../shared/Input'

declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    // extends React's HTMLAttributes
    directory?: string
    webkitdirectory?: string
  }
}

interface SettingModalProps {
  onClose: () => void
}

export default function SettingModal(props: SettingModalProps) {
  const { onClose } = props
  const [setting, setSettingState] = useRecoilState(settingState)
  const isSD = useRecoilValue(isSDState)
  const isPaintByExample = useRecoilValue(isPaintByExampleState)

  const handleOnClose = () => {
    setSettingState(old => {
      return { ...old, show: false }
    })
    onClose()
  }

  useHotKey(
    's',
    () => {
      setSettingState(old => {
        return { ...old, show: !old.show }
      })
    },
    {},
    []
  )

  return (
    <Modal
      onClose={handleOnClose}
      title="Settings"
      className="modal-setting"
      show={setting.show}
    >
      <Tabs.Root
        className="TabsRoot"
        defaultValue="tab1"
        orientation="vertical"
      >
        <Tabs.List className="TabsList">
          <Tabs.Trigger className="TabsTrigger" value="tab1">
            Model
          </Tabs.Trigger>
          <Tabs.Trigger className="TabsTrigger" value="tab2">
            File
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content className="TabsContent" value="tab1">
          {isSD || isPaintByExample ? (
            <></>
          ) : (
            <ManualRunInpaintingSettingBlock />
          )}
          <ModelSettingBlock />
          {isSD ? <></> : <HDSettingBlock />}
        </Tabs.Content>
        <Tabs.Content className="TabsContent" value="tab2">
          <DownloadMaskSettingBlock />
          <SettingBlock
            title="File Manager"
            desc="Toggle File Manager"
            input={
              <Switch
                checked={setting.enableFileManager}
                onCheckedChange={checked => {
                  setSettingState(old => {
                    return { ...old, enableFileManager: checked }
                  })
                }}
              >
                <SwitchThumb />
              </Switch>
            }
            optionDesc={
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 16,
                }}
              >
                <div className="folder-path-block">
                  <span>Image directory</span>
                  <TextInput
                    disabled={!setting.enableFileManager}
                    value={setting.imageDirectory}
                    placeholder="Image directory"
                    className="folder-path"
                    onInput={(evt: FormEvent<HTMLInputElement>) => {
                      evt.preventDefault()
                      evt.stopPropagation()
                      const target = evt.target as HTMLInputElement
                      setSettingState(old => {
                        return { ...old, imageDirectory: target.value }
                      })
                    }}
                  />
                </div>
                <div className="folder-path-block">
                  <span>Output directory</span>
                  <TextInput
                    disabled={!setting.enableFileManager}
                    value={setting.outputDirectory}
                    placeholder="Output directory"
                    className="folder-path"
                    onInput={(evt: FormEvent<HTMLInputElement>) => {
                      evt.preventDefault()
                      evt.stopPropagation()
                      const target = evt.target as HTMLInputElement
                      setSettingState(old => {
                        return { ...old, outputDirectory: target.value }
                      })
                    }}
                  />
                </div>
              </div>
            }
          />
        </Tabs.Content>
      </Tabs.Root>
    </Modal>
  )
}

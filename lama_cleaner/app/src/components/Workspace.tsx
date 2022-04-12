import React from 'react'
import { useRecoilValue } from 'recoil'
import Editor from './Editor/Editor'
import { shortcutsState } from '../store/Atoms'
import ShortcutsModal from './Shortcuts/ShortcutsModal'
import SettingModal from './Setting/SettingModal'

interface WorkspaceProps {
  file: File
}

const Workspace = ({ file }: WorkspaceProps) => {
  return (
    <>
      <Editor file={file} />
      <SettingModal />
      <ShortcutsModal />
    </>
  )
}

export default Workspace

import React from 'react'
import Editor from './Editor/Editor'
import ShortcutsModal from './Shortcuts/ShortcutsModal'
import SettingModal from './Settings/SettingsModal'

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

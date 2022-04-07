import React from 'react'
import { useRecoilValue } from 'recoil'
import Editor from './Editor/Editor'
import { shortcutsState } from '../store/Atoms'
import ShortcutsModal from './Shortcuts/ShortcutsModal'

interface WorkspaceProps {
  file: File
}

const Workspace = ({ file }: WorkspaceProps) => {
  const shortcutVisbility = useRecoilValue(shortcutsState)
  return (
    <>
      <Editor file={file} />
      <ShortcutsModal show={shortcutVisbility} />
    </>
  )
}

export default Workspace

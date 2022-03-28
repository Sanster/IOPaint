import React from 'react'
import { useRecoilValue } from 'recoil'
import Editor from './Editor/Editor'
import { shortcutsState } from '../store/Atoms'
import Header from './Header/Header'
import ShortcutsModal from './Shortcuts/ShortcutsModal'

interface WorkspaceProps {
  file: File
}

const Workspace = ({ file }: WorkspaceProps) => {
  const shortcutVisbility = useRecoilValue(shortcutsState)
  return (
    <>
      <Header />
      <Editor file={file} />
      {shortcutVisbility ? <ShortcutsModal /> : null}
    </>
  )
}

export default Workspace

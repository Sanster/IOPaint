import React, { ReactNode } from 'react'
import { useSetRecoilState } from 'recoil'
import { shortcutsState } from '../../store/Atoms'
import Modal from '../shared/Modal'

interface Shortcut {
  children: ReactNode
  content: string
}

function ShortCut(props: Shortcut) {
  const { children, content } = props

  return (
    <div className="shortcut-option">
      <div className="shortcut-key">{children}</div>
      <div className="shortcut-description">{content}</div>
    </div>
  )
}

export default function ShortcutsModal() {
  const setShortcutState = useSetRecoilState(shortcutsState)

  const shortcutStateHandler = () => {
    setShortcutState(prevShortcutState => !prevShortcutState)
  }

  return (
    <Modal
      onClose={shortcutStateHandler}
      title="Hotkeys"
      className="modal-shortcuts"
    >
      <div className="shortcut-options">
        <ShortCut content="Enable multi-stroke mask drawing">
          <p>Hold Cmd/Ctrl</p>
        </ShortCut>
        <ShortCut content="Undo inpainting">
          <p>Cmd/Ctrl + Z</p>
        </ShortCut>
        <ShortCut content="Pan">
          <p>Space & Drag</p>
        </ShortCut>
        <ShortCut content="View original image">
          <p>Hold Tab</p>
        </ShortCut>
        <ShortCut content="Reset zoom/pan & Cancel mask drawing">
          <p>Esc</p>
        </ShortCut>
        <ShortCut content="Decrease Brush Size">
          <p>[</p>
        </ShortCut>
        <ShortCut content="Increase Brush Size">
          <p>]</p>
        </ShortCut>
        <ShortCut content="Toggle Dark Mode">
          <p>Shift + D</p>
        </ShortCut>
        <ShortCut content="Toggle Hotkeys Panel">
          <p>H</p>
        </ShortCut>
      </div>
    </Modal>
  )
}

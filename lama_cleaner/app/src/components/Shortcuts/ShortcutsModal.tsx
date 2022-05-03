import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
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
      <div className="shortcut-description">{content}</div>
      <div className="shortcut-key">{children}</div>
    </div>
  )
}

export default function ShortcutsModal() {
  const [shortcutsShow, setShortcutState] = useRecoilState(shortcutsState)

  const shortcutStateHandler = () => {
    setShortcutState(false)
  }

  return (
    <Modal
      onClose={shortcutStateHandler}
      title="Hotkeys"
      className="modal-shortcuts"
      show={shortcutsShow}
    >
      <div className="shortcut-options">
        <ShortCut content="Enable Multi-Stroke Mask Drawing">
          <p>Hold Cmd/Ctrl</p>
        </ShortCut>
        <ShortCut content="Undo Inpainting">
          <p>Cmd/Ctrl + Z</p>
        </ShortCut>
        <ShortCut content="Pan">
          <p>Space & Drag</p>
        </ShortCut>
        <ShortCut content="View Original Image">
          <p>Hold Tab</p>
        </ShortCut>
        <ShortCut content="Reset Zoom/Pan">
          <p>Esc</p>
        </ShortCut>
        <ShortCut content="Cancel Mask Drawing">
          <p>Esc</p>
        </ShortCut>
        <ShortCut content="Run Inpainting Manually">
          <p>Shift + R</p>
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

import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { shortcutsState } from '../../store/Atoms'
import Modal from '../shared/Modal'

interface Shortcut {
  content: string
  keys: string[]
}

function ShortCut(props: Shortcut) {
  const { content, keys } = props

  return (
    <div className="shortcut-option">
      <div className="shortcut-description">{content}</div>
      <div style={{ display: 'flex', justifySelf: 'end', gap: '8px' }}>
        {keys.map((k, index) => (
          <div className="shortcut-key" key={k}>
            {k}
          </div>
        ))}
      </div>
    </div>
  )
}

const isMac = (function () {
  return /macintosh|mac os x/i.test(navigator.userAgent)
})()

const isWindows = (function () {
  return /windows|win32/i.test(navigator.userAgent)
})()

const CmdOrCtrl = isMac ? 'Cmd' : 'Ctrl'

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
        <ShortCut
          content="Enable Multi-Stroke Mask Drawing"
          keys={[`Hold ${CmdOrCtrl}`]}
        />
        <ShortCut content="Undo Inpainting" keys={[CmdOrCtrl, 'Z']} />
        <ShortCut content="Pan" keys={['Space & Drag']} />
        <ShortCut content="View Original Image" keys={['Hold Tag']} />
        <ShortCut content="Reset Zoom/Pan" keys={['Esc']} />
        <ShortCut content="Cancel Mask Drawing" keys={['Esc']} />
        <ShortCut content="Run Inpainting Manually" keys={['Shift', 'R']} />
        <ShortCut content="Decrease Brush Size" keys={['[']} />
        <ShortCut content="Increase Brush Size" keys={[']']} />
        <ShortCut content="Toggle Dark Mode" keys={['Shift', 'D']} />
        <ShortCut content="Toggle Hotkeys Panel" keys={['H']} />
      </div>
    </Modal>
  )
}

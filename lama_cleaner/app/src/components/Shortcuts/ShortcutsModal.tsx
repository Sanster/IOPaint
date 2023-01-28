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
        <div className="shortcut-options-column">
          <ShortCut content="Pan" keys={['Space + Drag']} />
          <ShortCut content="Reset Zoom/Pan" keys={['Esc']} />
          <ShortCut content="Decrease Brush Size" keys={['[']} />
          <ShortCut content="Increase Brush Size" keys={[']']} />
          <ShortCut content="View Original Image" keys={['Hold Tab']} />
          <ShortCut
            content="Multi-Stroke Drawing"
            keys={[`Hold ${CmdOrCtrl}`]}
          />
          <ShortCut content="Cancel Drawing" keys={['Esc']} />
        </div>

        <div className="shortcut-options-column">
          <ShortCut content="Undo" keys={[CmdOrCtrl, 'Z']} />
          <ShortCut content="Redo" keys={[CmdOrCtrl, 'Shift', 'Z']} />
          <ShortCut content="Copy Result" keys={[CmdOrCtrl, 'C']} />
          <ShortCut content="Paste Image" keys={[CmdOrCtrl, 'V']} />
          <ShortCut
            content="Trigger Manually Inpainting"
            keys={['Shift', 'R']}
          />
          <ShortCut content="Trigger Interactive Segmentation" keys={['I']} />
        </div>

        <div className="shortcut-options-column">
          <ShortCut content="Switch Theme" keys={['Shift', 'D']} />
          <ShortCut content="Toggle Hotkeys Dialog" keys={['H']} />
          <ShortCut content="Toggle Settings Dialog" keys={['S']} />
          <ShortCut content="Toggle File Manager" keys={['F']} />
        </div>
      </div>
    </Modal>
  )
}

import { ArrowLeftIcon } from '@heroicons/react/outline'
import React, { ReactNode } from 'react'
import Modal from './Modal'

interface Shortcut {
  children: ReactNode
  content: string
}

function ShortCut(props: Shortcut) {
  const { children, content } = props

  return (
    <div className="h-full flex flex-row space-x-6 justify-between">
      <div className="mr-12 border-2 rounded-xl px-2 py-1">{children}</div>
      <div className="flex flex-col justify-center">{content}</div>
    </div>
  )
}

interface ShortcutsModalProps {
  onClose?: () => void
}

export default function ShortcutsModal(props: ShortcutsModalProps) {
  const { onClose } = props
  return (
    <Modal onClose={onClose} className="h-full sm:h-auto p-0 sm:p-0">
      <div className="h-full sm:h-auto flex flex-col sm:flex-row">
        <div className="flex sm:p-14 flex flex-col justify-center space-y-6">
          <ShortCut content="Enable multi-stroke mask drawing">
            <p>Hold Cmd/Ctrl</p>
          </ShortCut>
          <ShortCut content="Enable panning">
            <p>Hold Space</p>
          </ShortCut>
          <ShortCut content="View original image">
            <p>Hold Tab</p>
          </ShortCut>
          <ShortCut content="Reset zoom/pan & Cancel mask drawing">
            <p>Esc</p>
          </ShortCut>
        </div>
      </div>
    </Modal>
  )
}

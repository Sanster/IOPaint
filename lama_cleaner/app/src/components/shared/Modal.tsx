import { XIcon } from '@heroicons/react/outline'
import React, { ReactNode, useRef } from 'react'
import { useClickAway, useKey, useKeyPress, useKeyPressEvent } from 'react-use'
import Button from './Button'

export interface ModalProps {
  show: boolean
  children?: ReactNode
  onClose?: () => void
  title: string
  className?: string
}

export default function Modal(props: ModalProps) {
  const { show, children, onClose, className, title } = props
  const ref = useRef(null)

  useClickAway(ref, () => {
    onClose?.()
  })

  useKeyPressEvent('Escape', e => {
    onClose?.()
  })

  return (
    <div
      className="modal-mask"
      style={{ visibility: show === true ? 'visible' : 'hidden' }}
    >
      <div ref={ref} className={`modal ${className}`}>
        <div className="modal-header">
          <h2>{title}</h2>
          <Button icon={<XIcon />} onClick={onClose} />
        </div>
        {children}
      </div>
    </div>
  )
}

import { XIcon } from '@heroicons/react/outline'
import React, { ReactNode, useRef } from 'react'
import { useClickAway, useKey } from 'react-use'
import Button from './Button'

interface ModalProps {
  children?: ReactNode
  onClose?: () => void
  title: string
  className?: string
}

export default function Modal(props: ModalProps) {
  const { children, onClose, className, title } = props
  const ref = useRef(null)

  useClickAway(ref, () => {
    onClose?.()
  })

  useKey('Escape', onClose, {
    event: 'keydown',
  })

  return (
    <div ref={ref} className={`modal ${className}`}>
      <div className="modal-header">
        <h3>{title}</h3>
        <Button icon={<XIcon />} onClick={onClose} />
      </div>
      {children}
    </div>
  )
}

import { XIcon } from '@heroicons/react/outline'
import React, { ReactNode } from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import Button from './Button'

export interface ModalProps {
  show: boolean
  children?: ReactNode
  onClose?: () => void
  title: string
  className?: string
}

const Modal = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Root>,
  ModalProps
>((props, forwardedRef) => {
  const { show, children, onClose, className, title } = props

  const onOpenChange = (open: boolean) => {
    if (!open) {
      onClose?.()
    }
  }

  return (
    <DialogPrimitive.Root open={show} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="modal-mask" />
        <DialogPrimitive.Content
          ref={forwardedRef}
          className={`modal ${className}`}
        >
          <div className="modal-header">
            <DialogPrimitive.Title>{title}</DialogPrimitive.Title>
            <Button icon={<XIcon />} onClick={onClose} />
          </div>
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
})

export default Modal

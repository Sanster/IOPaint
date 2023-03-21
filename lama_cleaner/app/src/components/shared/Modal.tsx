import { XMarkIcon } from '@heroicons/react/24/outline'
import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import Button from './Button'
import { appState } from '../../store/Atoms'

export interface ModalProps {
  show: boolean
  children?: ReactNode
  onClose?: () => void
  title: string | ReactNode
  showCloseIcon?: boolean
  className?: string
}

const Modal = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Root>,
  ModalProps
>((props, forwardedRef) => {
  const { show, children, onClose, className, title, showCloseIcon } = props
  const [_, setAppState] = useRecoilState(appState)

  const onOpenChange = (open: boolean) => {
    if (!open) {
      onClose?.()
      setAppState(old => {
        return { ...old, disableShortCuts: false }
      })
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
            {showCloseIcon ? (
              <Button icon={<XMarkIcon />} onClick={onClose} />
            ) : (
              <></>
            )}
          </div>
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
})

Modal.defaultProps = {
  showCloseIcon: true,
}

export default Modal

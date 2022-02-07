import { XIcon } from '@heroicons/react/outline'
import React, { ReactNode, useRef } from 'react'
import { useClickAway } from 'react-use'
import Button from './Button'

interface ModalProps {
  children?: ReactNode
  onClose?: () => void
  className?: string
}

export default function Modal(props: ModalProps) {
  const { children, onClose, className } = props
  const ref = useRef(null)

  useClickAway(ref, () => {
    onClose?.()
  })

  return (
    <div
      className={[
        'absolute w-full h-full flex justify-center items-center',
        'z-20',
        'bg-gray-300 bg-opacity-40 backdrop-filter backdrop-blur-md',
      ].join(' ')}
    >
      <div
        ref={ref}
        className={`bg-white max-w-4xl relative rounded-md shadow-md ${
          className || 'p-8 sm:p-12'
        }`}
      >
        <Button
          icon={<XIcon className="w-6 h-6" />}
          className={[
            'absolute right-4 top-4 rounded-full bg-gray-100 w-10 h-10',
            'flex justify-center items-center py-0 px-0 sm:px-0',
          ].join(' ')}
          onClick={onClose}
        />
        {children}
      </div>
    </div>
  )
}

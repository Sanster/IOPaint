import React, { ReactNode } from 'react'

interface ModalProps {
  children?: ReactNode
}

export default function Modal(props: ModalProps) {
  const { children } = props
  return (
    <div
      className={[
        'absolute w-full h-full flex justify-center items-center',
        'bg-white bg-opacity-40 backdrop-filter backdrop-blur-md',
      ].join(' ')}
    >
      <div className="bg-primary p-16 max-w-4xl">{children}</div>
    </div>
  )
}

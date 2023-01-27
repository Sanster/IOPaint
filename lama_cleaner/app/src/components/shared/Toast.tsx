import * as React from 'react'
import * as ToastPrimitive from '@radix-ui/react-toast'
import { ToastProps } from '@radix-ui/react-toast'
import { CheckIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline'

export const LoadingIcon = () => {
  return (
    <span className="loading-icon">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="12" y1="2" x2="12" y2="6" />
        <line x1="12" y1="18" x2="12" y2="22" />
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
        <line x1="2" y1="12" x2="6" y2="12" />
        <line x1="18" y1="12" x2="22" y2="12" />
        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
        <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
      </svg>
    </span>
  )
}

export type ToastState = 'default' | 'error' | 'loading' | 'success'

interface MyToastProps extends ToastProps {
  desc: string
  state?: ToastState
}

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitive.Root>,
  MyToastProps
>((props, forwardedRef) => {
  const { state, desc, ...itemProps } = props

  const getIcon = () => {
    switch (state) {
      case 'error':
        return <ExclamationCircleIcon className="error-icon" />
      case 'success':
        return <CheckIcon className="success-icon" />
      case 'loading':
        return <LoadingIcon />
      default:
        return <></>
    }
  }

  return (
    <ToastPrimitive.Provider>
      <ToastPrimitive.Root
        {...itemProps}
        ref={forwardedRef}
        className={`toast-root ${state}`}
      >
        <div className="toast-icon">{getIcon()}</div>
        <ToastPrimitive.Description className="toast-desc">
          {desc}
        </ToastPrimitive.Description>
      </ToastPrimitive.Root>
      <ToastPrimitive.Viewport className="toast-viewpoint" />
    </ToastPrimitive.Provider>
  )
})

Toast.defaultProps = {
  state: 'loading',
}

export default Toast

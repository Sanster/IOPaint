import React, { FocusEvent, TextareaHTMLAttributes } from 'react'
import { useRecoilState } from 'recoil'
import { appState } from '../../store/Atoms'

const TextAreaInput = React.forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement>
>((props, ref) => {
  const { onFocus, onBlur, ...itemProps } = props
  const [_, setAppState] = useRecoilState(appState)

  const handleOnFocus = (evt: FocusEvent<any>) => {
    setAppState(old => {
      return { ...old, disableShortCuts: true }
    })
    onFocus?.(evt)
  }

  const handleOnBlur = (evt: FocusEvent<any>) => {
    setAppState(old => {
      return { ...old, disableShortCuts: false }
    })
    onBlur?.(evt)
  }

  return (
    <textarea
      {...itemProps}
      ref={ref}
      onFocus={handleOnFocus}
      onBlur={handleOnBlur}
      onPaste={evt => evt.stopPropagation()}
      onKeyDown={e => {
        if (e.key === 'Escape') {
          e.currentTarget.blur()
        }
        if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
          e.stopPropagation()
        }
      }}
    />
  )
})

export default TextAreaInput

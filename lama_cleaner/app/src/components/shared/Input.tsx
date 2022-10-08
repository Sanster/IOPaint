import React, { FocusEvent, InputHTMLAttributes, RefObject } from 'react'
import { useClickAway } from 'react-use'
import { useRecoilState } from 'recoil'
import { appState } from '../../store/Atoms'

const TextInput = React.forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement>
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
    <input
      {...itemProps}
      ref={ref}
      type="text"
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

export default TextInput

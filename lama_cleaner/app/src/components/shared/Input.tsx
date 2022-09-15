import React, { FocusEvent, InputHTMLAttributes } from 'react'
import { useRecoilState } from 'recoil'
import { appState } from '../../store/Atoms'

const TextInput = React.forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement>
>((props: InputHTMLAttributes<HTMLInputElement>, forwardedRef) => {
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
      ref={forwardedRef}
      type="text"
      onFocus={handleOnFocus}
      onBlur={handleOnBlur}
      onKeyDown={e => {
        if (e.key === 'Escape') {
          e.currentTarget.blur()
        }
      }}
    />
  )
})

export default TextInput

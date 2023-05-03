import React, { FormEvent, useRef, useState } from 'react'
import { useClickAway } from 'react-use'
import { useRecoilState, useRecoilValue } from 'recoil'
import emitter, {
  DREAM_BUTTON_MOUSE_ENTER,
  DREAM_BUTTON_MOUSE_LEAVE,
  EVENT_PROMPT,
} from '../../event'
import { appState, isInpaintingState, propmtState } from '../../store/Atoms'
import Button from '../shared/Button'
import TextInput from '../shared/Input'

// TODO: show progress in input
const PromptInput = () => {
  const app = useRecoilValue(appState)
  const [prompt, setPrompt] = useRecoilState(propmtState)
  const isInpainting = useRecoilValue(isInpaintingState)
  const ref = useRef(null)

  const handleOnInput = (evt: FormEvent<HTMLInputElement>) => {
    evt.preventDefault()
    evt.stopPropagation()
    const target = evt.target as HTMLInputElement
    setPrompt(target.value)
  }

  const handleRepaintClick = () => {
    if (prompt.length !== 0 && !app.isInpainting) {
      emitter.emit(EVENT_PROMPT)
    }
  }

  useClickAway<MouseEvent>(ref, () => {
    if (ref?.current) {
      const input = ref.current as HTMLInputElement
      input.blur()
    }
  })

  const onKeyUp = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isInpainting) {
      handleRepaintClick()
    }
  }

  const onMouseEnter = () => {
    emitter.emit(DREAM_BUTTON_MOUSE_ENTER)
  }

  const onMouseLeave = () => {
    emitter.emit(DREAM_BUTTON_MOUSE_LEAVE)
  }

  return (
    <div className="prompt-wrapper">
      <TextInput
        ref={ref}
        value={prompt}
        onInput={handleOnInput}
        onKeyUp={onKeyUp}
        placeholder="I want to repaint of..."
      />
      <Button
        border
        onClick={handleRepaintClick}
        disabled={prompt.length === 0 || app.isInpainting}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
      >
        Dream
      </Button>
    </div>
  )
}

export default PromptInput

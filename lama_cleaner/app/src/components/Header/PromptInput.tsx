import React, { FormEvent, useState } from 'react'
import { useRecoilState } from 'recoil'
import emitter, { EVENT_PROMPT } from '../../event'
import { appState, propmtState } from '../../store/Atoms'
import Button from '../shared/Button'
import TextInput from '../shared/Input'

// TODO: show progress in input
const PromptInput = () => {
  const [app, setAppState] = useRecoilState(appState)
  const [prompt, setPrompt] = useRecoilState(propmtState)

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

  return (
    <div className="prompt-wrapper">
      <TextInput
        value={prompt}
        onInput={handleOnInput}
        placeholder="I want to repaint of..."
      />
      <Button
        border
        onClick={handleRepaintClick}
        disabled={prompt.length === 0 || app.isInpainting}
      >
        RePaint
      </Button>
    </div>
  )
}

export default PromptInput

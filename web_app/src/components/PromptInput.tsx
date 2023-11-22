import React, { FormEvent } from "react"
import { useRecoilState, useRecoilValue } from "recoil"
import emitter, {
  DREAM_BUTTON_MOUSE_ENTER,
  DREAM_BUTTON_MOUSE_LEAVE,
  EVENT_PROMPT,
} from "@/lib/event"
import { appState, isInpaintingState, propmtState } from "@/lib/store"
import { Button } from "./ui/button"
import { Input } from "./ui/input"

const PromptInput = () => {
  const app = useRecoilValue(appState)
  const [prompt, setPrompt] = useRecoilState(propmtState)
  const isInpainting = useRecoilValue(isInpaintingState)

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

  const onKeyUp = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isInpainting) {
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
    <div className="flex gap-4 items-center">
      <Input
        className="min-w-[600px]"
        value={prompt}
        onInput={handleOnInput}
        onKeyUp={onKeyUp}
        placeholder="I want to repaint of..."
      />
      <Button
        size="sm"
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

import React, { FormEvent } from "react"
import emitter, {
  DREAM_BUTTON_MOUSE_ENTER,
  DREAM_BUTTON_MOUSE_LEAVE,
  EVENT_PROMPT,
} from "@/lib/event"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { useStore } from "@/lib/states"

const PromptInput = () => {
  const [isInpainting, prompt, updateSettings] = useStore((state) => [
    state.isInpainting,
    state.settings.prompt,
    state.updateSettings,
  ])

  const handleOnInput = (evt: FormEvent<HTMLInputElement>) => {
    evt.preventDefault()
    evt.stopPropagation()
    const target = evt.target as HTMLInputElement
    updateSettings({ prompt: target.value })
  }

  const handleRepaintClick = () => {
    if (prompt.length !== 0 && isInpainting) {
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
        disabled={prompt.length === 0 || isInpainting}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
      >
        Dream
      </Button>
    </div>
  )
}

export default PromptInput

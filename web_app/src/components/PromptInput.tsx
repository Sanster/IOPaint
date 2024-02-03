import React, { FormEvent, useRef } from "react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { useStore } from "@/lib/states"
import { useClickAway } from "react-use"

const PromptInput = () => {
  const [
    isProcessing,
    prompt,
    updateSettings,
    runInpainting,
    showPrevMask,
    hidePrevMask,
  ] = useStore((state) => [
    state.getIsProcessing(),
    state.settings.prompt,
    state.updateSettings,
    state.runInpainting,
    state.showPrevMask,
    state.hidePrevMask,
  ])
  const ref = useRef(null)
  useClickAway<MouseEvent>(ref, () => {
    if (ref?.current) {
      const input = ref.current as HTMLInputElement
      input.blur()
    }
  })

  const handleOnInput = (evt: FormEvent<HTMLInputElement>) => {
    evt.preventDefault()
    evt.stopPropagation()
    const target = evt.target as HTMLInputElement
    updateSettings({ prompt: target.value })
  }

  const handleRepaintClick = () => {
    if (!isProcessing) {
      runInpainting()
    }
  }

  const onKeyUp = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isProcessing) {
      handleRepaintClick()
    }
  }

  const onMouseEnter = () => {
    showPrevMask()
  }

  const onMouseLeave = () => {
    hidePrevMask()
  }

  return (
    <div className="flex gap-4 items-center">
      <Input
        ref={ref}
        className="min-w-[500px]"
        value={prompt}
        onInput={handleOnInput}
        onKeyUp={onKeyUp}
        placeholder="I want to repaint of..."
      />
      <Button
        size="sm"
        onClick={handleRepaintClick}
        disabled={isProcessing}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
      >
        Paint
      </Button>
    </div>
  )
}

export default PromptInput

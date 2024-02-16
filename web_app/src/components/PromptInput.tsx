import React, { FormEvent, useRef } from "react"
import { Button } from "./ui/button"
import { useStore } from "@/lib/states"
import { useClickAway, useToggle } from "react-use"
import { Textarea } from "./ui/textarea"
import { cn } from "@/lib/utils"

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

  const [showScroll, toggleShowScroll] = useToggle(false)

  const ref = useRef(null)
  useClickAway<MouseEvent>(ref, () => {
    if (ref?.current) {
      const input = ref.current as HTMLTextAreaElement
      input.blur()
    }
  })

  const handleOnInput = (evt: FormEvent<HTMLTextAreaElement>) => {
    evt.preventDefault()
    evt.stopPropagation()
    const target = evt.target as HTMLTextAreaElement
    updateSettings({ prompt: target.value })
  }

  const handleRepaintClick = () => {
    if (!isProcessing) {
      runInpainting()
    }
  }

  const onKeyUp = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.ctrlKey && prompt.length !== 0) {
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
    <div className="flex gap-4 relative w-full justify-center h-full">
      <div className="absolute flex gap-4">
        <Textarea
          ref={ref}
          placeholder="I want to repaint of..."
          className={cn(
            showScroll ? "focus:overflow-y-auto" : "overflow-y-hidden",
            "min-h-[32px] h-[32px] overflow-x-hidden focus:h-[120px] overflow-y-hidden transition-[height] w-[500px] py-1 px-3 bg-background resize-none"
          )}
          style={{
            scrollbarGutter: "stable",
          }}
          value={prompt}
          onInput={handleOnInput}
          onKeyUp={onKeyUp}
          onTransitionEnd={toggleShowScroll}
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
    </div>
  )
}

export default PromptInput

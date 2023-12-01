import { useStore } from "@/lib/states"
import { Button } from "./ui/button"
import { Dialog, DialogContent, DialogTitle } from "./ui/dialog"
import { MousePointerClick } from "lucide-react"
import { DropdownMenuItem } from "./ui/dropdown-menu"

interface InteractiveSegReplaceModal {
  show: boolean
  onClose: () => void
  onCleanClick: () => void
  onReplaceClick: () => void
}

const InteractiveSegReplaceModal = (props: InteractiveSegReplaceModal) => {
  const { show, onClose, onCleanClick, onReplaceClick } = props

  const onOpenChange = (open: boolean) => {
    if (!open) {
      onClose()
    }
  }

  return (
    <Dialog open={show} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogTitle>Do you want to remove it or create a new one?</DialogTitle>
        <div className="flex gap-[12px] w-full justify-end items-center">
          <Button
            onClick={() => {
              onClose()
              onCleanClick()
            }}
          >
            Remove
          </Button>
          <Button onClick={onReplaceClick}>Create new</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

const InteractiveSegConfirmActions = () => {
  const [interactiveSegState, resetInteractiveSegState] = useStore((state) => [
    state.interactiveSegState,
    state.resetInteractiveSegState,
  ])

  if (!interactiveSegState.isInteractiveSeg) {
    return null
  }

  const onAcceptClick = () => {
    resetInteractiveSegState()
  }

  return (
    <div className="z-10 absolute top-[68px] rounded-xl border-solid border p-[8px] left-1/2 translate-x-[-50%] flex justify-center items-center gap-[8px] bg-background">
      <Button
        onClick={() => {
          resetInteractiveSegState()
        }}
        size="sm"
        variant="secondary"
      >
        Cancel
      </Button>
      <Button
        size="sm"
        onClick={() => {
          onAcceptClick()
        }}
      >
        Accept
      </Button>
    </div>
  )
}

interface ItemProps {
  x: number
  y: number
  positive: boolean
}

const Item = (props: ItemProps) => {
  const { x, y, positive } = props
  const name = positive
    ? "bg-[rgba(21,_215,_121,_0.936)] outline-[6px_solid_rgba(98,_255,_179,_0.31)]"
    : "bg-[rgba(237,_49,_55,_0.942)] outline-[6px_solid_rgba(255,_89,_95,_0.31)]"
  return (
    <div
      className={`absolute h-[8px] w-[8px] rounded-[50%] ${name}`}
      style={{
        left: x,
        top: y,
        transform: "translate(-50%, -50%)",
      }}
    />
  )
}

const InteractiveSegPoints = () => {
  const clicks = useStore((state) => state.interactiveSegState.clicks)

  return (
    <div className="absolute h-full w-full overflow-hidden pointer-events-none">
      {clicks.map((click) => {
        return (
          <Item
            key={click[3]}
            x={click[0]}
            y={click[1]}
            positive={click[2] === 1}
          />
        )
      })}
    </div>
  )
}

const InteractiveSeg = () => {
  const [interactiveSegState, updateInteractiveSegState] = useStore((state) => [
    state.interactiveSegState,
    state.updateInteractiveSegState,
  ])

  return (
    <div>
      <InteractiveSegConfirmActions />
      {/* <InteractiveSegReplaceModal /> */}
    </div>
  )
}

export { InteractiveSeg, InteractiveSegPoints }

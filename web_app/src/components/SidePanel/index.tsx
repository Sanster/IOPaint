import { useToggle } from "react-use"
import { useStore } from "@/lib/states"
import { Separator } from "../ui/separator"
import { ScrollArea } from "../ui/scroll-area"
import { Sheet, SheetContent, SheetHeader, SheetTrigger } from "../ui/sheet"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "../ui/button"
import useHotKey from "@/hooks/useHotkey"
import { RowContainer } from "./LabelTitle"
import { CV2, LDM, MODEL_TYPE_INPAINT } from "@/lib/const"
import LDMOptions from "./LDMOptions"
import DiffusionOptions from "./DiffusionOptions"
import CV2Options from "./CV2Options"

const SidePanel = () => {
  const [settings, windowSize] = useStore((state) => [
    state.settings,
    state.windowSize,
  ])

  const [open, toggleOpen] = useToggle(true)

  useHotKey("c", () => {
    toggleOpen()
  })

  if (
    settings.model.name !== LDM &&
    settings.model.name !== CV2 &&
    settings.model.model_type === MODEL_TYPE_INPAINT
  ) {
    return null
  }

  const renderSidePanelOptions = () => {
    if (settings.model.name === LDM) {
      return <LDMOptions />
    }
    if (settings.model.name === CV2) {
      return <CV2Options />
    }
    return <DiffusionOptions />
  }

  return (
    <Sheet open={open} modal={false}>
      <SheetTrigger
        tabIndex={-1}
        className="z-10 outline-none absolute top-[68px] right-6 rounded-lg border bg-background"
        hidden={open}
      >
        <Button
          variant="ghost"
          size="icon"
          asChild
          className="p-1.5"
          onClick={toggleOpen}
        >
          <ChevronLeft strokeWidth={1} />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-[286px] mt-[60px] outline-none pl-3 pr-1"
        onOpenAutoFocus={(event) => event.preventDefault()}
        onPointerDownOutside={(event) => event.preventDefault()}
      >
        <SheetHeader>
          <RowContainer>
            <div className="overflow-hidden mr-8">
              {
                settings.model.name.split("/")[
                  settings.model.name.split("/").length - 1
                ]
              }
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="border h-6 w-6"
              onClick={toggleOpen}
            >
              <ChevronRight strokeWidth={1} />
            </Button>
          </RowContainer>
          <Separator />
        </SheetHeader>
        <ScrollArea style={{ height: windowSize.height - 160 }}>
          {renderSidePanelOptions()}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}

export default SidePanel

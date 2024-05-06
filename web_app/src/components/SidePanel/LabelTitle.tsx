import { cn } from "@/lib/utils"
import { Button } from "../ui/button"
import { Label } from "../ui/label"
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip"

const RowContainer = ({ children }: { children: React.ReactNode }) => (
  <div className="flex justify-between items-center pr-4">{children}</div>
)

const LabelTitle = ({
  text,
  toolTip = "",
  url,
  htmlFor,
  disabled = false,
  className = "",
}: {
  text: string
  toolTip?: string
  url?: string
  htmlFor?: string
  disabled?: boolean
  className?: string
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Label
          htmlFor={htmlFor ? htmlFor : text.toLowerCase().replace(" ", "-")}
          className={cn("font-medium min-w-[65px]", className)}
          disabled={disabled}
        >
          {text}
        </Label>
      </TooltipTrigger>
      {toolTip || url ? (
        <TooltipContent className="flex flex-col max-w-xs text-sm" side="left">
          <p>{toolTip}</p>
          {url ? (
            <Button variant="link" className="justify-end">
              <a href={url} target="_blank">
                More info
              </a>
            </Button>
          ) : (
            <></>
          )}
        </TooltipContent>
      ) : (
        <></>
      )}
    </Tooltip>
  )
}

export { LabelTitle, RowContainer }

import { Button } from "../ui/button"
import { Label } from "../ui/label"
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip"

const RowContainer = ({ children }: { children: React.ReactNode }) => (
  <div className="flex justify-between items-center pr-2">{children}</div>
)

const LabelTitle = ({
  text,
  toolTip = "",
  url,
  htmlFor,
  disabled = false,
}: {
  text: string
  toolTip?: string
  url?: string
  htmlFor?: string
  disabled?: boolean
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Label
          htmlFor={htmlFor ? htmlFor : text.toLowerCase().replace(" ", "-")}
          className="font-medium"
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

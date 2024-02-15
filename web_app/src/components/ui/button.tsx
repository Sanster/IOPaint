import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"
import { Input } from "./input"
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90",
        outline:
          "border border-input bg-transparent shadow-sm hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(
          buttonVariants({ variant, size, className }),
          "outline-none cursor-default select-none"
        )}
        ref={ref}
        tabIndex={-1}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export interface IconButtonProps extends ButtonProps {
  tooltip: string
}

const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ tooltip, children, ...rest }, ref) => {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            {...rest}
            ref={ref}
            tabIndex={-1}
            className="cursor-default bg-background"
          >
            <div className="icon-button-icon-wrapper">{children}</div>
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    )
  }
)

export interface UploadButtonProps extends IconButtonProps {
  onFileUpload: (file: File) => void
}

const ImageUploadButton = (props: UploadButtonProps) => {
  const { onFileUpload, children, ...rest } = props

  const [uploadElemId] = React.useState(
    `file-upload-${Math.random().toString()}`
  )

  const handleChange = (ev: React.ChangeEvent<HTMLInputElement>) => {
    const newFile = ev.currentTarget.files?.[0]
    if (newFile) {
      onFileUpload(newFile)
    }
  }

  return (
    <>
      <label htmlFor={uploadElemId}>
        <IconButton {...rest} asChild>
          {children}
        </IconButton>
      </label>
      <Input
        style={{ display: "none" }}
        id={uploadElemId}
        name={uploadElemId}
        type="file"
        onChange={handleChange}
        accept="image/png, image/jpeg"
      />
    </>
  )
}

export { Button, IconButton, ImageUploadButton, buttonVariants }

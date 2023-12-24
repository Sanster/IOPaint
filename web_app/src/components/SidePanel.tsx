import { FormEvent, useState } from "react"
import { useToggle } from "react-use"
import { useStore } from "@/lib/states"
import { Switch } from "./ui/switch"
import { Label } from "./ui/label"
import { NumberInput } from "./ui/input"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"
import { Textarea } from "./ui/textarea"
import { SDSampler } from "@/lib/types"
import { Separator } from "./ui/separator"
import { ScrollArea } from "./ui/scroll-area"
import { Sheet, SheetContent, SheetHeader, SheetTrigger } from "./ui/sheet"
import {
  ArrowDownFromLine,
  ArrowLeftFromLine,
  ArrowRightFromLine,
  ArrowUpFromLine,
  ChevronLeft,
  ChevronRight,
  HelpCircle,
  LucideIcon,
  Maximize,
  Move,
  MoveHorizontal,
  MoveVertical,
  Upload,
} from "lucide-react"
import { Button, ImageUploadButton } from "./ui/button"
import useHotKey from "@/hooks/useHotkey"
import { Slider } from "./ui/slider"
import { useImage } from "@/hooks/useImage"
import {
  EXTENDER_ALL,
  EXTENDER_BUILTIN_ALL,
  EXTENDER_BUILTIN_X_LEFT,
  EXTENDER_BUILTIN_X_RIGHT,
  EXTENDER_BUILTIN_Y_BOTTOM,
  EXTENDER_BUILTIN_Y_TOP,
  EXTENDER_X,
  EXTENDER_Y,
  INSTRUCT_PIX2PIX,
  PAINT_BY_EXAMPLE,
} from "@/lib/const"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs"
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip"

const RowContainer = ({ children }: { children: React.ReactNode }) => (
  <div className="flex justify-between items-center pr-2">{children}</div>
)

const ExtenderButton = ({
  IconCls,
  text,
  onClick,
}: {
  IconCls: LucideIcon
  text: string
  onClick: () => void
}) => {
  const [showExtender] = useStore((state) => [state.settings.showExtender])
  return (
    <Button
      variant="outline"
      size="sm"
      className="p-1"
      disabled={!showExtender}
      onClick={onClick}
    >
      <div className="flex items-center gap-1">
        <IconCls size={15} strokeWidth={1} />
        {text}
      </div>
    </Button>
  )
}

const LabelTitle = ({
  text,
  toolTip,
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
      {toolTip ? (
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

const SidePanel = () => {
  const [
    settings,
    windowSize,
    paintByExampleFile,
    isProcessing,
    updateSettings,
    showSidePanel,
    runInpainting,
    updateAppState,
    updateExtenderByBuiltIn,
    updateExtenderDirection,
  ] = useStore((state) => [
    state.settings,
    state.windowSize,
    state.paintByExampleFile,
    state.getIsProcessing(),
    state.updateSettings,
    state.showSidePanel(),
    state.runInpainting,
    state.updateAppState,
    state.updateExtenderByBuiltIn,
    state.updateExtenderDirection,
  ])
  const [exampleImage, isExampleImageLoaded] = useImage(paintByExampleFile)
  const [open, toggleOpen] = useToggle(true)

  useHotKey("c", () => {
    toggleOpen()
  })

  if (!showSidePanel) {
    return null
  }

  const onKeyUp = (e: React.KeyboardEvent) => {
    // negativePrompt 回车触发 inpainting
    if (e.key === "Enter" && e.ctrlKey && settings.prompt.length !== 0) {
      runInpainting()
    }
  }

  const renderConterNetSetting = () => {
    if (!settings.model.support_controlnet) {
      return null
    }

    return (
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-center pr-2">
            <LabelTitle
              text="ControlNet"
              toolTip="Using an additional conditioning image to control how an image is generated"
              url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#controlnet"
            />
            <Switch
              id="controlnet"
              checked={settings.enableControlnet}
              onCheckedChange={(value) => {
                updateSettings({ enableControlnet: value })
              }}
            />
          </div>

          <div className="flex flex-col gap-1">
            <RowContainer>
              <Slider
                className="w-[180px]"
                defaultValue={[100]}
                min={1}
                max={100}
                step={1}
                disabled={!settings.enableControlnet}
                value={[Math.floor(settings.controlnetConditioningScale * 100)]}
                onValueChange={(vals) =>
                  updateSettings({ controlnetConditioningScale: vals[0] / 100 })
                }
              />
              <NumberInput
                id="controlnet-weight"
                className="w-[60px] rounded-full"
                disabled={!settings.enableControlnet}
                numberValue={settings.controlnetConditioningScale}
                allowFloat={false}
                onNumberValueChange={(val) => {
                  updateSettings({ controlnetConditioningScale: val })
                }}
              />
            </RowContainer>
          </div>

          <div className="pr-2">
            <Select
              value={settings.controlnetMethod}
              onValueChange={(value) => {
                updateSettings({ controlnetMethod: value })
              }}
              disabled={!settings.enableControlnet}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select control method" />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectGroup>
                  {Object.values(settings.model.controlnets).map((method) => (
                    <SelectItem key={method} value={method}>
                      {method.split("/")[1]}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </div>
        <Separator />
      </div>
    )
  }

  const renderLCMLora = () => {
    if (!settings.model.support_lcm_lora) {
      return null
    }

    return (
      <>
        <RowContainer>
          <LabelTitle
            text="LCM Lora"
            url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora"
            toolTip="Enable quality image generation in typically 2-4 steps. Suggest disabling guidance_scale by setting it to 0. You can also try values between 1.0 and 2.0. When LCM Lora is enabled, LCMSampler will be used automatically."
          />
          <Switch
            id="lcm-lora"
            checked={settings.enableLCMLora}
            onCheckedChange={(value) => {
              updateSettings({ enableLCMLora: value })
            }}
          />
        </RowContainer>
        <Separator />
      </>
    )
  }

  const renderFreeu = () => {
    if (!settings.model.support_freeu) {
      return null
    }

    return (
      <div className="flex flex-col gap-4">
        <div className="flex justify-between items-center pr-2">
          <LabelTitle
            text="FreeU"
            toolTip="FreeU is a technique for improving image quality. Different models may require different FreeU-specific hyperparameters, which can be viewed in the more info section."
            url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu"
          />
          <Switch
            id="freeu"
            checked={settings.enableFreeu}
            onCheckedChange={(value) => {
              updateSettings({ enableFreeu: value })
            }}
          />
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex justify-center gap-6">
            <div className="flex gap-2 items-center justify-center">
              <LabelTitle
                htmlFor="freeu-s1"
                text="s1"
                disabled={!settings.enableFreeu}
              />
              <NumberInput
                id="freeu-s1"
                className="w-14"
                disabled={!settings.enableFreeu}
                numberValue={settings.freeuConfig.s1}
                allowFloat
                onNumberValueChange={(value) => {
                  updateSettings({
                    freeuConfig: { ...settings.freeuConfig, s1: value },
                  })
                }}
              />
            </div>
            <div className="flex gap-2 items-center justify-center">
              <LabelTitle
                htmlFor="freeu-s2"
                text="s2"
                disabled={!settings.enableFreeu}
              />
              <NumberInput
                id="freeu-s2"
                className="w-14"
                disabled={!settings.enableFreeu}
                numberValue={settings.freeuConfig.s2}
                allowFloat
                onNumberValueChange={(value) => {
                  updateSettings({
                    freeuConfig: { ...settings.freeuConfig, s2: value },
                  })
                }}
              />
            </div>
          </div>

          <div className="flex justify-center gap-6">
            <div className="flex gap-2 items-center justify-center">
              <LabelTitle
                htmlFor="freeu-b1"
                text="b1"
                disabled={!settings.enableFreeu}
              />
              <NumberInput
                id="freeu-b1"
                className="w-14"
                disabled={!settings.enableFreeu}
                numberValue={settings.freeuConfig.b1}
                allowFloat
                onNumberValueChange={(value) => {
                  updateSettings({
                    freeuConfig: { ...settings.freeuConfig, b1: value },
                  })
                }}
              />
            </div>
            <div className="flex gap-2 items-center justify-center">
              <LabelTitle
                htmlFor="freeu-b2"
                text="b2"
                disabled={!settings.enableFreeu}
              />
              <NumberInput
                id="freeu-b2"
                className="w-14"
                disabled={!settings.enableFreeu}
                numberValue={settings.freeuConfig.b2}
                allowFloat
                onNumberValueChange={(value) => {
                  updateSettings({
                    freeuConfig: { ...settings.freeuConfig, b2: value },
                  })
                }}
              />
            </div>
          </div>
        </div>
        <Separator />
      </div>
    )
  }

  const renderNegativePrompt = () => {
    if (!settings.model.need_prompt) {
      return null
    }

    return (
      <div className="flex flex-col gap-4">
        <LabelTitle
          text="Negative prompt"
          url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#negative-prompt"
          toolTip="Negative prompt guides the model away from generating certain things in an image"
        />
        <div className="pl-2 pr-4">
          <Textarea
            rows={4}
            onKeyUp={onKeyUp}
            className="max-h-[8rem] overflow-y-auto mb-2"
            placeholder=""
            id="negative-prompt"
            value={settings.negativePrompt}
            onInput={(evt: FormEvent<HTMLTextAreaElement>) => {
              evt.preventDefault()
              evt.stopPropagation()
              const target = evt.target as HTMLTextAreaElement
              updateSettings({ negativePrompt: target.value })
            }}
          />
        </div>
      </div>
    )
  }

  const renderPaintByExample = () => {
    if (settings.model.name !== PAINT_BY_EXAMPLE) {
      return null
    }

    return (
      <div>
        <RowContainer>
          <LabelTitle
            text="Example Image"
            toolTip="An example image to guide image generation."
          />
          <ImageUploadButton
            tooltip="Upload example image"
            onFileUpload={(file) => {
              updateAppState({ paintByExampleFile: file })
            }}
          >
            <Upload />
          </ImageUploadButton>
        </RowContainer>
        {isExampleImageLoaded ? (
          <div className="flex justify-center items-center">
            <img
              src={exampleImage.src}
              alt="example"
              className="max-w-[200px] max-h-[200px] m-3"
            />
          </div>
        ) : (
          <></>
        )}
        <Button
          variant="default"
          className="w-full"
          disabled={isProcessing || !isExampleImageLoaded}
          onClick={() => {
            runInpainting()
          }}
        >
          Paint
        </Button>
      </div>
    )
  }

  const renderP2PImageGuidanceScale = () => {
    if (settings.model.name !== INSTRUCT_PIX2PIX) {
      return null
    }
    return (
      <div className="flex flex-col gap-1">
        <LabelTitle
          text="Image guidance scale"
          toolTip="Push the generated image towards the inital image. Higher image guidance scale encourages generated images that are closely linked to the source image, usually at the expense of lower image quality."
          url="https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix"
        />
        <RowContainer>
          <Slider
            className="w-[180px]"
            defaultValue={[150]}
            min={100}
            max={1000}
            step={1}
            value={[Math.floor(settings.p2pImageGuidanceScale * 100)]}
            onValueChange={(vals) =>
              updateSettings({ p2pImageGuidanceScale: vals[0] / 100 })
            }
          />
          <NumberInput
            id="image-guidance-scale"
            className="w-[60px] rounded-full"
            numberValue={settings.p2pImageGuidanceScale}
            allowFloat
            onNumberValueChange={(val) => {
              updateSettings({ p2pImageGuidanceScale: val })
            }}
          />
        </RowContainer>
      </div>
    )
  }

  const renderStrength = () => {
    if (!settings.model.support_strength) {
      return null
    }

    return (
      <div className="flex flex-col gap-1">
        <LabelTitle
          text="Strength"
          url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#strength"
          toolTip="Strength is a measure of how much noise is added to the base image, which influences how similar the output is to the base image. Higher value means more noise and more different from the base image"
        />
        <RowContainer>
          <Slider
            className="w-[180px]"
            defaultValue={[100]}
            min={10}
            max={100}
            step={1}
            value={[Math.floor(settings.sdStrength * 100)]}
            onValueChange={(vals) =>
              updateSettings({ sdStrength: vals[0] / 100 })
            }
          />
          <NumberInput
            id="strength"
            className="w-[60px] rounded-full"
            numberValue={settings.sdStrength}
            allowFloat
            onNumberValueChange={(val) => {
              updateSettings({ sdStrength: val })
            }}
          />
        </RowContainer>
      </div>
    )
  }

  const renderExtender = () => {
    if (!settings.model.support_outpainting) {
      return null
    }
    return (
      <>
        <div className="flex flex-col gap-4">
          <RowContainer>
            <LabelTitle
              text="Extender"
              toolTip="Perform outpainting on images to expand it's content."
            />
            <Switch
              id="extender"
              checked={settings.showExtender}
              onCheckedChange={(value) => {
                updateSettings({ showExtender: value })
                if (value) {
                  updateSettings({ showCropper: false })
                }
              }}
            />
          </RowContainer>

          <Tabs
            defaultValue={settings.extenderDirection}
            onValueChange={(value) => updateExtenderDirection(value)}
            className="flex flex-col justify-center items-center"
          >
            <TabsList className="w-[140px] mb-2">
              <TabsTrigger value={EXTENDER_X} disabled={!settings.showExtender}>
                <MoveHorizontal size={20} strokeWidth={1} />
              </TabsTrigger>
              <TabsTrigger value={EXTENDER_Y} disabled={!settings.showExtender}>
                <MoveVertical size={20} strokeWidth={1} />
              </TabsTrigger>
              <TabsTrigger
                value={EXTENDER_ALL}
                disabled={!settings.showExtender}
              >
                <Move size={20} strokeWidth={1} />
              </TabsTrigger>
            </TabsList>

            <TabsContent
              value={EXTENDER_X}
              className="flex gap-2 justify-center mt-0"
            >
              <ExtenderButton
                IconCls={ArrowLeftFromLine}
                text="1.5x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_X_LEFT, 1.5)
                }
              />
              <ExtenderButton
                IconCls={ArrowLeftFromLine}
                text="2.0x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_X_LEFT, 2.0)
                }
              />
              <ExtenderButton
                IconCls={ArrowRightFromLine}
                text="1.5x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_X_RIGHT, 1.5)
                }
              />
              <ExtenderButton
                IconCls={ArrowRightFromLine}
                text="2.0x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_X_RIGHT, 2.0)
                }
              />
            </TabsContent>
            <TabsContent
              value={EXTENDER_Y}
              className="flex gap-2 justify-center mt-0"
            >
              <ExtenderButton
                IconCls={ArrowUpFromLine}
                text="1.5x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_Y_TOP, 1.5)
                }
              />
              <ExtenderButton
                IconCls={ArrowUpFromLine}
                text="2.0x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_Y_TOP, 2.0)
                }
              />
              <ExtenderButton
                IconCls={ArrowDownFromLine}
                text="1.5x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_Y_BOTTOM, 1.5)
                }
              />
              <ExtenderButton
                IconCls={ArrowDownFromLine}
                text="2.0x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_Y_BOTTOM, 2.0)
                }
              />
            </TabsContent>
            <TabsContent
              value={EXTENDER_ALL}
              className="flex gap-2 justify-center mt-0"
            >
              <ExtenderButton
                IconCls={Maximize}
                text="1.25x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_ALL, 1.25)
                }
              />
              <ExtenderButton
                IconCls={Maximize}
                text="1.5x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_ALL, 1.5)
                }
              />
              <ExtenderButton
                IconCls={Maximize}
                text="1.75x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_ALL, 1.75)
                }
              />
              <ExtenderButton
                IconCls={Maximize}
                text="2.0x"
                onClick={() =>
                  updateExtenderByBuiltIn(EXTENDER_BUILTIN_ALL, 2.0)
                }
              />
            </TabsContent>
          </Tabs>
        </div>
        <Separator />
      </>
    )
  }

  return (
    <Sheet open={open} modal={false}>
      <SheetTrigger
        tabIndex={-1}
        className="z-10 outline-none absolute top-[68px] right-6 rounded-lg border bg-background"
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
        className="w-[300px] mt-[60px] outline-none pl-4 pr-1"
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
        <ScrollArea
          style={{ height: windowSize.height - 160 }}
          className="pr-3"
        >
          <div className="flex flex-col gap-4 mt-4">
            <RowContainer>
              <LabelTitle
                text="Cropper"
                toolTip="Inpainting on part of image, improve inference speed and reduce memory usage."
              />
              <Switch
                id="cropper"
                checked={settings.showCropper}
                onCheckedChange={(value) => {
                  updateSettings({ showCropper: value })
                  if (value) {
                    updateSettings({ showExtender: false })
                  }
                }}
              />
            </RowContainer>

            {renderExtender()}

            <div className="flex flex-col gap-1">
              <LabelTitle
                htmlFor="steps"
                text="Steps"
                toolTip="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."
              />
              <RowContainer>
                <Slider
                  className="w-[180px]"
                  defaultValue={[30]}
                  min={1}
                  max={100}
                  step={1}
                  value={[Math.floor(settings.sdSteps)]}
                  onValueChange={(vals) => updateSettings({ sdSteps: vals[0] })}
                />
                <NumberInput
                  id="steps"
                  className="w-[60px] rounded-full"
                  numberValue={settings.sdSteps}
                  allowFloat={false}
                  onNumberValueChange={(val) => {
                    updateSettings({ sdSteps: val })
                  }}
                />
              </RowContainer>
            </div>

            <div className="flex flex-col gap-1">
              <LabelTitle
                text="Guidance scale"
                url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#guidance-scale"
                toolTip="Guidance scale affects how aligned the text prompt and generated image are. Higher value means the prompt and generated image are closely aligned, so the output is a stricter interpretation of the prompt"
              />
              <RowContainer>
                <Slider
                  className="w-[180px]"
                  defaultValue={[750]}
                  min={0}
                  max={1500}
                  step={1}
                  value={[Math.floor(settings.sdGuidanceScale * 100)]}
                  onValueChange={(vals) =>
                    updateSettings({ sdGuidanceScale: vals[0] / 100 })
                  }
                />
                <NumberInput
                  id="guidance-scale"
                  className="w-[60px] rounded-full"
                  numberValue={settings.sdGuidanceScale}
                  allowFloat
                  onNumberValueChange={(val) => {
                    updateSettings({ sdGuidanceScale: val })
                  }}
                />
              </RowContainer>
            </div>

            {renderP2PImageGuidanceScale()}
            {renderStrength()}

            <RowContainer>
              <LabelTitle text="Sampler" />
              <Select
                value={settings.sdSampler as string}
                onValueChange={(value) => {
                  const sampler = value as SDSampler
                  updateSettings({ sdSampler: sampler })
                }}
              >
                <SelectTrigger className="w-[100px]">
                  <SelectValue placeholder="Select sampler" />
                </SelectTrigger>
                <SelectContent align="end">
                  <SelectGroup>
                    {Object.values(SDSampler).map((sampler) => (
                      <SelectItem
                        key={sampler as string}
                        value={sampler as string}
                      >
                        {sampler as string}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </RowContainer>

            <RowContainer>
              {/* 每次会从服务器返回更新该值 */}
              <LabelTitle
                text="Seed"
                toolTip="Using same parameters and a fixed seed can generate same result image."
              />
              {/* <Pin /> */}
              <div className="flex gap-2 justify-center items-center">
                <Switch
                  id="seed"
                  checked={settings.seedFixed}
                  onCheckedChange={(value) => {
                    updateSettings({ seedFixed: value })
                  }}
                />
                <NumberInput
                  id="seed"
                  className="w-[100px]"
                  disabled={!settings.seedFixed}
                  numberValue={settings.seed}
                  allowFloat={false}
                  onNumberValueChange={(val) => {
                    updateSettings({ seed: val })
                  }}
                />
              </div>
            </RowContainer>

            {renderNegativePrompt()}

            <Separator />

            {renderConterNetSetting()}
            {renderFreeu()}
            {renderLCMLora()}

            <div className="flex flex-col gap-1">
              <LabelTitle
                text="Mask blur"
                toolTip="How much to blur the mask before processing, in pixels."
              />
              <RowContainer>
                <Slider
                  className="w-[180px]"
                  defaultValue={[5]}
                  min={0}
                  max={35}
                  step={1}
                  value={[Math.floor(settings.sdMaskBlur)]}
                  onValueChange={(vals) =>
                    updateSettings({ sdMaskBlur: vals[0] })
                  }
                />
                <NumberInput
                  id="mask-blur"
                  className="w-[60px] rounded-full"
                  numberValue={settings.sdMaskBlur}
                  allowFloat={false}
                  onNumberValueChange={(value) => {
                    updateSettings({ sdMaskBlur: value })
                  }}
                />
              </RowContainer>
            </div>

            <RowContainer>
              <LabelTitle
                text="Match histograms"
                toolTip="Match the inpainting result histogram to the source image histogram"
                url="https://github.com/Sanster/lama-cleaner/pull/143#issuecomment-1325859307"
              />
              <Switch
                id="match-histograms"
                checked={settings.sdMatchHistograms}
                onCheckedChange={(value) => {
                  updateSettings({ sdMatchHistograms: value })
                }}
              />
            </RowContainer>

            <Separator />

            {renderPaintByExample()}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}

export default SidePanel

import { FormEvent } from "react"
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
import { ChevronLeft, ChevronRight, Upload } from "lucide-react"
import { Button, ImageUploadButton } from "./ui/button"
import useHotKey from "@/hooks/useHotkey"
import { Slider } from "./ui/slider"
import { useImage } from "@/hooks/useImage"
import { INSTRUCT_PIX2PIX, PAINT_BY_EXAMPLE } from "@/lib/const"

const RowContainer = ({ children }: { children: React.ReactNode }) => (
  <div className="flex justify-between items-center pr-2">{children}</div>
)

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
  ] = useStore((state) => [
    state.settings,
    state.windowSize,
    state.paintByExampleFile,
    state.getIsProcessing(),
    state.updateSettings,
    state.showSidePanel(),
    state.runInpainting,
    state.updateAppState,
  ])
  const [exampleImage, isExampleImageLoaded] = useImage(paintByExampleFile)
  const [open, toggleOpen] = useToggle(false)

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
            <Label htmlFor="controlnet">Controlnet</Label>
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
          <Label htmlFor="lcm-lora">LCM Lora</Label>
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
          <Label htmlFor="freeu">Freeu</Label>
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
              <Label htmlFor="freeu-s1" disabled={!settings.enableFreeu}>
                s1
              </Label>
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
              <Label htmlFor="freeu-s2" disabled={!settings.enableFreeu}>
                s2
              </Label>
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
              <Label htmlFor="freeu-b1" disabled={!settings.enableFreeu}>
                b1
              </Label>
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
              <Label htmlFor="freeu-b2" disabled={!settings.enableFreeu}>
                b2
              </Label>
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
        <Label htmlFor="negative-prompt">Negative prompt</Label>
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
          <div>Example Image</div>
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
        <Label htmlFor="image-guidance-scale">Image guidance scale</Label>
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
        className="w-[300px] mt-[60px] outline-none pl-4 pr-1 backdrop-filter backdrop-blur-md bg-background/70"
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
              <Label htmlFor="cropper">Cropper</Label>
              <Switch
                id="cropper"
                checked={settings.showCroper}
                onCheckedChange={(value) => {
                  updateSettings({ showCroper: value })
                }}
              />
            </RowContainer>

            <div className="flex flex-col gap-1">
              <Label htmlFor="steps">Steps</Label>
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
              <Label htmlFor="guidance-scale">Guidance scale</Label>
              <RowContainer>
                <Slider
                  className="w-[180px]"
                  defaultValue={[750]}
                  min={100}
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

            <div className="flex flex-col gap-1">
              <Label htmlFor="strength">Strength</Label>
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

            <RowContainer>
              <Label htmlFor="sampler">Sampler</Label>
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
              <Label htmlFor="seed">Seed</Label>
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
              <Label htmlFor="mask-blur">Mask blur</Label>
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
              <Label htmlFor="match-histograms">Match histograms</Label>
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

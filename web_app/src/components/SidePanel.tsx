import { FormEvent, useState } from "react"
import { useToggle, useWindowSize } from "react-use"
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
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "./ui/sheet"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "./ui/button"
import useHotKey from "@/hooks/useHotkey"
import { Slider } from "./ui/slider"

const RowContainer = ({ children }: { children: React.ReactNode }) => (
  <div className="flex justify-between items-center pr-2">{children}</div>
)

const SidePanel = () => {
  const [settings, updateSettings, showSidePanel, runInpainting] = useStore(
    (state) => [
      state.settings,
      state.updateSettings,
      state.showSidePanel(),
      state.runInpainting,
    ]
  )
  const [open, toggleOpen] = useToggle(false)

  useHotKey("c", () => {
    toggleOpen()
  })

  const windowSize = useWindowSize()

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
              checked={settings.enableControlNet}
              onCheckedChange={(value) => {
                updateSettings({ enableControlNet: value })
              }}
            />
          </div>

          <div className="pl-1 pr-2">
            <Select
              value={settings.controlnetMethod}
              onValueChange={(value) => {
                updateSettings({ controlnetMethod: value })
              }}
              disabled={!settings.enableControlNet}
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

        <RowContainer>
          <Label
            htmlFor="controlnet-weight"
            disabled={!settings.enableControlNet}
          >
            weight
          </Label>
          <NumberInput
            id="controlnet-weight"
            className="w-14"
            disabled={!settings.enableControlNet}
            numberValue={settings.controlnetConditioningScale}
            allowFloat
            onNumberValueChange={(value) => {
              updateSettings({ controlnetConditioningScale: value })
            }}
          />
        </RowContainer>

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
        <div className="flex gap-3">
          <div className="flex flex-col gap-2 items-start">
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
          <div className="flex flex-col gap-2 items-start">
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
          <div className="flex flex-col gap-2 items-start">
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
          <div className="flex flex-col gap-2 items-start">
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
        <Separator />
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
        <SheetHeader className="mb-4">
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
          <div className="flex flex-col gap-3">
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

            <RowContainer>
              <Label htmlFor="steps">Steps</Label>
              <NumberInput
                id="steps"
                className="w-14"
                numberValue={settings.sdSteps}
                allowFloat={false}
                onNumberValueChange={(value) => {
                  updateSettings({ sdSteps: value })
                }}
              />
            </RowContainer>

            <RowContainer>
              <Label htmlFor="guidance-scale">Guidance scale</Label>
              <NumberInput
                id="guidance-scale"
                className="w-14"
                numberValue={settings.sdGuidanceScale}
                allowFloat
                onNumberValueChange={(value) => {
                  updateSettings({ sdGuidanceScale: value })
                }}
              />
            </RowContainer>

            <RowContainer>
              <div className="flex gap-2 items-center">
                <Label htmlFor="strength">Strength</Label>
                <div className="text-sm">({settings.sdStrength})</div>
              </div>
              <Slider
                className="w-24"
                defaultValue={[100]}
                min={10}
                max={100}
                step={1}
                tabIndex={-1}
                value={[Math.floor(settings.sdStrength * 100)]}
                onValueChange={(vals) =>
                  updateSettings({ sdStrength: vals[0] / 100 })
                }
              />
            </RowContainer>

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
                  title="Seed"
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

            <Separator />

            <div className="flex flex-col gap-4">
              {renderConterNetSetting()}
            </div>
            {renderFreeu()}
            {renderLCMLora()}

            <RowContainer>
              <Label htmlFor="mask-blur">Mask blur</Label>
              <NumberInput
                id="mask-blur"
                className="w-14"
                numberValue={settings.sdMaskBlur}
                allowFloat={false}
                onNumberValueChange={(value) => {
                  updateSettings({ sdMaskBlur: value })
                }}
              />
            </RowContainer>

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
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}

export default SidePanel

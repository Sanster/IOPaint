import { FormEvent, useRef } from "react"
import { useStore } from "@/lib/states"
import { Switch } from "../ui/switch"
import { NumberInput } from "../ui/input"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select"
import { Textarea } from "../ui/textarea"
import { ExtenderDirection, PowerPaintTask } from "@/lib/types"
import { Separator } from "../ui/separator"
import { Button, ImageUploadButton } from "../ui/button"
import { Slider } from "../ui/slider"
import { useImage } from "@/hooks/useImage"
import {
  ANYTEXT,
  INSTRUCT_PIX2PIX,
  PAINT_BY_EXAMPLE,
  POWERPAINT,
} from "@/lib/const"
import { RowContainer, LabelTitle } from "./LabelTitle"
import { Upload } from "lucide-react"
import { useClickAway } from "react-use"

const ExtenderButton = ({
  text,
  onClick,
}: {
  text: string
  onClick: () => void
}) => {
  const [showExtender] = useStore((state) => [state.settings.showExtender])
  return (
    <Button
      variant="outline"
      className="p-1 h-8"
      disabled={!showExtender}
      onClick={onClick}
    >
      <div className="flex items-center gap-1">{text}</div>
    </Button>
  )
}

const DiffusionOptions = () => {
  const [
    samplers,
    settings,
    paintByExampleFile,
    isProcessing,
    updateSettings,
    runInpainting,
    updateAppState,
    updateExtenderByBuiltIn,
    updateExtenderDirection,
    adjustMask,
    clearMask,
    updateEnablePowerPaintV2,
    updateEnableBrushNet,
    updateEnableControlnet,
    updateLCMLora,
  ] = useStore((state) => [
    state.serverConfig.samplers,
    state.settings,
    state.paintByExampleFile,
    state.getIsProcessing(),
    state.updateSettings,
    state.runInpainting,
    state.updateAppState,
    state.updateExtenderByBuiltIn,
    state.updateExtenderDirection,
    state.adjustMask,
    state.clearMask,
    state.updateEnablePowerPaintV2,
    state.updateEnableBrushNet,
    state.updateEnableControlnet,
    state.updateLCMLora,
  ])
  const [exampleImage, isExampleImageLoaded] = useImage(paintByExampleFile)
  const negativePromptRef = useRef(null)
  useClickAway<MouseEvent>(negativePromptRef, () => {
    if (negativePromptRef?.current) {
      const input = negativePromptRef.current as HTMLInputElement
      input.blur()
    }
  })

  const onKeyUp = (e: React.KeyboardEvent) => {
    // negativePrompt 回车触发 inpainting
    if (e.key === "Enter" && e.ctrlKey && settings.prompt.length !== 0) {
      runInpainting()
    }
  }

  const renderCropper = () => {
    return (
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
    )
  }

  const renderBrushNetSetting = () => {
    if (!settings.model.support_brushnet) {
      return null
    }

    let toolTip =
      "BrushNet is a plug-and-play image inpainting model works on any SD1.5 base models."

    return (
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4">
          <RowContainer>
            <LabelTitle
              text="BrushNet"
              url="https://github.com/TencentARC/BrushNet"
              toolTip={toolTip}
            />
            <Switch
              id="brushnet"
              checked={settings.enableBrushNet}
              onCheckedChange={(value) => {
                updateEnableBrushNet(value)
              }}
            />
          </RowContainer>
          {/* <RowContainer>
            <Slider
              defaultValue={[100]}
              className="w-[180px]"
              min={1}
              max={100}
              step={1}
              disabled={!settings.enableBrushNet || disable}
              value={[Math.floor(settings.brushnetConditioningScale * 100)]}
              onValueChange={(vals) =>
                updateSettings({ brushnetConditioningScale: vals[0] / 100 })
              }
            />
            <NumberInput
              id="brushnet-weight"
              className="w-[50px] rounded-full"
              numberValue={settings.brushnetConditioningScale}
              allowFloat={false}
              onNumberValueChange={(val) => {
                updateSettings({ brushnetConditioningScale: val })
              }}
            />
          </RowContainer> */}

          <RowContainer>
            <Select
              defaultValue={settings.brushnetMethod}
              value={settings.brushnetMethod}
              onValueChange={(value) => {
                updateSettings({ brushnetMethod: value })
              }}
              disabled={!settings.enableBrushNet}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select brushnet model" />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectGroup>
                  {Object.values(settings.model.brushnets).map((method) => (
                    <SelectItem key={method} value={method}>
                      {method.split("/")[1]}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </RowContainer>
        </div>
        <Separator />
      </div>
    )
  }

  const renderConterNetSetting = () => {
    if (!settings.model.support_controlnet) {
      return null
    }

    let toolTip =
      "Using an additional conditioning image to control how an image is generated"

    return (
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4">
          <RowContainer>
            <LabelTitle
              text="ControlNet"
              url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#controlnet"
              toolTip={toolTip}
            />
            <Switch
              id="controlnet"
              checked={settings.enableControlnet}
              onCheckedChange={(value) => {
                updateEnableControlnet(value)
              }}
            />
          </RowContainer>

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
                className="w-[50px] rounded-full"
                disabled={!settings.enableControlnet}
                numberValue={settings.controlnetConditioningScale}
                allowFloat={false}
                onNumberValueChange={(val) => {
                  updateSettings({ controlnetConditioningScale: val })
                }}
              />
            </RowContainer>
          </div>

          <RowContainer>
            <Select
              defaultValue={settings.controlnetMethod}
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
          </RowContainer>
        </div>
        <Separator />
      </div>
    )
  }

  const renderLCMLora = () => {
    if (!settings.model.support_lcm_lora) {
      return null
    }

    let toolTip =
      "Enable quality image generation in typically 2-8 steps. Suggest disabling guidance_scale by setting it to 0. You can also try values between 1.0 and 2.0. When LCM Lora is enabled, LCMSampler will be used automatically."

    return (
      <>
        <RowContainer>
          <LabelTitle
            text="LCM LoRA"
            url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora"
            toolTip={toolTip}
          />
          <Switch
            id="lcm-lora"
            checked={settings.enableLCMLora}
            onCheckedChange={(value) => {
              updateLCMLora(value)
            }}
          />
        </RowContainer>
        <Separator />
      </>
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
            ref={negativePromptRef}
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
            className="w-[50px] rounded-full"
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

    let toolTip =
      "Strength is a measure of how much noise is added to the base image, which influences how similar the output is to the base image. Higher value means more noise and more different from the base image"
    // if (disable) {
    //   toolTip = "BrushNet is enabled, Strength is disabled."
    // }

    return (
      <RowContainer>
        <LabelTitle
          text="Strength"
          url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#strength"
          toolTip={toolTip}
          // disabled={disable}
        />
        <Slider
          className="w-[110px]"
          defaultValue={[100]}
          min={10}
          max={100}
          step={1}
          value={[Math.floor(settings.sdStrength * 100)]}
          onValueChange={(vals) =>
            updateSettings({ sdStrength: vals[0] / 100 })
          }
          // disabled={disable}
        />
        <NumberInput
          id="strength"
          className="w-[50px] rounded-full"
          numberValue={settings.sdStrength}
          allowFloat
          onNumberValueChange={(val) => {
            updateSettings({ sdStrength: val })
          }}
          // disabled={disable}
        />
      </RowContainer>
    )
  }

  const renderExtender = () => {
    if (!settings.model.support_outpainting) {
      return null
    }
    return (
      <>
        <div className="flex flex-col gap-2">
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

          <RowContainer>
            <Select
              defaultValue={settings.extenderDirection}
              value={settings.extenderDirection}
              onValueChange={(value) => {
                updateExtenderDirection(value as ExtenderDirection)
              }}
            >
              <SelectTrigger
                className="w-[65px] h-7"
                disabled={!settings.showExtender}
              >
                <SelectValue placeholder="Select axis" />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectGroup>
                  {Object.values(ExtenderDirection).map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>

            <div className="flex gap-1 justify-center mt-0">
              <ExtenderButton
                text="1.25x"
                onClick={() =>
                  updateExtenderByBuiltIn(settings.extenderDirection, 1.25)
                }
              />
              <ExtenderButton
                text="1.5x"
                onClick={() =>
                  updateExtenderByBuiltIn(settings.extenderDirection, 1.5)
                }
              />
              <ExtenderButton
                text="1.75x"
                onClick={() =>
                  updateExtenderByBuiltIn(settings.extenderDirection, 1.75)
                }
              />
              <ExtenderButton
                text="2.0x"
                onClick={() =>
                  updateExtenderByBuiltIn(settings.extenderDirection, 2.0)
                }
              />
            </div>
          </RowContainer>
        </div>
        <Separator />
      </>
    )
  }

  const renderPowerPaintTaskType = () => {
    return (
      <RowContainer>
        <LabelTitle
          text="Task"
          toolTip="PowerPaint task. When using extender, image-outpainting task will be auto used. For object-removal and image-outpainting, it is recommended to set the guidance_scale at 10 or above."
        />
        <Select
          defaultValue={settings.powerpaintTask}
          value={settings.powerpaintTask}
          onValueChange={(value: PowerPaintTask) => {
            updateSettings({ powerpaintTask: value })
          }}
          disabled={settings.showExtender}
        >
          <SelectTrigger className="w-[130px]">
            <SelectValue placeholder="Select task" />
          </SelectTrigger>
          <SelectContent align="end">
            <SelectGroup>
              {[
                PowerPaintTask.text_guided,
                PowerPaintTask.object_remove,
                PowerPaintTask.context_aware,
                PowerPaintTask.shape_guided,
              ].map((task) => (
                <SelectItem key={task} value={task}>
                  {task}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </RowContainer>
    )
  }

  const renderPowerPaintV1 = () => {
    if (settings.model.name !== POWERPAINT) {
      return null
    }
    return (
      <>
        {renderPowerPaintTaskType()}
        <Separator />
      </>
    )
  }

  const renderPowerPaintV2 = () => {
    if (settings.model.support_powerpaint_v2 === false) {
      return null
    }

    return (
      <>
        <RowContainer>
          <LabelTitle
            text="PowerPaint V2"
            toolTip="PowerPaint is a plug-and-play image inpainting model works on any SD1.5 base models."
          />
          <Switch
            id="powerpaint-v2"
            checked={settings.enablePowerPaintV2}
            onCheckedChange={(value) => {
              updateEnablePowerPaintV2(value)
            }}
          />
        </RowContainer>
        {renderPowerPaintTaskType()}
        <Separator />
      </>
    )
  }

  const renderSteps = () => {
    return (
      <RowContainer>
        <LabelTitle
          htmlFor="steps"
          text="Steps"
          toolTip="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."
        />
        <Slider
          className="w-[110px]"
          defaultValue={[30]}
          min={1}
          max={100}
          step={1}
          value={[Math.floor(settings.sdSteps)]}
          onValueChange={(vals) => updateSettings({ sdSteps: vals[0] })}
        />
        <NumberInput
          id="steps"
          className="w-[50px] rounded-full"
          numberValue={settings.sdSteps}
          allowFloat={false}
          onNumberValueChange={(val) => {
            updateSettings({ sdSteps: val })
          }}
        />
      </RowContainer>
    )
  }

  const renderGuidanceScale = () => {
    return (
      <RowContainer>
        <LabelTitle
          text="Guidance"
          url="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint#guidance-scale"
          toolTip="Guidance scale affects how aligned the text prompt and generated image are. Higher value means the prompt and generated image are closely aligned, so the output is a stricter interpretation of the prompt"
        />
        <Slider
          className="w-[110px]"
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
          id="guid"
          className="w-[50px] rounded-full"
          numberValue={settings.sdGuidanceScale}
          allowFloat
          onNumberValueChange={(val) => {
            updateSettings({ sdGuidanceScale: val })
          }}
        />
      </RowContainer>
    )
  }

  const renderSampler = () => {
    if (settings.model.name === ANYTEXT) {
      return null
    }

    return (
      <RowContainer>
        <LabelTitle text="Sampler" />
        <Select
          defaultValue={settings.sdSampler}
          value={settings.sdSampler}
          onValueChange={(value) => {
            updateSettings({ sdSampler: value })
          }}
        >
          <SelectTrigger className="w-[175px] text-xs">
            <SelectValue placeholder="Select sampler" />
          </SelectTrigger>
          <SelectContent align="end">
            <SelectGroup>
              {samplers.map((sampler) => (
                <SelectItem key={sampler} value={sampler} className="text-xs">
                  {sampler}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </RowContainer>
    )
  }

  const renderSeed = () => {
    return (
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
            className="w-[110px]"
            disabled={!settings.seedFixed}
            numberValue={settings.seed}
            allowFloat={false}
            onNumberValueChange={(val) => {
              updateSettings({ seed: val })
            }}
          />
        </div>
      </RowContainer>
    )
  }

  const renderMaskBlur = () => {
    return (
      <>
        <RowContainer>
          <LabelTitle
            text="Mask blur"
            toolTip="How much to blur the mask before processing, in pixels. Make the generated inpainting boundaries appear more natural."
          />
          <Slider
            className="w-[110px]"
            defaultValue={[settings.sdMaskBlur]}
            min={0}
            max={96}
            step={1}
            value={[Math.floor(settings.sdMaskBlur)]}
            onValueChange={(vals) => updateSettings({ sdMaskBlur: vals[0] })}
          />
          <NumberInput
            id="mask-blur"
            className="w-[50px] rounded-full"
            numberValue={settings.sdMaskBlur}
            allowFloat={false}
            onNumberValueChange={(value) => {
              updateSettings({ sdMaskBlur: value })
            }}
          />
        </RowContainer>
        <Separator />
      </>
    )
  }

  const renderMatchHistograms = () => {
    return (
      <>
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
      </>
    )
  }

  const renderMaskAdjuster = () => {
    return (
      <>
        <div className="flex flex-col gap-2">
          <RowContainer>
            <LabelTitle
              htmlFor="adjustMaskKernelSize"
              text="Mask OP"
              toolTip="Expand or shrink mask. Using the slider to adjust the kernel size for dilation or erosion."
            />
            <Slider
              className="w-[110px]"
              defaultValue={[12]}
              min={1}
              max={100}
              step={1}
              value={[Math.floor(settings.adjustMaskKernelSize)]}
              onValueChange={(vals) =>
                updateSettings({ adjustMaskKernelSize: vals[0] })
              }
            />
            <NumberInput
              id="adjustMaskKernelSize"
              className="w-[50px] rounded-full"
              numberValue={settings.adjustMaskKernelSize}
              allowFloat={false}
              onNumberValueChange={(val) => {
                updateSettings({ adjustMaskKernelSize: val })
              }}
            />
          </RowContainer>

          <RowContainer>
            <Button
              variant="outline"
              className="p-1 h-8"
              onClick={() => adjustMask("expand")}
              disabled={isProcessing}
            >
              <div className="flex items-center gap-1 select-none">
                {/* <Plus size={16} /> */}
                Expand
              </div>
            </Button>

            <Button
              variant="outline"
              className="p-1 h-8"
              onClick={() => adjustMask("shrink")}
              disabled={isProcessing}
            >
              <div className="flex items-center gap-1 select-none">
                {/* <Minus size={16} /> */}
                Shrink
              </div>
            </Button>

            <Button
              variant="outline"
              className="p-1 h-8"
              onClick={() => adjustMask("reverse")}
              disabled={isProcessing}
            >
              <div className="flex items-center gap-1 select-none">Reverse</div>
            </Button>

            <Button
              variant="outline"
              className="p-1 h-8 justify-self-end"
              onClick={clearMask}
              disabled={isProcessing}
            >
              <div className="flex items-center gap-1 select-none">Clear</div>
            </Button>
          </RowContainer>
        </div>
        <Separator />
      </>
    )
  }

  return (
    <div className="flex flex-col gap-[14px] mt-4">
      {renderCropper()}
      {renderExtender()}
      {renderMaskBlur()}
      {renderMaskAdjuster()}
      {renderMatchHistograms()}
      {renderPowerPaintV1()}
      {renderSteps()}
      {renderGuidanceScale()}
      {renderP2PImageGuidanceScale()}
      {renderStrength()}
      {renderSampler()}
      {renderSeed()}
      {renderNegativePrompt()}
      <Separator />
      {renderBrushNetSetting()}
      {renderPowerPaintV2()}
      {renderConterNetSetting()}
      {renderLCMLora()}
      {renderPaintByExample()}
    </div>
  )
}

export default DiffusionOptions

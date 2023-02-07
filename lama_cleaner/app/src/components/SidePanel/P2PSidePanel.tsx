import React, { FormEvent } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import * as PopoverPrimitive from '@radix-ui/react-popover'
import { useToggle } from 'react-use'
import {
  isInpaintingState,
  negativePropmtState,
  propmtState,
  settingState,
} from '../../store/Atoms'
import NumberInputSetting from '../Settings/NumberInputSetting'
import SettingBlock from '../Settings/SettingBlock'
import { Switch, SwitchThumb } from '../shared/Switch'
import TextAreaInput from '../shared/Textarea'
import emitter, { EVENT_PROMPT } from '../../event'
import ImageResizeScale from './ImageResizeScale'

const INPUT_WIDTH = 30

const P2PSidePanel = () => {
  const [open, toggleOpen] = useToggle(true)
  const [setting, setSettingState] = useRecoilState(settingState)
  const [negativePrompt, setNegativePrompt] =
    useRecoilState(negativePropmtState)
  const isInpainting = useRecoilValue(isInpaintingState)
  const prompt = useRecoilValue(propmtState)

  const handleOnInput = (evt: FormEvent<HTMLTextAreaElement>) => {
    evt.preventDefault()
    evt.stopPropagation()
    const target = evt.target as HTMLTextAreaElement
    setNegativePrompt(target.value)
  }

  const onKeyUp = (e: React.KeyboardEvent) => {
    if (
      e.key === 'Enter' &&
      (e.ctrlKey || e.metaKey) &&
      prompt.length !== 0 &&
      !isInpainting
    ) {
      emitter.emit(EVENT_PROMPT)
    }
  }

  return (
    <div className="side-panel">
      <PopoverPrimitive.Root open={open}>
        <PopoverPrimitive.Trigger
          className="btn-primary side-panel-trigger"
          onClick={() => toggleOpen()}
        >
          Config
        </PopoverPrimitive.Trigger>
        <PopoverPrimitive.Portal>
          <PopoverPrimitive.Content className="side-panel-content">
            <SettingBlock
              title="Croper"
              input={
                <Switch
                  checked={setting.showCroper}
                  onCheckedChange={value => {
                    setSettingState(old => {
                      return { ...old, showCroper: value }
                    })
                  }}
                >
                  <SwitchThumb />
                </Switch>
              }
            />

            <ImageResizeScale />

            <NumberInputSetting
              title="Steps"
              width={INPUT_WIDTH}
              value={`${setting.p2pSteps}`}
              desc="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."
              onValue={value => {
                const val = value.length === 0 ? 0 : parseInt(value, 10)
                setSettingState(old => {
                  return { ...old, p2pSteps: val }
                })
              }}
            />

            <NumberInputSetting
              title="Guidance Scale"
              width={INPUT_WIDTH}
              allowFloat
              value={`${setting.p2pGuidanceScale}`}
              desc="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality."
              onValue={value => {
                const val = value.length === 0 ? 0 : parseFloat(value)
                setSettingState(old => {
                  return { ...old, p2pGuidanceScale: val }
                })
              }}
            />

            <NumberInputSetting
              title="Image Guidance Scale"
              width={INPUT_WIDTH}
              allowFloat
              value={`${setting.p2pImageGuidanceScale}`}
              desc=""
              onValue={value => {
                const val = value.length === 0 ? 0 : parseFloat(value)
                setSettingState(old => {
                  return { ...old, p2pImageGuidanceScale: val }
                })
              }}
            />

            {/* <NumberInputSetting
              title="Mask Blur"
              width={INPUT_WIDTH}
              value={`${setting.sdMaskBlur}`}
              desc="Blur the edge of mask area. The higher the number the smoother blend with the original image"
              onValue={value => {
                const val = value.length === 0 ? 0 : parseInt(value, 10)
                setSettingState(old => {
                  return { ...old, sdMaskBlur: val }
                })
              }}
            /> */}

            {/* <SettingBlock
              title="Match Histograms"
              desc="Match the inpainting result histogram to the source image histogram, will improves the inpainting quality for some images."
              input={
                <Switch
                  checked={setting.sdMatchHistograms}
                  onCheckedChange={value => {
                    setSettingState(old => {
                      return { ...old, sdMatchHistograms: value }
                    })
                  }}
                >
                  <SwitchThumb />
                </Switch>
              }
            /> */}

            {/* <SettingBlock
              className="sub-setting-block"
              title="Sampler"
              input={
                <Selector
                  width={80}
                  value={setting.sdSampler as string}
                  options={Object.values(SDSampler)}
                  onChange={val => {
                    const sampler = val as SDSampler
                    setSettingState(old => {
                      return { ...old, sdSampler: sampler }
                    })
                  }}
                />
              }
            /> */}

            <SettingBlock
              title="Seed"
              input={
                <div
                  style={{
                    display: 'flex',
                    gap: 0,
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <NumberInputSetting
                    title=""
                    width={80}
                    value={`${setting.sdSeed}`}
                    desc=""
                    disable={!setting.sdSeedFixed}
                    onValue={value => {
                      const val = value.length === 0 ? 0 : parseInt(value, 10)
                      setSettingState(old => {
                        return { ...old, sdSeed: val }
                      })
                    }}
                  />
                  <Switch
                    checked={setting.sdSeedFixed}
                    onCheckedChange={value => {
                      setSettingState(old => {
                        return { ...old, sdSeedFixed: value }
                      })
                    }}
                    style={{ marginLeft: '8px' }}
                  >
                    <SwitchThumb />
                  </Switch>
                </div>
              }
            />

            <SettingBlock
              className="sub-setting-block"
              title="Negative prompt"
              layout="v"
              input={
                <TextAreaInput
                  className="negative-prompt"
                  value={negativePrompt}
                  onInput={handleOnInput}
                  onKeyUp={onKeyUp}
                  placeholder=""
                />
              }
            />
          </PopoverPrimitive.Content>
        </PopoverPrimitive.Portal>
      </PopoverPrimitive.Root>
    </div>
  )
}

export default P2PSidePanel

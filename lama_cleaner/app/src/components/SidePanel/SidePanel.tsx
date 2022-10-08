import React, { useState } from 'react'
import { useRecoilState } from 'recoil'
import * as PopoverPrimitive from '@radix-ui/react-popover'
import { useToggle } from 'react-use'
import { SDSampler, settingState } from '../../store/Atoms'
import NumberInputSetting from '../Settings/NumberInputSetting'
import SettingBlock from '../Settings/SettingBlock'
import Selector from '../shared/Selector'
import { Switch, SwitchThumb } from '../shared/Switch'

const INPUT_WIDTH = 30

// TODO: 添加收起来的按钮
const SidePanel = () => {
  const [open, toggleOpen] = useToggle(true)
  const [setting, setSettingState] = useRecoilState(settingState)

  return (
    <div className="side-panel">
      <PopoverPrimitive.Root open={open}>
        <PopoverPrimitive.Trigger
          className="btn-primary side-panel-trigger"
          onClick={() => toggleOpen()}
        >
          Configurations
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
            {/* 
            <NumberInputSetting
              title="Num Samples"
              width={INPUT_WIDTH}
              value={`${setting.sdNumSamples}`}
              desc=""
              onValue={value => {
                const val = value.length === 0 ? 0 : parseInt(value, 10)
                setSettingState(old => {
                  return { ...old, sdNumSamples: val }
                })
              }}
            /> */}

            <NumberInputSetting
              title="Steps"
              width={INPUT_WIDTH}
              value={`${setting.sdSteps}`}
              desc="Large steps result in better result, but more time-consuming"
              onValue={value => {
                const val = value.length === 0 ? 0 : parseInt(value, 10)
                setSettingState(old => {
                  return { ...old, sdSteps: val }
                })
              }}
            />

            <NumberInputSetting
              title="Strength"
              width={INPUT_WIDTH}
              allowFloat
              value={`${setting.sdStrength}`}
              desc="TODO"
              onValue={value => {
                const val = value.length === 0 ? 0 : parseFloat(value)
                console.log(val)
                setSettingState(old => {
                  return { ...old, sdStrength: val }
                })
              }}
            />

            <NumberInputSetting
              title="Guidance Scale"
              width={INPUT_WIDTH}
              allowFloat
              value={`${setting.sdGuidanceScale}`}
              desc="TODO"
              onValue={value => {
                const val = value.length === 0 ? 0 : parseFloat(value)
                setSettingState(old => {
                  return { ...old, sdGuidanceScale: val }
                })
              }}
            />

            <NumberInputSetting
              title="Mask Blur"
              width={INPUT_WIDTH}
              value={`${setting.sdMaskBlur}`}
              desc="TODO"
              onValue={value => {
                const val = value.length === 0 ? 0 : parseInt(value, 10)
                setSettingState(old => {
                  return { ...old, sdMaskBlur: val }
                })
              }}
            />

            <SettingBlock
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
            />

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
                  {/* 每次会从服务器返回更新该值 */}
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
          </PopoverPrimitive.Content>
        </PopoverPrimitive.Portal>
      </PopoverPrimitive.Root>
    </div>
  )
}

export default SidePanel

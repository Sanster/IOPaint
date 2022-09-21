import React, { useState } from 'react'
import { useRecoilState } from 'recoil'
import * as PopoverPrimitive from '@radix-ui/react-popover'
import { useToggle } from 'react-use'
import { SDSampler, settingState } from '../../store/Atoms'
import NumberInputSetting from '../Settings/NumberInputSetting'
import SettingBlock from '../Settings/SettingBlock'
import Selector from '../shared/Selector'
import { Switch, SwitchThumb } from '../shared/Switch'
import Button from '../shared/Button'
import emitter, { EVENT_RERUN } from '../../event'

const INPUT_WIDTH = 30

// TODO: 添加收起来的按钮
const SidePanel = () => {
  const [open, toggleOpen] = useToggle(false)
  const [setting, setSettingState] = useRecoilState(settingState)

  const onReRunBtnClick = () => {
    emitter.emit(EVENT_RERUN)
  }

  return (
    <div className="side-panel">
      <PopoverPrimitive.Root open={open}>
        <PopoverPrimitive.Trigger
          className="btn-primary side-panel-trigger"
          onClick={() => toggleOpen()}
        >
          Stable Diffusion
        </PopoverPrimitive.Trigger>
        <PopoverPrimitive.Portal>
          <PopoverPrimitive.Content className="side-panel-content">
            <SettingBlock
              title="Show Croper"
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
                  <Button
                    onClick={onReRunBtnClick}
                    icon={
                      <svg
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        width="20px"
                        height="20px"
                        viewBox="0 0 66.459 66.46"
                      >
                        <path
                          d="M65.542,11.777L33.467,0.037c-0.133-0.049-0.283-0.049-0.42,0L0.916,11.748c-0.242,0.088-0.402,0.32-0.402,0.576 l0.09,40.484c0,0.25,0.152,0.475,0.385,0.566l31.047,12.399v0.072c0,0.203,0.102,0.393,0.27,0.508 c0.168,0.111,0.379,0.135,0.57,0.062l0.385-0.154l0.385,0.154c0.072,0.028,0.15,0.045,0.227,0.045c0.121,0,0.24-0.037,0.344-0.105 c0.168-0.115,0.27-0.305,0.27-0.508v-0.072l31.047-12.399c0.232-0.093,0.385-0.316,0.385-0.568l0.027-40.453 C65.943,12.095,65.784,11.867,65.542,11.777z M32.035,63.134L3.052,51.562V15.013l28.982,11.572L32.035,63.134L32.035,63.134z M33.259,24.439L4.783,13.066l28.48-10.498l28.735,10.394L33.259,24.439z M63.465,51.562L34.484,63.134V26.585l28.981-11.572 V51.562z M14.478,38.021c0-1.692,1.35-2.528,3.016-1.867c1.665,0.663,3.016,2.573,3.016,4.269 c-0.001,1.692-1.351,2.529-3.017,1.867C15.827,41.626,14.477,39.714,14.478,38.021z M5.998,25.375c0-1.693,1.351-2.529,3.017-1.866 c1.666,0.662,3.016,2.572,3.016,4.267c0,1.695-1.351,2.529-3.017,1.867C7.347,28.979,5.998,27.069,5.998,25.375z M22.959,32.124 c0-1.694,1.351-2.53,3.017-1.867c1.666,0.663,3.016,2.573,3.016,4.267c0,1.695-1.352,2.53-3.017,1.867 C24.309,35.728,22.959,33.818,22.959,32.124z M5.995,43.103c0.001-1.692,1.351-2.529,3.017-1.867 c1.666,0.664,3.016,2.573,3.016,4.269c0,1.694-1.351,2.53-3.017,1.867C7.344,46.709,5.995,44.797,5.995,43.103z M22.957,49.853 c0.001-1.695,1.351-2.529,3.017-1.867s3.016,2.572,3.016,4.269c0,1.692-1.351,2.528-3.017,1.866 C24.306,53.458,22.957,51.546,22.957,49.853z M27.81,12.711c-0.766,1.228-3.209,2.087-5.462,1.917 c-2.253-0.169-3.46-1.301-2.695-2.528c0.765-1.227,3.207-2.085,5.461-1.916C27.365,10.352,28.573,11.484,27.81,12.711z M43.928,13.921c-0.764,1.229-3.208,2.086-5.46,1.917c-2.255-0.169-3.46-1.302-2.696-2.528c0.764-1.229,3.209-2.086,5.462-1.918 C43.485,11.563,44.693,12.695,43.928,13.921z M47.04,42.328c-1.041-1.278-0.764-3.705,0.619-5.421 c1.381-1.716,3.344-2.069,4.381-0.792c1.041,1.276,0.764,3.704-0.617,5.42S48.079,43.604,47.04,42.328z"
                          fill="currentColor"
                        />
                      </svg>
                    }
                  />

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

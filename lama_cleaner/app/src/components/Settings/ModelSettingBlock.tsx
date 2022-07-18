import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { AIModel, settingState } from '../../store/Atoms'
import Selector from '../shared/Selector'
import { Switch, SwitchThumb } from '../shared/Switch'
import Tooltip from '../shared/Tooltip'
import { LDMSampler } from './HDSettingBlock'
import NumberInputSetting from './NumberInputSetting'
import SettingBlock from './SettingBlock'

function ModelSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)

  const onModelChange = (value: AIModel) => {
    setSettingState(old => {
      return { ...old, model: value }
    })
  }

  const onLDMSamplerChange = (value: LDMSampler) => {
    setSettingState(old => {
      return { ...old, ldmSampler: value }
    })
  }

  const renderModelDesc = (
    name: string,
    paperUrl: string,
    githubUrl: string
  ) => {
    return (
      <div style={{ display: 'flex', gap: '12px' }}>
        <Tooltip content={name}>
          <a
            className="model-desc-link"
            href={paperUrl}
            target="_blank"
            rel="noreferrer noopener"
          >
            Paper
          </a>
        </Tooltip>

        <Tooltip content={githubUrl}>
          <a
            className="model-desc-link"
            href={githubUrl}
            target="_blank"
            rel="noreferrer noopener"
          >
            Code
          </a>
        </Tooltip>
      </div>
    )
  }

  const renderLDMModelDesc = () => {
    return (
      <>
        <NumberInputSetting
          title="Steps"
          value={`${setting.ldmSteps}`}
          desc="Large steps result in better result, but more time-consuming"
          onValue={value => {
            const val = value.length === 0 ? 0 : parseInt(value, 10)
            setSettingState(old => {
              return { ...old, ldmSteps: val }
            })
          }}
        />

        <SettingBlock
          className="sub-setting-block"
          title="Sampler"
          input={
            <Selector
              width={80}
              value={setting.ldmSampler as string}
              options={Object.values(LDMSampler)}
              onChange={val => onLDMSamplerChange(val as LDMSampler)}
            />
          }
        />
      </>
    )
  }

  const renderZITSModelDesc = () => {
    return (
      <div>
        <SettingBlock
          className="sub-setting-block"
          title="Wireframe"
          desc="Enable edge and line detect"
          input={
            <Switch
              checked={setting.zitsWireframe}
              onCheckedChange={checked => {
                setSettingState(old => {
                  return { ...old, zitsWireframe: checked }
                })
              }}
            >
              <SwitchThumb />
            </Switch>
          }
        />
      </div>
    )
  }

  const renderOptionDesc = (): ReactNode => {
    switch (setting.model) {
      case AIModel.LAMA:
        return undefined
      case AIModel.LDM:
        return renderLDMModelDesc()
      case AIModel.ZITS:
        return renderZITSModelDesc()
      default:
        return <></>
    }
  }

  const renderPaperCodeBadge = (): ReactNode => {
    switch (setting.model) {
      case AIModel.LAMA:
        return renderModelDesc(
          'Resolution-robust Large Mask Inpainting with Fourier Convolutions',
          'https://arxiv.org/abs/2109.07161',
          'https://github.com/saic-mdal/lama'
        )
      case AIModel.LDM:
        return renderModelDesc(
          'High-Resolution Image Synthesis with Latent Diffusion Models',
          'https://arxiv.org/abs/2112.10752',
          'https://github.com/CompVis/latent-diffusion'
        )
      case AIModel.ZITS:
        return renderModelDesc(
          'Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding',
          'https://arxiv.org/abs/2203.00867',
          'https://github.com/DQiaole/ZITS_inpainting'
        )
      default:
        return <></>
    }
  }

  return (
    <SettingBlock
      className="model-setting-block"
      title="Inpainting Model"
      titleSuffix={renderPaperCodeBadge()}
      input={
        <Selector
          width={80}
          value={setting.model as string}
          options={Object.values(AIModel)}
          onChange={val => onModelChange(val as AIModel)}
        />
      }
      optionDesc={renderOptionDesc()}
    />
  )
}

export default ModelSettingBlock

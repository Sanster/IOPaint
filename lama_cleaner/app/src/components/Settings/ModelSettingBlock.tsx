import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import Selector from '../shared/Selector'
import NumberInputSetting from './NumberInputSetting'
import SettingBlock from './SettingBlock'

export enum AIModel {
  LAMA = 'lama',
  LDM = 'ldm',
}

function ModelSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)
  console.log(setting.model)

  const onModelChange = (value: AIModel) => {
    setSettingState(old => {
      return { ...old, model: value }
    })
  }

  const renderModelDesc = (
    name: string,
    paperUrl: string,
    githubUrl: string
  ) => {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <a
          className="model-desc-link"
          href={paperUrl}
          target="_blank"
          rel="noreferrer noopener"
        >
          {name}
        </a>

        <a
          className="model-desc-link"
          href={githubUrl}
          target="_blank"
          rel="noreferrer noopener"
        >
          {githubUrl}
        </a>
      </div>
    )
  }

  const renderLDMModelDesc = () => {
    return (
      <div>
        {renderModelDesc(
          'High-Resolution Image Synthesis with Latent Diffusion Models',
          'https://arxiv.org/abs/2112.10752',
          'https://github.com/CompVis/latent-diffusion'
        )}
        <NumberInputSetting
          title="Steps"
          value={`${setting.ldmSteps}`}
          onValue={value => {
            const val = value.length === 0 ? 0 : parseInt(value, 10)
            setSettingState(old => {
              return { ...old, ldmSteps: val }
            })
          }}
        />
      </div>
    )
  }

  const renderOptionDesc = (): ReactNode => {
    switch (setting.model) {
      case AIModel.LAMA:
        return renderModelDesc(
          'Resolution-robust Large Mask Inpainting with Fourier Convolutions',
          'https://arxiv.org/abs/2109.07161',
          'https://github.com/saic-mdal/lama'
        )
      case AIModel.LDM:
        return renderLDMModelDesc()
      default:
        return <></>
    }
  }

  return (
    <SettingBlock
      className="model-setting-block"
      title="Inpainting Model"
      input={
        <Selector
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

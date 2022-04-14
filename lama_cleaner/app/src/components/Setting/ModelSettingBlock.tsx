import React, { ReactNode } from 'react'
import { useRecoilState } from 'recoil'
import { settingState } from '../../store/Atoms'
import Selector from '../shared/Selector'
import SettingBlock from './SettingBlock'

export enum AIModel {
  LAMA = 'LaMa',
  LDM = 'LDM',
}

function ModelSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)

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
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <a
          className="model-desc-link"
          href={paperUrl}
          target="_blank"
          rel="noreferrer noopener"
        >
          {name}
        </a>

        <br />

        <a
          className="model-desc-link"
          href={githubUrl}
          target="_blank"
          rel="noreferrer noopener"
          style={{ marginTop: '8px' }}
        >
          {githubUrl}
        </a>
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
        return renderModelDesc(
          'High-Resolution Image Synthesis with Latent Diffusion Models',
          'https://arxiv.org/abs/2112.10752',
          'https://github.com/CompVis/latent-diffusion'
        )
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

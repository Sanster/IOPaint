import React, { ReactNode, useEffect, useState } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import { getIsDisableModelSwitch } from '../../adapters/inpainting'
import {
  AIModel,
  CV2Flag,
  isDisableModelSwitchState,
  SDSampler,
  settingState,
} from '../../store/Atoms'
import Selector from '../shared/Selector'
import { Switch, SwitchThumb } from '../shared/Switch'
import Tooltip from '../shared/Tooltip'
import { LDMSampler } from './HDSettingBlock'
import NumberInputSetting from './NumberInputSetting'
import SettingBlock from './SettingBlock'

function ModelSettingBlock() {
  const [setting, setSettingState] = useRecoilState(settingState)
  const isDisableModelSwitch = useRecoilValue(isDisableModelSwitchState)

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
        <Tooltip content={githubUrl}>
          <a
            className="model-desc-link"
            href={githubUrl}
            target="_blank"
            rel="noreferrer noopener"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 15 15"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M7.49933 0.25C3.49635 0.25 0.25 3.49593 0.25 7.50024C0.25 10.703 2.32715 13.4206 5.2081 14.3797C5.57084 14.446 5.70302 14.2222 5.70302 14.0299C5.70302 13.8576 5.69679 13.4019 5.69323 12.797C3.67661 13.235 3.25112 11.825 3.25112 11.825C2.92132 10.9874 2.44599 10.7644 2.44599 10.7644C1.78773 10.3149 2.49584 10.3238 2.49584 10.3238C3.22353 10.375 3.60629 11.0711 3.60629 11.0711C4.25298 12.1788 5.30335 11.8588 5.71638 11.6732C5.78225 11.205 5.96962 10.8854 6.17658 10.7043C4.56675 10.5209 2.87415 9.89918 2.87415 7.12104C2.87415 6.32925 3.15677 5.68257 3.62053 5.17563C3.54576 4.99226 3.29697 4.25521 3.69174 3.25691C3.69174 3.25691 4.30015 3.06196 5.68522 3.99973C6.26337 3.83906 6.8838 3.75895 7.50022 3.75583C8.1162 3.75895 8.73619 3.83906 9.31523 3.99973C10.6994 3.06196 11.3069 3.25691 11.3069 3.25691C11.7026 4.25521 11.4538 4.99226 11.3795 5.17563C11.8441 5.68257 12.1245 6.32925 12.1245 7.12104C12.1245 9.9063 10.4292 10.5192 8.81452 10.6985C9.07444 10.9224 9.30633 11.3648 9.30633 12.0413C9.30633 13.0102 9.29742 13.7922 9.29742 14.0299C9.29742 14.2239 9.42828 14.4496 9.79591 14.3788C12.6746 13.4179 14.75 10.7025 14.75 7.50024C14.75 3.49593 11.5036 0.25 7.49933 0.25Z"
                fill="currentColor"
                fillRule="evenodd"
                clipRule="evenodd"
              />
            </svg>
          </a>
        </Tooltip>

        {/* <Tooltip content={name}>
          <a
            className="model-desc-link"
            href={paperUrl}
            target="_blank"
            rel="noreferrer noopener"
          >
            Paper
          </a>
        </Tooltip> */}
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

  const renderFCFModelDesc = () => {
    return (
      <div>
        FcF only support fixed size(512x512) image input. Lama Cleaner will take
        care of resize and crop process, it still recommended applies to small
        defects.
      </div>
    )
  }

  const renderOpenCV2Desc = () => {
    return (
      <>
        <NumberInputSetting
          title="Radius"
          value={`${setting.cv2Radius}`}
          desc="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm."
          onValue={value => {
            const val = value.length === 0 ? 0 : parseInt(value, 10)
            setSettingState(old => {
              return { ...old, cv2Radius: val }
            })
          }}
        />

        <SettingBlock
          className="sub-setting-block"
          title="Flag"
          desc="Inpainting method"
          input={
            <Selector
              width={140}
              value={setting.cv2Flag as string}
              options={Object.values(CV2Flag)}
              onChange={val => {
                setSettingState(old => {
                  return { ...old, cv2Flag: val as CV2Flag }
                })
              }}
            />
          }
        />
      </>
    )
  }

  const renderOptionDesc = (): ReactNode => {
    switch (setting.model) {
      case AIModel.LDM:
        return renderLDMModelDesc()
      case AIModel.ZITS:
        return renderZITSModelDesc()
      case AIModel.FCF:
        return renderFCFModelDesc()
      case AIModel.CV2:
        return renderOpenCV2Desc()
      default:
        return undefined
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
      case AIModel.MAT:
        return renderModelDesc(
          'Mask-Aware Transformer for Large Hole Image Inpainting',
          'https://arxiv.org/abs/2203.15270',
          'https://github.com/fenglinglwb/MAT'
        )
      case AIModel.FCF:
        return renderModelDesc(
          'Keys to Better Image Inpainting: Structure and Texture Go Hand in Hand',
          'https://arxiv.org/abs/2208.03382',
          'https://github.com/SHI-Labs/FcF-Inpainting'
        )
      case AIModel.SD15:
        return renderModelDesc(
          'Stable Diffusion 1.5',
          'https://ommer-lab.com/research/latent-diffusion-models/',
          'https://github.com/CompVis/stable-diffusion'
        )
      case AIModel.SD2:
        return renderModelDesc(
          'Stable Diffusion 2',
          'https://ommer-lab.com/research/latent-diffusion-models/',
          'https://github.com/Stability-AI/stablediffusion'
        )
      case AIModel.Mange:
        return renderModelDesc(
          'Manga Inpainting',
          'https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html',
          'https://github.com/msxie92/MangaInpainting'
        )
      case AIModel.CV2:
        return renderModelDesc(
          'OpenCV Image Inpainting',
          'https://docs.opencv.org/4.6.0/df/d3d/tutorial_py_inpainting.html',
          'https://docs.opencv.org/4.6.0/df/d3d/tutorial_py_inpainting.html'
        )
      case AIModel.PAINT_BY_EXAMPLE:
        return renderModelDesc(
          'Paint by Example',
          'https://arxiv.org/abs/2211.13227',
          'https://github.com/Fantasy-Studio/Paint-by-Example'
        )
      case AIModel.PIX2PIX:
        return renderModelDesc(
          'InstructPix2Pix',
          'https://arxiv.org/abs/2211.09800',
          'https://github.com/timothybrooks/instruct-pix2pix'
        )
      default:
        return <></>
    }
  }

  return (
    <SettingBlock
      className="model-setting-block"
      title="Model"
      titleSuffix={renderPaperCodeBadge()}
      input={
        <Selector
          value={setting.model as string}
          options={Object.values(AIModel)}
          onChange={val => onModelChange(val as AIModel)}
          disabled={isDisableModelSwitch}
        />
      }
      optionDesc={renderOptionDesc()}
    />
  )
}

export default ModelSettingBlock

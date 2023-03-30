import React, { FormEvent } from 'react'
import { useRecoilValue } from 'recoil'
import { CursorArrowRaysIcon, GifIcon } from '@heroicons/react/24/outline'
import {
  BoxModelIcon,
  ChevronRightIcon,
  FaceIcon,
  HobbyKnifeIcon,
  MixIcon,
} from '@radix-ui/react-icons'
import { useToggle } from 'react-use'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import {
  fileState,
  isProcessingState,
  serverConfigState,
} from '../../store/Atoms'
import emitter from '../../event'
import Button from '../shared/Button'

export enum PluginName {
  RemoveBG = 'RemoveBG',
  RealESRGAN = 'RealESRGAN',
  GFPGAN = 'GFPGAN',
  RestoreFormer = 'RestoreFormer',
  InteractiveSeg = 'InteractiveSeg',
  MakeGIF = 'MakeGIF',
}

const pluginMap = {
  [PluginName.RemoveBG]: {
    IconClass: HobbyKnifeIcon,
    showName: 'RemoveBG',
  },
  [PluginName.RealESRGAN]: {
    IconClass: BoxModelIcon,
    showName: 'RealESRGAN 4x',
  },
  [PluginName.GFPGAN]: {
    IconClass: FaceIcon,
    showName: 'GFPGAN',
  },
  [PluginName.RestoreFormer]: {
    IconClass: FaceIcon,
    showName: 'RestoreFormer',
  },
  [PluginName.InteractiveSeg]: {
    IconClass: CursorArrowRaysIcon,
    showName: 'Interactive Seg',
  },
  [PluginName.MakeGIF]: {
    IconClass: GifIcon,
    showName: 'Make GIF',
  },
}

const Plugins = () => {
  // const [open, toggleOpen] = useToggle(true)
  const serverConfig = useRecoilValue(serverConfigState)
  const file = useRecoilValue(fileState)
  const isProcessing = useRecoilValue(isProcessingState)
  const disabled = !file || isProcessing

  const onPluginClick = (pluginName: string) => {
    if (!disabled) {
      emitter.emit(pluginName)
    }
  }

  const onRealESRGANClick = (upscale: number) => {
    if (!disabled) {
      emitter.emit(PluginName.RealESRGAN, { upscale })
    }
  }

  const renderRealESRGANPlugin = () => {
    return (
      <DropdownMenu.Sub key="RealESRGAN">
        <DropdownMenu.SubTrigger
          className="DropdownMenuSubTrigger"
          disabled={disabled}
        >
          <BoxModelIcon />
          RealESRGAN
          <div className="RightSlot">
            <ChevronRightIcon />
          </div>
        </DropdownMenu.SubTrigger>
        <DropdownMenu.Portal>
          <DropdownMenu.SubContent className="DropdownMenuSubContent">
            <DropdownMenu.Item
              className="DropdownMenuItem"
              onClick={() => onRealESRGANClick(2)}
            >
              upscale 2x
            </DropdownMenu.Item>
            <DropdownMenu.Item
              className="DropdownMenuItem"
              onClick={() => onRealESRGANClick(4)}
              disabled={disabled}
            >
              upscale 4x
            </DropdownMenu.Item>
          </DropdownMenu.SubContent>
        </DropdownMenu.Portal>
      </DropdownMenu.Sub>
    )
  }

  const renderPlugins = () => {
    return serverConfig.plugins.map((plugin: string) => {
      const { IconClass } = pluginMap[plugin as PluginName]
      if (plugin === PluginName.RealESRGAN) {
        return renderRealESRGANPlugin()
      }
      return (
        <DropdownMenu.Item
          key={plugin}
          className="DropdownMenuItem"
          onClick={() => onPluginClick(plugin)}
          disabled={disabled}
        >
          <IconClass style={{ width: 15 }} />
          {plugin}
        </DropdownMenu.Item>
      )
    })
  }
  if (serverConfig.plugins.length === 0) {
    return null
  }

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger className="plugins">
        <Button icon={<MixIcon />} />
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="DropdownMenuContent"
          side="bottom"
          align="start"
          sideOffset={5}
          onCloseAutoFocus={event => event.preventDefault()}
        >
          {renderPlugins()}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

export default Plugins

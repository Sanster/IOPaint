import React, { FormEvent } from 'react'
import { useRecoilValue } from 'recoil'
import { CursorArrowRaysIcon, GifIcon } from '@heroicons/react/24/outline'
import { BoxModelIcon, MarginIcon, HobbyKnifeIcon } from '@radix-ui/react-icons'
import { useToggle } from 'react-use'
import * as PopoverPrimitive from '@radix-ui/react-popover'
import {
  fileState,
  isInpaintingState,
  isPluginRunningState,
  isProcessingState,
  serverConfigState,
} from '../../store/Atoms'
import emitter from '../../event'
import Button from '../shared/Button'

export enum PluginName {
  RemoveBG = 'RemoveBG',
  RealESRGAN = 'RealESRGAN',
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
  const [open, toggleOpen] = useToggle(true)
  const serverConfig = useRecoilValue(serverConfigState)
  const file = useRecoilValue(fileState)
  const isProcessing = useRecoilValue(isProcessingState)

  const onPluginClick = (pluginName: string) => {
    if (isProcessing) {
      return
    }
    emitter.emit(pluginName)
  }

  const renderPlugins = () => {
    return serverConfig.plugins.map((plugin: string) => {
      const { IconClass } = pluginMap[plugin as PluginName]
      return (
        <Button
          style={{ gap: 6 }}
          icon={<IconClass style={{ width: 15 }} />}
          onClick={() => onPluginClick(plugin)}
          disabled={!file || isProcessing}
        >
          {pluginMap[plugin as PluginName].showName}
        </Button>
      )
    })
  }

  return (
    <div className="plugins">
      <PopoverPrimitive.Root open={open}>
        <PopoverPrimitive.Trigger
          className="btn-primary plugins-trigger"
          onClick={() => toggleOpen()}
        >
          Plugins
        </PopoverPrimitive.Trigger>
        <PopoverPrimitive.Portal>
          <PopoverPrimitive.Content className="plugins-content">
            {renderPlugins()}
          </PopoverPrimitive.Content>
        </PopoverPrimitive.Portal>
      </PopoverPrimitive.Root>
    </div>
  )
}

export default Plugins

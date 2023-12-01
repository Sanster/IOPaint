import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu"
import { Button } from "./ui/button"
import { Fullscreen, MousePointerClick, Slice, Smile } from "lucide-react"
import { MixIcon } from "@radix-ui/react-icons"
import { useStore } from "@/lib/states"
import { InteractiveSeg } from "./InteractiveSeg"

export enum PluginName {
  RemoveBG = "RemoveBG",
  AnimeSeg = "AnimeSeg",
  RealESRGAN = "RealESRGAN",
  GFPGAN = "GFPGAN",
  RestoreFormer = "RestoreFormer",
  InteractiveSeg = "InteractiveSeg",
}

const pluginMap = {
  [PluginName.RemoveBG]: {
    IconClass: Slice,
    showName: "RemoveBG",
  },
  [PluginName.AnimeSeg]: {
    IconClass: Slice,
    showName: "Anime Segmentation",
  },
  [PluginName.RealESRGAN]: {
    IconClass: Fullscreen,
    showName: "RealESRGAN 4x",
  },
  [PluginName.GFPGAN]: {
    IconClass: Smile,
    showName: "GFPGAN",
  },
  [PluginName.RestoreFormer]: {
    IconClass: Smile,
    showName: "RestoreFormer",
  },
  [PluginName.InteractiveSeg]: {
    IconClass: MousePointerClick,
    showName: "Interactive Segmentation",
  },
}

const Plugins = () => {
  const [plugins, updateInteractiveSegState] = useStore((state) => [
    state.serverConfig.plugins,
    state.updateInteractiveSegState,
  ])

  if (plugins.length === 0) {
    return null
  }

  const onPluginClick = (pluginName: string) => {
    // if (!disabled) {
    //   emitter.emit(pluginName)
    // }
    if (pluginName === PluginName.InteractiveSeg) {
      updateInteractiveSegState({ isInteractiveSeg: true })
    }
  }

  const onRealESRGANClick = (upscale: number) => {
    // if (!disabled) {
    //   emitter.emit(PluginName.RealESRGAN, { upscale })
    // }
  }

  const renderRealESRGANPlugin = () => {
    return (
      <DropdownMenuSub key="RealESRGAN">
        <DropdownMenuSubTrigger>
          <div className="flex gap-2 items-center">
            <Fullscreen />
            RealESRGAN
          </div>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent>
          <DropdownMenuItem onClick={() => onRealESRGANClick(2)}>
            upscale 2x
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => onRealESRGANClick(4)}>
            upscale 4x
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    )
  }

  const renderPlugins = () => {
    return plugins.map((plugin: string) => {
      const { IconClass, showName } = pluginMap[plugin as PluginName]
      if (plugin === PluginName.RealESRGAN) {
        return renderRealESRGANPlugin()
      }
      return (
        <DropdownMenuItem key={plugin} onClick={() => onPluginClick(plugin)}>
          <div className="flex gap-2 items-center">
            <IconClass className="p-1" />
            {showName}
          </div>
        </DropdownMenuItem>
      )
    })
  }

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger
        className="border rounded-lg z-10 bg-background"
        tabIndex={-1}
      >
        <Button variant="ghost" size="icon" asChild>
          <MixIcon className="p-2" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="start">
        {renderPlugins()}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default Plugins

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
import {
  Blocks,
  Fullscreen,
  MousePointerClick,
  Slice,
  Smile,
} from "lucide-react"
import { useStore } from "@/lib/states"
import { PluginInfo } from "@/lib/types"

export enum PluginName {
  RemoveBG = "RemoveBG",
  AnimeSeg = "AnimeSeg",
  RealESRGAN = "RealESRGAN",
  GFPGAN = "GFPGAN",
  RestoreFormer = "RestoreFormer",
  InteractiveSeg = "InteractiveSeg",
}

// TODO: get plugin config from server and using form-render??
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
    showName: "RealESRGAN",
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
  const [file, plugins, updateInteractiveSegState, runRenderablePlugin] =
    useStore((state) => [
      state.file,
      state.serverConfig.plugins,
      state.updateInteractiveSegState,
      state.runRenderablePlugin,
    ])
  const disabled = !file

  if (plugins.length === 0) {
    return null
  }

  const onPluginClick = (genMask: boolean, pluginName: string) => {
    if (pluginName === PluginName.InteractiveSeg) {
      updateInteractiveSegState({ isInteractiveSeg: true })
    } else {
      runRenderablePlugin(genMask, pluginName)
    }
  }

  const renderRealESRGANPlugin = () => {
    return (
      <DropdownMenuSub key="RealESRGAN">
        <DropdownMenuSubTrigger disabled={disabled}>
          <div className="flex gap-2 items-center">
            <Fullscreen />
            RealESRGAN
          </div>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent>
          <DropdownMenuItem
            onClick={() =>
              runRenderablePlugin(false, PluginName.RealESRGAN, { upscale: 2 })
            }
          >
            upscale 2x
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() =>
              runRenderablePlugin(false, PluginName.RealESRGAN, { upscale: 4 })
            }
          >
            upscale 4x
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    )
  }

  const renderGenImageAndMaskPlugin = (plugin: PluginInfo) => {
    const { IconClass, showName } = pluginMap[plugin.name as PluginName]
    return (
      <DropdownMenuSub key={plugin.name}>
        <DropdownMenuSubTrigger disabled={disabled}>
          <div className="flex gap-2 items-center">
            <IconClass className="p-1" />
            {showName}
          </div>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent>
          <DropdownMenuItem onClick={() => onPluginClick(false, plugin.name)}>
            Remove Background
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => onPluginClick(true, plugin.name)}>
            Generate Mask
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    )
  }

  const renderPlugins = () => {
    return plugins.map((plugin: PluginInfo) => {
      const { IconClass, showName } = pluginMap[plugin.name as PluginName]
      if (plugin.name === PluginName.RealESRGAN) {
        return renderRealESRGANPlugin()
      }
      if (
        plugin.name === PluginName.RemoveBG ||
        plugin.name === PluginName.AnimeSeg
      ) {
        return renderGenImageAndMaskPlugin(plugin)
      }
      return (
        <DropdownMenuItem
          key={plugin.name}
          onClick={() => onPluginClick(false, plugin.name)}
          disabled={disabled}
        >
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
        className="border rounded-lg z-10 bg-background outline-none"
        tabIndex={-1}
      >
        <Button variant="ghost" size="icon" asChild className="p-1.5">
          <Blocks strokeWidth={1} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="start">
        {renderPlugins()}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default Plugins

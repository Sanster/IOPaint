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
  const [
    file,
    plugins,
    isPluginRunning,
    updateInteractiveSegState,
    runRenderablePlugin,
  ] = useStore((state) => [
    state.file,
    state.serverConfig.plugins,
    state.isPluginRunning,
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
          {isPluginRunning ? (
            <div role="status">
              <svg
                aria-hidden="true"
                className="w-5 h-5 animate-spin fill-primary"
                viewBox="0 0 100 101"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                  fill="currentColor"
                />
                <path
                  d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                  fill="currentFill"
                />
              </svg>
            </div>
          ) : (
            <Blocks strokeWidth={1} />
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="start">
        {renderPlugins()}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default Plugins

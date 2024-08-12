import { PlayIcon } from "@radix-ui/react-icons"
import { useState } from "react"
import { IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"
import { useImage } from "@/hooks/useImage"

import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover"
import PromptInput from "./PromptInput"
import { RotateCw, Image, Upload } from "lucide-react"
import FileManager, { MASK_TAB } from "./FileManager"
import { getMediaBlob, getMediaFile } from "@/lib/api"
import { useStore } from "@/lib/states"
import SettingsDialog from "./Settings"
import { cn, fileToImage } from "@/lib/utils"
import Coffee from "./Coffee"
import { useToast } from "./ui/use-toast"

const Header = () => {
  const [
    file,
    customMask,
    isInpainting,
    serverConfig,
    runMannually,
    enableUploadMask,
    model,
    setFile,
    setCustomFile,
    runInpainting,
    showPrevMask,
    hidePrevMask,
    imageHeight,
    imageWidth,
    handleFileManagerMaskSelect,
  ] = useStore((state) => [
    state.file,
    state.customMask,
    state.isInpainting,
    state.serverConfig,
    state.runMannually(),
    state.settings.enableUploadMask,
    state.settings.model,
    state.setFile,
    state.setCustomFile,
    state.runInpainting,
    state.showPrevMask,
    state.hidePrevMask,
    state.imageHeight,
    state.imageWidth,
    state.handleFileManagerMaskSelect,
  ])

  const { toast } = useToast()
  const [maskImage, maskImageLoaded] = useImage(customMask)
  const [openMaskPopover, setOpenMaskPopover] = useState(false)

  const handleRerunLastMask = () => {
    runInpainting()
  }

  const onRerunMouseEnter = () => {
    showPrevMask()
  }

  const onRerunMouseLeave = () => {
    hidePrevMask()
  }

  const handleOnPhotoClick = async (tab: string, filename: string) => {
    try {
      if (tab === MASK_TAB) {
        const maskBlob = await getMediaBlob(tab, filename)
        handleFileManagerMaskSelect(maskBlob)
      } else {
        const newFile = await getMediaFile(tab, filename)
        setFile(newFile)
      }
    } catch (e: any) {
      toast({
        variant: "destructive",
        description: e.message ? e.message : e.toString(),
      })
      return
    }
  }

  return (
    <header className="h-[60px] px-6 py-4 absolute top-[0] flex justify-between items-center w-full z-20 border-b backdrop-filter backdrop-blur-md bg-background/70">
      <div className="flex items-center gap-1">
        {serverConfig.enableFileManager ? (
          <FileManager photoWidth={512} onPhotoClick={handleOnPhotoClick} />
        ) : (
          <></>
        )}

        <ImageUploadButton
          disabled={isInpainting}
          tooltip="Upload image"
          onFileUpload={(file) => {
            setFile(file)
          }}
        >
          <Image />
        </ImageUploadButton>

        <div
          className={cn([
            "flex items-center gap-1",
            file && enableUploadMask ? "visible" : "hidden",
          ])}
        >
          <ImageUploadButton
            disabled={isInpainting}
            tooltip="Upload custom mask"
            onFileUpload={async (file) => {
              let newCustomMask: HTMLImageElement | null = null
              try {
                newCustomMask = await fileToImage(file)
              } catch (e: any) {
                toast({
                  variant: "destructive",
                  description: e.message ? e.message : e.toString(),
                })
                return
              }
              if (
                newCustomMask.naturalHeight !== imageHeight ||
                newCustomMask.naturalWidth !== imageWidth
              ) {
                toast({
                  variant: "destructive",
                  description: `The size of the mask must same as image: ${imageWidth}x${imageHeight}`,
                })
                return
              }

              setCustomFile(file)
              if (!runMannually) {
                runInpainting()
              }
            }}
          >
            <Upload />
          </ImageUploadButton>

          {customMask ? (
            <Popover open={openMaskPopover}>
              <PopoverTrigger
                className="btn-primary side-panel-trigger"
                onMouseEnter={() => setOpenMaskPopover(true)}
                onMouseLeave={() => setOpenMaskPopover(false)}
                style={{
                  visibility: customMask ? "visible" : "hidden",
                  outline: "none",
                }}
                onClick={() => {
                  if (customMask) {
                  }
                }}
              >
                <IconButton tooltip="Run custom mask">
                  <PlayIcon />
                </IconButton>
              </PopoverTrigger>
              <PopoverContent>
                {maskImageLoaded ? (
                  <img src={maskImage.src} alt="Custom mask" />
                ) : (
                  <></>
                )}
              </PopoverContent>
            </Popover>
          ) : (
            <></>
          )}
        </div>

        {file && !model.need_prompt ? (
          <IconButton
            disabled={isInpainting}
            tooltip="Rerun previous mask"
            onClick={handleRerunLastMask}
            onMouseEnter={onRerunMouseEnter}
            onMouseLeave={onRerunMouseLeave}
          >
            <RotateCw />
          </IconButton>
        ) : (
          <></>
        )}
      </div>

      {model.need_prompt ? <PromptInput /> : <></>}

      <div className="flex gap-1">
        <Coffee />
        <Shortcuts />
        {serverConfig.disableModelSwitch ? <></> : <SettingsDialog />}
      </div>
    </header>
  )
}

export default Header

import { PlayIcon } from "@radix-ui/react-icons"
import { useCallback, useState } from "react"
import { IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"
import emitter, {
  DREAM_BUTTON_MOUSE_ENTER,
  DREAM_BUTTON_MOUSE_LEAVE,
  EVENT_CUSTOM_MASK,
  RERUN_LAST_MASK,
} from "@/lib/event"
import { useImage } from "@/hooks/useImage"

import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover"
import PromptInput from "./PromptInput"
import { RotateCw, Image, Upload } from "lucide-react"
import FileManager from "./FileManager"
import { getMediaFile } from "@/lib/api"
import { useStore } from "@/lib/states"
import SettingsDialog from "./Settings"
import { cn } from "@/lib/utils"
import useHotKey from "@/hooks/useHotkey"
import Coffee from "./Coffee"

const Header = () => {
  const [
    file,
    customMask,
    isInpainting,
    enableFileManager,
    enableManualInpainting,
    enableUploadMask,
    model,
    setFile,
    setCustomFile,
  ] = useStore((state) => [
    state.file,
    state.customMask,
    state.isInpainting,
    state.serverConfig.enableFileManager,
    state.settings.enableManualInpainting,
    state.settings.enableUploadMask,
    state.settings.model,
    state.setFile,
    state.setCustomFile,
  ])
  const [maskImage, maskImageLoaded] = useImage(customMask)
  const [openMaskPopover, setOpenMaskPopover] = useState(false)

  const handleRerunLastMask = useCallback(() => {
    emitter.emit(RERUN_LAST_MASK)
  }, [])

  const onRerunMouseEnter = () => {
    emitter.emit(DREAM_BUTTON_MOUSE_ENTER)
  }

  const onRerunMouseLeave = () => {
    emitter.emit(DREAM_BUTTON_MOUSE_LEAVE)
  }

  useHotKey(
    "r",
    () => {
      if (!isInpainting) {
        handleRerunLastMask()
      }
    },
    {},
    [isInpainting, handleRerunLastMask]
  )

  return (
    <header className="h-[60px] px-6 py-4 absolute top-[0] flex justify-between items-center w-full z-20 border-b backdrop-filter backdrop-blur-md bg-background/70">
      <div className="flex items-center gap-1">
        {enableFileManager ? (
          <FileManager
            photoWidth={512}
            onPhotoClick={async (tab: string, filename: string) => {
              const newFile = await getMediaFile(tab, filename)
              setFile(newFile)
            }}
          />
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
            onFileUpload={(file) => {
              setCustomFile(file)
              if (!enableManualInpainting) {
                emitter.emit(EVENT_CUSTOM_MASK, { mask: file })
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
                    emitter.emit(EVENT_CUSTOM_MASK, { mask: customMask })
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
            tooltip="Rerun last mask"
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
        <SettingsDialog />
      </div>
    </header>
  )
}

export default Header

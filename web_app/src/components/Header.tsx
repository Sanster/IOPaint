import { FolderIcon, PhotoIcon } from "@heroicons/react/24/outline"
import { PlayIcon } from "@radix-ui/react-icons"
import React, { useCallback, useState } from "react"
import { useRecoilState, useRecoilValue } from "recoil"
import { useHotkeys } from "react-hotkeys-hook"
import {
  enableFileManagerState,
  fileState,
  isInpaintingState,
  isPix2PixState,
  isSDState,
  maskState,
  runManuallyState,
  showFileManagerState,
} from "@/lib/store"
import { Button, IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"
// import SettingIcon from "../Settings/SettingIcon"
// import PromptInput from "./PromptInput"
// import CoffeeIcon from '../CoffeeIcon/CoffeeIcon'
import emitter, {
  DREAM_BUTTON_MOUSE_ENTER,
  DREAM_BUTTON_MOUSE_LEAVE,
  EVENT_CUSTOM_MASK,
  RERUN_LAST_MASK,
} from "@/lib/event"
import { useImage } from "@/hooks/useImage"

import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover"
import PromptInput from "./PromptInput"
import { RotateCw } from "lucide-react"
import FileManager from "./FileManager"
import { getMediaFile } from "@/lib/api"

const Header = () => {
  const isInpainting = useRecoilValue(isInpaintingState)
  const [file, setFile] = useRecoilState(fileState)
  const [mask, setMask] = useRecoilState(maskState)
  const [maskImage, maskImageLoaded] = useImage(mask)
  const isSD = useRecoilValue(isSDState)
  const isPix2Pix = useRecoilValue(isPix2PixState)
  const runManually = useRecoilValue(runManuallyState)
  const [openMaskPopover, setOpenMaskPopover] = useState(false)
  const enableFileManager = useRecoilValue(enableFileManagerState)

  const handleRerunLastMask = useCallback(() => {
    emitter.emit(RERUN_LAST_MASK)
  }, [])

  const onRerunMouseEnter = () => {
    emitter.emit(DREAM_BUTTON_MOUSE_ENTER)
  }

  const onRerunMouseLeave = () => {
    emitter.emit(DREAM_BUTTON_MOUSE_LEAVE)
  }

  useHotkeys(
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
    <header className="h-[60px] px-6 py-4 absolute top-[0] flex justify-between items-center w-full z-20 backdrop-filter backdrop-blur-md border-b">
      <div className="flex items-center">
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
          <PhotoIcon />
        </ImageUploadButton>

        <div
          style={{
            visibility: file ? "visible" : "hidden",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <ImageUploadButton
            disabled={isInpainting}
            tooltip="Upload custom mask"
            onFileUpload={(file) => {
              setMask(file)
              console.info("Send custom mask")
              if (!runManually) {
                emitter.emit(EVENT_CUSTOM_MASK, { mask: file })
              }
            }}
          >
            <div>M</div>
          </ImageUploadButton>

          {mask ? (
            <Popover open={openMaskPopover}>
              <PopoverTrigger
                className="btn-primary side-panel-trigger"
                onMouseEnter={() => setOpenMaskPopover(true)}
                onMouseLeave={() => setOpenMaskPopover(false)}
                style={{
                  visibility: mask ? "visible" : "hidden",
                  outline: "none",
                }}
                onClick={() => {
                  if (mask) {
                    emitter.emit(EVENT_CUSTOM_MASK, { mask })
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

          <IconButton
            disabled={isInpainting}
            tooltip="Rerun last mask"
            onClick={handleRerunLastMask}
            onMouseEnter={onRerunMouseEnter}
            onMouseLeave={onRerunMouseLeave}
          >
            <RotateCw />
          </IconButton>
        </div>
      </div>

      {isSD ? <PromptInput /> : <></>}

      <div className="header-icons-wrapper">
        {/* <CoffeeIcon /> */}
        <div className="header-icons">
          <Shortcuts />
          {/* <SettingIcon /> */}
        </div>
      </div>
    </header>
  )
}

export default Header

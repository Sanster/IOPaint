import { useStore } from "@/lib/states"
import { ExtenderDirection } from "@/lib/types"
import { cn } from "@/lib/utils"
import React, { useEffect, useState } from "react"
import { twMerge } from "tailwind-merge"

const DOC_MOVE_OPTS = { capture: true, passive: false }

const DRAG_HANDLE_BORDER = 2

interface EVData {
  initX: number
  initY: number
  initHeight: number
  initWidth: number
  startResizeX: number
  startResizeY: number
  ord: string // top/right/bottom/left
}

interface Props {
  scale: number
  minHeight: number
  minWidth: number
  show: boolean
}

const clamp = (
  newPos: number,
  newLength: number,
  oldPos: number,
  minLength: number
) => {
  if (newLength < minLength) {
    if (newPos === oldPos) {
      return [newPos, minLength]
    }
    return [newPos + newLength - minLength, minLength]
  }

  return [newPos, newLength]
}

const Extender = (props: Props) => {
  const { minHeight, minWidth, scale, show } = props

  const [
    isInpainting,
    imageHeight,
    imageWdith,
    isSD,
    { x, y, width, height },
    setX,
    setY,
    setWidth,
    setHeight,
    extenderDirection,
    isResizing,
    setIsResizing,
  ] = useStore((state) => [
    state.isInpainting,
    state.imageHeight,
    state.imageWidth,
    state.isSD(),
    state.extenderState,
    state.setExtenderX,
    state.setExtenderY,
    state.setExtenderWidth,
    state.setExtenderHeight,
    state.settings.extenderDirection,
    state.isCropperExtenderResizing,
    state.setIsCropperExtenderResizing,
  ])

  const [evData, setEVData] = useState<EVData>({
    initX: 0,
    initY: 0,
    initHeight: 0,
    initWidth: 0,
    startResizeX: 0,
    startResizeY: 0,
    ord: "top",
  })

  const onDragFocus = () => {
    // console.log("focus")
  }

  const clampLeftRight = (newX: number, newWidth: number) => {
    return clamp(newX, newWidth, x, minWidth)
  }

  const clampTopBottom = (newY: number, newHeight: number) => {
    return clamp(newY, newHeight, y, minHeight)
  }

  const onPointerMove = (e: PointerEvent) => {
    if (isInpainting) {
      return
    }
    const curX = e.clientX
    const curY = e.clientY

    const offsetY = Math.round((curY - evData.startResizeY) / scale)
    const offsetX = Math.round((curX - evData.startResizeX) / scale)

    const moveTop = () => {
      const newHeight = evData.initHeight - offsetY
      const newY = evData.initY + offsetY
      let clampedY = newY
      let clampedHeight = newHeight
      if (extenderDirection === ExtenderDirection.xy) {
        if (clampedY > 0) {
          clampedY = 0
          clampedHeight = evData.initHeight - Math.abs(evData.initY)
        }
      } else {
        const clamped = clampTopBottom(newY, newHeight)
        clampedY = clamped[0]
        clampedHeight = clamped[1]
      }
      setHeight(clampedHeight)
      setY(clampedY)
    }

    const moveBottom = () => {
      const newHeight = evData.initHeight + offsetY
      let [clampedY, clampedHeight] = clampTopBottom(evData.initY, newHeight)
      if (extenderDirection === ExtenderDirection.xy) {
        if (clampedHeight < Math.abs(clampedY) + imageHeight) {
          clampedHeight = Math.abs(clampedY) + imageHeight
        }
      }
      setHeight(clampedHeight)
      setY(clampedY)
    }

    const moveLeft = () => {
      const newWidth = evData.initWidth - offsetX
      const newX = evData.initX + offsetX
      let clampedX = newX
      let clampedWidth = newWidth
      if (extenderDirection === ExtenderDirection.xy) {
        if (clampedX > 0) {
          clampedX = 0
          clampedWidth = evData.initWidth - Math.abs(evData.initX)
        }
      } else {
        const clamped = clampLeftRight(newX, newWidth)
        clampedX = clamped[0]
        clampedWidth = clamped[1]
      }
      setWidth(clampedWidth)
      setX(clampedX)
    }

    const moveRight = () => {
      const newWidth = evData.initWidth + offsetX
      let [clampedX, clampedWidth] = clampLeftRight(evData.initX, newWidth)
      if (extenderDirection === ExtenderDirection.xy) {
        if (clampedWidth < Math.abs(clampedX) + imageWdith) {
          clampedWidth = Math.abs(clampedX) + imageWdith
        }
      }
      setWidth(clampedWidth)
      setX(clampedX)
    }

    if (isResizing) {
      switch (evData.ord) {
        case "topleft": {
          moveTop()
          moveLeft()
          break
        }
        case "topright": {
          moveTop()
          moveRight()
          break
        }
        case "bottomleft": {
          moveBottom()
          moveLeft()
          break
        }
        case "bottomright": {
          moveBottom()
          moveRight()
          break
        }
        case "top": {
          moveTop()
          break
        }
        case "right": {
          moveRight()
          break
        }
        case "bottom": {
          moveBottom()
          break
        }
        case "left": {
          moveLeft()
          break
        }

        default:
          break
      }
    }
  }

  const onPointerDone = () => {
    if (isResizing) {
      setIsResizing(false)
    }
  }

  useEffect(() => {
    if (isResizing) {
      document.addEventListener("pointermove", onPointerMove, DOC_MOVE_OPTS)
      document.addEventListener("pointerup", onPointerDone, DOC_MOVE_OPTS)
      document.addEventListener("pointercancel", onPointerDone, DOC_MOVE_OPTS)
      return () => {
        document.removeEventListener(
          "pointermove",
          onPointerMove,
          DOC_MOVE_OPTS
        )
        document.removeEventListener("pointerup", onPointerDone, DOC_MOVE_OPTS)
        document.removeEventListener(
          "pointercancel",
          onPointerDone,
          DOC_MOVE_OPTS
        )
      }
    }
  }, [isResizing, width, height, evData])

  const onCropPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const { ord } = (e.target as HTMLElement).dataset
    if (ord) {
      setIsResizing(true)
      setEVData({
        initX: x,
        initY: y,
        initHeight: height,
        initWidth: width,
        startResizeX: e.clientX,
        startResizeY: e.clientY,
        ord,
      })
    }
  }

  const createDragHandle = (cursor: string, side1: string, side2: string) => {
    const sideLength = 12
    const halfSideLength = sideLength / 2
    const draghandleCls = `w-[${sideLength}px] h-[${sideLength}px] z-[4] absolute content-[''] block border-2 border-primary borde pointer-events-auto hover:bg-primary`

    let xTrans = "0"
    let yTrans = "0"

    let side2Key = side2
    let side2Val = `${-halfSideLength}px`
    if (side2 === "") {
      side2Val = "50%"
      if (side1 === "left" || side1 === "right") {
        side2Key = "top"
        yTrans = "-50%"
      } else {
        side2Key = "left"
        xTrans = "-50%"
      }
    }

    return (
      <div
        className={cn(draghandleCls, cursor)}
        style={{
          [side1]: -halfSideLength,
          [side2Key]: side2Val,
          transform: `translate(${xTrans}, ${yTrans}) scale(${1 / scale})`,
        }}
        data-ord={side1 + side2}
        aria-label={side1 + side2}
        tabIndex={-1}
        role="button"
      />
    )
  }

  const createCropSelection = () => {
    return (
      <div
        onFocus={onDragFocus}
        onPointerDown={onCropPointerDown}
        className="absolute top-0 h-full w-full"
      >
        {[ExtenderDirection.y, ExtenderDirection.xy].includes(
          extenderDirection
        ) ? (
          <>
            <div
              className="absolute pointer-events-auto top-0 left-0 w-full cursor-ns-resize h-[12px] mt-[-6px]"
              data-ord="top"
            />
            <div
              className="absolute pointer-events-auto bottom-0 left-0 w-full cursor-ns-resize h-[12px] mb-[-6px]"
              data-ord="bottom"
            />
            {createDragHandle("cursor-ns-resize", "top", "")}
            {createDragHandle("cursor-ns-resize", "bottom", "")}
          </>
        ) : (
          <></>
        )}

        {[ExtenderDirection.x, ExtenderDirection.xy].includes(
          extenderDirection
        ) ? (
          <>
            <div
              className="absolute pointer-events-auto top-0 right-0 h-full cursor-ew-resize w-[12px] mr-[-6px]"
              data-ord="right"
            />
            <div
              className="absolute pointer-events-auto top-0 left-0 h-full cursor-ew-resize w-[12px] ml-[-6px]"
              data-ord="left"
            />
            {createDragHandle("cursor-ew-resize", "left", "")}
            {createDragHandle("cursor-ew-resize", "right", "")}
          </>
        ) : (
          <></>
        )}

        {extenderDirection === ExtenderDirection.xy ? (
          <>
            {createDragHandle("cursor-nw-resize", "top", "left")}
            {createDragHandle("cursor-ne-resize", "top", "right")}
            {createDragHandle("cursor-sw-resize", "bottom", "left")}
            {createDragHandle("cursor-se-resize", "bottom", "right")}
          </>
        ) : (
          <></>
        )}
      </div>
    )
  }

  const onInfoBarPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    setEVData({
      initX: x,
      initY: y,
      initHeight: height,
      initWidth: width,
      startResizeX: e.clientX,
      startResizeY: e.clientY,
      ord: "",
    })
  }

  const createInfoBar = () => {
    return (
      <div
        className={twMerge(
          "border absolute pointer-events-auto px-2 py-1 rounded-full bg-background",
          "origin-top-left top-0 left-0"
        )}
        style={{
          transform: `scale(${(1 / scale) * 0.8})`,
        }}
        onPointerDown={onInfoBarPointerDown}
      >
        {/* TODO: 移动的时候会显示 brush */}
        {width} x {height}
      </div>
    )
  }

  const createBorder = () => {
    return (
      <div
        className={cn("outline-dashed outline-primary")}
        style={{
          height,
          width,
          outlineWidth: `${(DRAG_HANDLE_BORDER / scale) * 1.3}px`,
        }}
      />
    )
  }

  if (show === false || !isSD) {
    return null
  }

  return (
    <div className="absolute h-full w-full pointer-events-none z-[2]">
      <div
        className="relative pointer-events-none z-[2] [box-shadow:0_0_0_9999px_rgba(0,_0,_0,_0.5)]"
        style={{ height, width, left: x, top: y }}
      >
        {createBorder()}
        {createInfoBar()}
        {createCropSelection()}
      </div>
    </div>
  )
}

export default Extender

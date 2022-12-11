import React, { useEffect, useState } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import {
  croperHeight,
  croperWidth,
  croperX,
  croperY,
  isInpaintingState,
} from '../../store/Atoms'

const DOC_MOVE_OPTS = { capture: true, passive: false }

const DRAG_HANDLE_BORDER = 2
const DRAG_HANDLE_SHORT = 12
const DRAG_HANDLE_LONG = 40

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
  maxHeight: number
  maxWidth: number
  scale: number
  minHeight: number
  minWidth: number
  show: boolean
}

const clamp = (
  newPos: number,
  newLength: number,
  oldPos: number,
  oldLength: number,
  minLength: number,
  maxLength: number
) => {
  if (newPos !== oldPos && newLength === oldLength) {
    if (newPos < 0) {
      return [0, oldLength]
    }
    if (newPos + newLength > maxLength) {
      return [maxLength - oldLength, oldLength]
    }
  } else {
    if (newLength < minLength) {
      if (newPos === oldPos) {
        return [newPos, minLength]
      }
      return [newPos + newLength - minLength, minLength]
    }
    if (newPos < 0) {
      return [0, newPos + newLength]
    }
    if (newPos + newLength > maxLength) {
      return [newPos, maxLength - newPos]
    }
  }

  return [newPos, newLength]
}

const Croper = (props: Props) => {
  const { minHeight, minWidth, maxHeight, maxWidth, scale, show } = props
  const [x, setX] = useRecoilState(croperX)
  const [y, setY] = useRecoilState(croperY)
  const [height, setHeight] = useRecoilState(croperHeight)
  const [width, setWidth] = useRecoilState(croperWidth)
  const isInpainting = useRecoilValue(isInpaintingState)

  const [isResizing, setIsResizing] = useState(false)
  const [isMoving, setIsMoving] = useState(false)

  useEffect(() => {
    setX(Math.round((maxWidth - 512) / 2))
    setY(Math.round((maxHeight - 512) / 2))
  }, [maxHeight, maxWidth])

  const [evData, setEVData] = useState<EVData>({
    initX: 0,
    initY: 0,
    initHeight: 0,
    initWidth: 0,
    startResizeX: 0,
    startResizeY: 0,
    ord: 'top',
  })

  const onDragFocus = () => {
    console.log('focus')
  }

  const clampLeftRight = (newX: number, newWidth: number) => {
    return clamp(newX, newWidth, x, width, minWidth, maxWidth)
  }

  const clampTopBottom = (newY: number, newHeight: number) => {
    return clamp(newY, newHeight, y, height, minHeight, maxHeight)
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
      const [clampedY, clampedHeight] = clampTopBottom(newY, newHeight)
      setHeight(clampedHeight)
      setY(clampedY)
    }

    const moveBottom = () => {
      const newHeight = evData.initHeight + offsetY
      const [clampedY, clampedHeight] = clampTopBottom(evData.initY, newHeight)
      setHeight(clampedHeight)
      setY(clampedY)
    }

    const moveLeft = () => {
      const newWidth = evData.initWidth - offsetX
      const newX = evData.initX + offsetX
      const [clampedX, clampedWidth] = clampLeftRight(newX, newWidth)
      setWidth(clampedWidth)
      setX(clampedX)
    }

    const moveRight = () => {
      const newWidth = evData.initWidth + offsetX
      const [clampedX, clampedWidth] = clampLeftRight(evData.initX, newWidth)
      setWidth(clampedWidth)
      setX(clampedX)
    }

    if (isResizing) {
      switch (evData.ord) {
        case 'topleft': {
          moveTop()
          moveLeft()
          break
        }
        case 'topright': {
          moveTop()
          moveRight()
          break
        }
        case 'bottomleft': {
          moveBottom()
          moveLeft()
          break
        }
        case 'bottomright': {
          moveBottom()
          moveRight()
          break
        }
        case 'top': {
          moveTop()
          break
        }
        case 'right': {
          moveRight()
          break
        }
        case 'bottom': {
          moveBottom()
          break
        }
        case 'left': {
          moveLeft()
          break
        }

        default:
          break
      }
    }

    if (isMoving) {
      const newX = evData.initX + offsetX
      const newY = evData.initY + offsetY
      const [clampedX, clampedWidth] = clampLeftRight(newX, evData.initWidth)
      const [clampedY, clampedHeight] = clampTopBottom(newY, evData.initHeight)
      setWidth(clampedWidth)
      setHeight(clampedHeight)
      setX(clampedX)
      setY(clampedY)
    }
  }

  const onPointerDone = (e: PointerEvent) => {
    if (isResizing) {
      setIsResizing(false)
    }

    if (isMoving) {
      setIsMoving(false)
    }
  }

  useEffect(() => {
    if (isResizing || isMoving) {
      document.addEventListener('pointermove', onPointerMove, DOC_MOVE_OPTS)
      document.addEventListener('pointerup', onPointerDone, DOC_MOVE_OPTS)
      document.addEventListener('pointercancel', onPointerDone, DOC_MOVE_OPTS)
      return () => {
        document.removeEventListener(
          'pointermove',
          onPointerMove,
          DOC_MOVE_OPTS
        )
        document.removeEventListener('pointerup', onPointerDone, DOC_MOVE_OPTS)
        document.removeEventListener(
          'pointercancel',
          onPointerDone,
          DOC_MOVE_OPTS
        )
      }
    }
  }, [isResizing, isMoving, width, height, evData])

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

  const createCropSelection = () => {
    return (
      <div
        className="drag-elements"
        onFocus={onDragFocus}
        onPointerDown={onCropPointerDown}
      >
        <div
          className="drag-bar ord-top"
          data-ord="top"
          style={{ transform: `scale(${1 / scale})` }}
        />
        <div
          className="drag-bar ord-right"
          data-ord="right"
          style={{ transform: `scale(${1 / scale})` }}
        />
        <div
          className="drag-bar ord-bottom"
          data-ord="bottom"
          style={{ transform: `scale(${1 / scale})` }}
        />
        <div
          className="drag-bar ord-left"
          data-ord="left"
          style={{ transform: `scale(${1 / scale})` }}
        />

        <div
          className="drag-handle ord-topleft"
          data-ord="topleft"
          aria-label="topleft"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />

        <div
          className="drag-handle ord-topright"
          data-ord="topright"
          aria-label="topright"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />

        <div
          className="drag-handle ord-bottomleft"
          data-ord="bottomleft"
          aria-label="bottomleft"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />

        <div
          className="drag-handle ord-bottomright"
          data-ord="bottomright"
          aria-label="bottomright"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />

        <div
          className="drag-handle ord-top"
          data-ord="top"
          aria-label="top"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />
        <div
          className="drag-handle ord-right"
          data-ord="right"
          aria-label="right"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />
        <div
          className="drag-handle ord-bottom"
          data-ord="bottom"
          aria-label="bottom"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />
        <div
          className="drag-handle ord-left"
          data-ord="left"
          aria-label="left"
          tabIndex={0}
          role="button"
          style={{ transform: `scale(${1 / scale})` }}
        />
      </div>
    )
  }

  const onInfoBarPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsMoving(true)
    setEVData({
      initX: x,
      initY: y,
      initHeight: height,
      initWidth: width,
      startResizeX: e.clientX,
      startResizeY: e.clientY,
      ord: '',
    })
  }

  const createInfoBar = () => {
    return (
      <div
        className="info-bar"
        onPointerDown={onInfoBarPointerDown}
        style={{
          transform: `scale(${1 / scale})`,
          top: `${10 / scale}px`,
          left: `${10 / scale}px`,
        }}
      >
        <div className="crop-size">
          {width} x {height}
        </div>
      </div>
    )
  }

  const createBorder = () => {
    return (
      <div
        className="crop-border"
        style={{
          height,
          width,
          outlineWidth: `${DRAG_HANDLE_BORDER / scale}px`,
        }}
      />
    )
  }

  return (
    <div
      className="croper-wrapper"
      style={{ visibility: show ? 'visible' : 'hidden' }}
    >
      <div className="croper" style={{ height, width, left: x, top: y }}>
        {createBorder()}
        {createInfoBar()}
        {createCropSelection()}
      </div>
    </div>
  )
}

export default Croper

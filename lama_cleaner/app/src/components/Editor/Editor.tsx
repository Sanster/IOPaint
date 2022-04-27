import {
  ArrowsExpandIcon,
  DownloadIcon,
  EyeIcon,
} from '@heroicons/react/outline'
import React, {
  SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  ReactZoomPanPinchRef,
  TransformComponent,
  TransformWrapper,
} from 'react-zoom-pan-pinch'
import { useRecoilValue } from 'recoil'
import { useWindowSize, useKey, useKeyPressEvent } from 'react-use'
import inpaint from '../../adapters/inpainting'
import Button from '../shared/Button'
import Slider from './Slider'
import SizeSelector from './SizeSelector'
import {
  downloadImage,
  isMidClick,
  isRightClick,
  loadImage,
  useImage,
} from '../../utils'
import { settingState } from '../../store/Atoms'

const TOOLBAR_SIZE = 200
const BRUSH_COLOR = '#ffcc00bb'

interface EditorProps {
  file: File
}

interface Line {
  size?: number
  pts: { x: number; y: number }[]
}

type LineGroup = Array<Line>

function drawLines(
  ctx: CanvasRenderingContext2D,
  lines: LineGroup,
  color = BRUSH_COLOR
) {
  ctx.strokeStyle = color
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  lines.forEach(line => {
    if (!line?.pts.length || !line.size) {
      return
    }
    ctx.lineWidth = line.size
    ctx.beginPath()
    ctx.moveTo(line.pts[0].x, line.pts[0].y)
    line.pts.forEach(pt => ctx.lineTo(pt.x, pt.y))
    ctx.stroke()
  })
}

function mouseXY(ev: SyntheticEvent) {
  const mouseEvent = ev.nativeEvent as MouseEvent
  return { x: mouseEvent.offsetX, y: mouseEvent.offsetY }
}

export default function Editor(props: EditorProps) {
  const { file } = props
  const settings = useRecoilValue(settingState)
  const [brushSize, setBrushSize] = useState(40)
  const [original, isOriginalLoaded] = useImage(file)
  const [renders, setRenders] = useState<HTMLImageElement[]>([])
  const [context, setContext] = useState<CanvasRenderingContext2D>()
  const [maskCanvas] = useState<HTMLCanvasElement>(() => {
    return document.createElement('canvas')
  })
  const [lineGroups, setLineGroups] = useState<LineGroup[]>([])
  const [curLineGroup, setCurLineGroup] = useState<LineGroup>([])
  const [{ x, y }, setCoords] = useState({ x: -1, y: -1 })
  const [showBrush, setShowBrush] = useState(false)
  const [showRefBrush, setShowRefBrush] = useState(false)
  const [isPanning, setIsPanning] = useState<boolean>(false)
  const [showOriginal, setShowOriginal] = useState(false)
  const [isInpaintingLoading, setIsInpaintingLoading] = useState(false)
  const [scale, setScale] = useState<number>(1)
  const [panned, setPanned] = useState<boolean>(false)
  const [minScale, setMinScale] = useState<number>(1.0)
  const [sizeLimit, setSizeLimit] = useState<number>(1080)
  const windowSize = useWindowSize()
  const windowCenterX = windowSize.width / 2
  const windowCenterY = windowSize.height / 2
  const viewportRef = useRef<ReactZoomPanPinchRef | undefined | null>()
  // Indicates that the image has been loaded and is centered on first load
  const [initialCentered, setInitialCentered] = useState(false)

  const [isDraging, setIsDraging] = useState(false)
  const [isMultiStrokeKeyPressed, setIsMultiStrokeKeyPressed] = useState(false)

  const [sliderPos, setSliderPos] = useState<number>(0)

  const draw = useCallback(
    (render: HTMLImageElement, lineGroup: LineGroup) => {
      if (!context) {
        return
      }
      context.clearRect(0, 0, context.canvas.width, context.canvas.height)
      context.drawImage(
        render,
        0,
        0,
        original.naturalWidth,
        original.naturalHeight
      )
      drawLines(context, lineGroup)
    },
    [context, original]
  )

  const drawAllLinesOnMask = (_lineGroups: LineGroup[]) => {
    if (!context?.canvas.width || !context?.canvas.height) {
      throw new Error('canvas has invalid size')
    }
    maskCanvas.width = context?.canvas.width
    maskCanvas.height = context?.canvas.height
    const ctx = maskCanvas.getContext('2d')
    if (!ctx) {
      throw new Error('could not retrieve mask canvas')
    }

    _lineGroups.forEach(lineGroup => {
      drawLines(ctx, lineGroup, 'white')
    })
  }

  const runInpainting = async () => {
    if (!hadDrawSomething()) {
      return
    }

    const newLineGroups = [...lineGroups, curLineGroup]
    setLineGroups(newLineGroups)
    setCurLineGroup([])
    setIsDraging(false)
    setIsInpaintingLoading(true)

    drawAllLinesOnMask(newLineGroups)
    try {
      const res = await inpaint(
        file,
        maskCanvas.toDataURL(),
        settings,
        sizeLimit.toString()
      )
      if (!res) {
        throw new Error('empty response')
      }
      const newRender = new Image()
      await loadImage(newRender, res)
      const newRenders = [...renders, newRender]
      setRenders(newRenders)
      draw(newRender, [])
    } catch (e: any) {
      // eslint-disable-next-line
      alert(e.message ? e.message : e.toString())
    }
    setIsInpaintingLoading(false)
  }

  const hadDrawSomething = () => {
    return curLineGroup.length !== 0
  }

  const hadRunInpainting = () => {
    return renders.length !== 0
  }

  const drawOnCurrentRender = useCallback(
    (lineGroup: LineGroup) => {
      if (renders.length === 0) {
        draw(original, lineGroup)
      } else {
        draw(renders[renders.length - 1], lineGroup)
      }
    },
    [original, renders, draw]
  )

  const handleMultiStrokeKeyDown = () => {
    if (isInpaintingLoading) {
      return
    }
    setIsMultiStrokeKeyPressed(true)
  }

  const handleMultiStrokeKeyup = () => {
    if (!isMultiStrokeKeyPressed) {
      return
    }
    if (isInpaintingLoading) {
      return
    }

    setIsMultiStrokeKeyPressed(false)

    if (!settings.runInpaintingManually) {
      runInpainting()
    }
  }

  const predicate = (event: KeyboardEvent) => {
    return event.key === 'Control' || event.key === 'Meta'
  }

  useKey(predicate, handleMultiStrokeKeyup, { event: 'keyup' }, [
    isInpaintingLoading,
    isMultiStrokeKeyPressed,
    hadDrawSomething,
  ])

  useKey(
    predicate,
    handleMultiStrokeKeyDown,
    {
      event: 'keydown',
    },
    [isInpaintingLoading]
  )

  // Draw once the original image is loaded
  useEffect(() => {
    if (!isOriginalLoaded) {
      return
    }

    const rW = windowSize.width / original.naturalWidth
    const rH = (windowSize.height - TOOLBAR_SIZE) / original.naturalHeight

    let s = 1.0
    if (rW < 1 || rH < 1) {
      s = Math.min(rW, rH)
    }
    setMinScale(s)
    setScale(s)

    const imageSizeLimit = Math.max(original.width, original.height)
    setSizeLimit(imageSizeLimit)

    if (context?.canvas) {
      context.canvas.width = original.naturalWidth
      context.canvas.height = original.naturalHeight
      drawOnCurrentRender([])
    }

    if (!initialCentered) {
      viewportRef.current?.centerView(s, 1)
      setInitialCentered(true)
    }
  }, [
    context?.canvas,
    viewportRef,
    original,
    isOriginalLoaded,
    windowSize,
    initialCentered,
  ])

  // Zoom reset
  const resetZoom = useCallback(() => {
    if (!minScale || !original || !windowSize) {
      return
    }
    const viewport = viewportRef.current
    if (!viewport) {
      throw new Error('no viewport')
    }
    const offsetX = (windowSize.width - original.width * minScale) / 2
    const offsetY = (windowSize.height - original.height * minScale) / 2
    viewport.setTransform(offsetX, offsetY, minScale, 200, 'easeOutQuad')
    viewport.state.scale = minScale
    setScale(minScale)
    setPanned(false)
  }, [viewportRef, minScale, original, windowSize])

  useEffect(() => {
    window.addEventListener('resize', () => {
      resetZoom()
    })
    return () => {
      window.removeEventListener('resize', () => {
        resetZoom()
      })
    }
  }, [windowSize, resetZoom])

  const handleEscPressed = () => {
    if (isInpaintingLoading) {
      return
    }
    if (isDraging || isMultiStrokeKeyPressed) {
      setIsDraging(false)
      setCurLineGroup([])
      drawOnCurrentRender([])
    } else {
      resetZoom()
    }
  }

  useKey(
    'Escape',
    handleEscPressed,
    {
      event: 'keydown',
    },
    [
      isDraging,
      isInpaintingLoading,
      isMultiStrokeKeyPressed,
      resetZoom,
      drawOnCurrentRender,
    ]
  )

  const onMouseMove = (ev: SyntheticEvent) => {
    const mouseEvent = ev.nativeEvent as MouseEvent
    setCoords({ x: mouseEvent.pageX, y: mouseEvent.pageY })
  }

  const onMouseDrag = (ev: SyntheticEvent) => {
    if (isPanning) {
      return
    }
    if (!isDraging) {
      return
    }
    if (curLineGroup.length === 0) {
      return
    }
    const lineGroup = [...curLineGroup]
    lineGroup[lineGroup.length - 1].pts.push(mouseXY(ev))
    setCurLineGroup(lineGroup)
    drawOnCurrentRender(lineGroup)
  }

  const onPointerUp = (ev: SyntheticEvent) => {
    if (isMidClick(ev)) {
      setIsPanning(false)
    }

    if (isPanning) {
      return
    }
    if (!original.src) {
      return
    }
    const canvas = context?.canvas
    if (!canvas) {
      return
    }
    if (isInpaintingLoading) {
      return
    }
    if (!isDraging) {
      return
    }

    if (isMultiStrokeKeyPressed) {
      setIsDraging(false)
      return
    }

    if (settings.runInpaintingManually) {
      setIsDraging(false)
    } else {
      runInpainting()
    }
  }

  const onMouseDown = (ev: SyntheticEvent) => {
    if (isPanning) {
      return
    }
    if (!original.src) {
      return
    }
    const canvas = context?.canvas
    if (!canvas) {
      return
    }
    if (isInpaintingLoading) {
      return
    }

    if (isRightClick(ev)) {
      return
    }

    if (isMidClick(ev)) {
      setIsPanning(true)
      return
    }

    setIsDraging(true)

    let lineGroup: LineGroup = []
    if (isMultiStrokeKeyPressed || settings.runInpaintingManually) {
      lineGroup = [...curLineGroup]
    }
    lineGroup.push({ size: brushSize, pts: [mouseXY(ev)] })
    setCurLineGroup(lineGroup)
    drawOnCurrentRender(lineGroup)
  }

  const undoStroke = useCallback(() => {
    if (curLineGroup.length === 0) {
      return
    }
    const newLineGroup = curLineGroup.slice(0, curLineGroup.length - 1)
    setCurLineGroup(newLineGroup)
    drawOnCurrentRender(newLineGroup)
  }, [curLineGroup, drawOnCurrentRender])

  const undoRender = useCallback(() => {
    if (!renders.length) {
      return
    }

    const groups = lineGroups.slice(0, lineGroups.length - 1)
    setLineGroups(groups)
    setCurLineGroup([])
    setIsDraging(false)
    const newRenders = renders.slice(0, renders.length - 1)
    setRenders(newRenders)
    if (newRenders.length === 0) {
      draw(original, [])
    } else {
      draw(newRenders[newRenders.length - 1], [])
    }
  }, [draw, renders, lineGroups, original])

  const undo = () => {
    if (settings.runInpaintingManually && curLineGroup.length !== 0) {
      undoStroke()
    } else {
      undoRender()
    }
  }

  // Handle Cmd+Z
  const undoPredicate = (event: KeyboardEvent) => {
    const isCmdZ = (event.metaKey || event.ctrlKey) && event.key === 'z'
    // Handle tab switch
    if (event.key === 'Tab') {
      event.preventDefault()
    }
    if (isCmdZ) {
      event.preventDefault()
      return true
    }
    return false
  }

  useKey(undoPredicate, undo, undefined, [undoStroke, undoRender])

  const disableUndo = () => {
    if (renders.length > 0) {
      return false
    }

    if (settings.runInpaintingManually) {
      if (curLineGroup.length === 0) {
        return true
      }
    } else if (renders.length === 0) {
      return true
    }

    return false
  }

  useKeyPressEvent(
    'Tab',
    ev => {
      ev?.preventDefault()
      ev?.stopPropagation()
      if (hadRunInpainting()) {
        setShowOriginal(() => {
          window.setTimeout(() => {
            setSliderPos(100)
          }, 10)
          return true
        })
      }
    },
    ev => {
      ev?.preventDefault()
      ev?.stopPropagation()
      if (hadRunInpainting()) {
        setSliderPos(0)
        window.setTimeout(() => {
          setShowOriginal(false)
        }, 350)
      }
    }
  )

  function download() {
    const name = file.name.replace(/(\.[\w\d_-]+)$/i, '_cleanup$1')
    const curRender = renders[renders.length - 1]
    downloadImage(curRender.currentSrc, name)
  }

  const onSizeLimitChange = (_sizeLimit: number) => {
    setSizeLimit(_sizeLimit)
  }

  const toggleShowBrush = (newState: boolean) => {
    if (newState !== showBrush && !isPanning) {
      setShowBrush(newState)
    }
  }

  const getCursor = useCallback(() => {
    if (isPanning) {
      return 'grab'
    }
    if (showBrush) {
      return 'none'
    }
    return undefined
  }, [showBrush, isPanning])

  // Standard Hotkeys for Brush Size
  useKeyPressEvent('[', () => {
    setBrushSize(currentBrushSize => {
      if (currentBrushSize > 10) {
        return currentBrushSize - 10
      }
      if (currentBrushSize <= 10 && currentBrushSize > 0) {
        return currentBrushSize - 5
      }
      return currentBrushSize
    })
  })

  useKeyPressEvent(']', () => {
    setBrushSize(currentBrushSize => {
      return currentBrushSize + 10
    })
  })

  // Toggle clean/zoom tool on spacebar.
  useKeyPressEvent(
    ' ',
    ev => {
      ev?.preventDefault()
      ev?.stopPropagation()
      setShowBrush(false)
      setIsPanning(true)
    },
    ev => {
      ev?.preventDefault()
      ev?.stopPropagation()
      setShowBrush(true)
      setIsPanning(false)
    }
  )

  const getCurScale = (): number => {
    let s = minScale
    if (viewportRef.current?.state.scale !== undefined) {
      s = viewportRef.current?.state.scale
    }
    return s!
  }

  const getBrushStyle = (_x: number, _y: number) => {
    const curScale = getCurScale()
    return {
      width: `${brushSize * curScale}px`,
      height: `${brushSize * curScale}px`,
      left: `${_x}px`,
      top: `${_y}px`,
      transform: 'translate(-50%, -50%)',
    }
  }

  const handleSliderChange = (value: number) => {
    setBrushSize(value)

    if (!showRefBrush) {
      setShowRefBrush(true)
      window.setTimeout(() => {
        setShowRefBrush(false)
      }, 10000)
    }
  }

  return (
    <div
      className="editor-container"
      aria-hidden="true"
      onMouseMove={onMouseMove}
      onMouseUp={onPointerUp}
    >
      <TransformWrapper
        ref={r => {
          if (r) {
            viewportRef.current = r
          }
        }}
        panning={{ disabled: !isPanning, velocityDisabled: true }}
        wheel={{ step: 0.05 }}
        centerZoomedOut
        alignmentAnimation={{ disabled: true }}
        // centerOnInit
        limitToBounds={false}
        doubleClick={{ disabled: true }}
        initialScale={minScale}
        minScale={minScale}
        onPanning={ref => {
          if (!panned) {
            setPanned(true)
          }
        }}
        onZoom={ref => {
          setScale(ref.state.scale)
        }}
      >
        <TransformComponent
          contentClass={isInpaintingLoading ? 'editor-canvas-loading' : ''}
          contentStyle={{
            visibility: initialCentered ? 'visible' : 'hidden',
          }}
        >
          <div className="editor-canvas-container">
            <canvas
              className="editor-canvas"
              style={{
                cursor: getCursor(),
                clipPath: `inset(0 ${sliderPos}% 0 0)`,
                transition: 'clip-path 350ms ease-in-out',
              }}
              onContextMenu={e => {
                e.preventDefault()
              }}
              onMouseOver={() => {
                toggleShowBrush(true)
                setShowRefBrush(false)
              }}
              onFocus={() => toggleShowBrush(true)}
              onMouseLeave={() => toggleShowBrush(false)}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseDrag}
              ref={r => {
                if (r && !context) {
                  const ctx = r.getContext('2d')
                  if (ctx) {
                    setContext(ctx)
                  }
                }
              }}
            />
            <div
              className="original-image-container"
              style={{
                width: `${original.naturalWidth}px`,
                height: `${original.naturalHeight}px`,
              }}
            >
              {showOriginal && (
                <div
                  className="editor-slider"
                  style={{
                    marginRight: `${sliderPos}%`,
                  }}
                />
              )}

              <img
                className="original-image"
                src={original.src}
                alt="original"
                style={{
                  width: `${original.naturalWidth}px`,
                  height: `${original.naturalHeight}px`,
                }}
              />
            </div>
          </div>
        </TransformComponent>
      </TransformWrapper>

      {showBrush && !isInpaintingLoading && !isPanning && (
        <div className="brush-shape" style={getBrushStyle(x, y)} />
      )}

      {showRefBrush && (
        <div
          className="brush-shape"
          style={getBrushStyle(windowCenterX, windowCenterY)}
        />
      )}

      <div className="editor-toolkit-panel">
        <SizeSelector
          onChange={onSizeLimitChange}
          originalWidth={original.naturalWidth}
          originalHeight={original.naturalHeight}
        />
        <Slider
          label="Brush"
          min={10}
          max={150}
          value={brushSize}
          onChange={handleSliderChange}
          onClick={() => setShowRefBrush(false)}
        />
        <div className="editor-toolkit-btns">
          <Button
            icon={<ArrowsExpandIcon />}
            disabled={scale === minScale && panned === false}
            onClick={resetZoom}
          />
          <Button
            icon={
              <svg
                width="19"
                height="9"
                viewBox="0 0 19 9"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M2 1C2 0.447715 1.55228 0 1 0C0.447715 0 0 0.447715 0 1H2ZM1 8H0V9H1V8ZM8 9C8.55228 9 9 8.55229 9 8C9 7.44771 8.55228 7 8 7V9ZM16.5963 7.42809C16.8327 7.92721 17.429 8.14016 17.9281 7.90374C18.4272 7.66731 18.6402 7.07103 18.4037 6.57191L16.5963 7.42809ZM16.9468 5.83205L17.8505 5.40396L16.9468 5.83205ZM0 1V8H2V1H0ZM1 9H8V7H1V9ZM1.66896 8.74329L6.66896 4.24329L5.33104 2.75671L0.331035 7.25671L1.66896 8.74329ZM16.043 6.26014L16.5963 7.42809L18.4037 6.57191L17.8505 5.40396L16.043 6.26014ZM6.65079 4.25926C9.67554 1.66661 14.3376 2.65979 16.043 6.26014L17.8505 5.40396C15.5805 0.61182 9.37523 -0.710131 5.34921 2.74074L6.65079 4.25926Z"
                  fill="currentColor"
                />
              </svg>
            }
            onClick={undo}
            disabled={disableUndo()}
          />
          <Button
            icon={<EyeIcon />}
            className={showOriginal ? 'eyeicon-active' : ''}
            onDown={ev => {
              ev.preventDefault()
              setShowOriginal(() => {
                window.setTimeout(() => {
                  setSliderPos(100)
                }, 10)
                return true
              })
            }}
            onUp={() => {
              setSliderPos(0)
              window.setTimeout(() => {
                setShowOriginal(false)
              }, 350)
            }}
            disabled={renders.length === 0}
          />
          <Button
            icon={<DownloadIcon />}
            disabled={!renders.length}
            onClick={download}
          />

          {settings.runInpaintingManually && (
            <Button
              icon={
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M2 13L1.34921 12.2407C1.16773 12.3963 1.04797 12.6117 1.01163 12.8479L2 13ZM22.5 4L23.49 4.14142C23.5309 3.85444 23.4454 3.5638 23.2555 3.3448C23.0655 3.1258 22.7899 3 22.5 3V4ZM12.5 4V3C12.2613 3 12.0305 3.08539 11.8492 3.24074L12.5 4ZM1 19.5L0.0116283 19.3479C-0.0327373 19.6363 0.051055 19.9297 0.241035 20.1511C0.431014 20.3726 0.708231 20.5 1 20.5V19.5ZM11.5 19.5V20.5C11.7373 20.5 11.9668 20.4156 12.1476 20.2619L11.5 19.5ZM21.5 11L22.1476 11.7619C22.3337 11.6038 22.4554 11.3831 22.49 11.1414L21.5 11ZM2 14H12.5V12H2V14ZM13.169 13.7433L23.169 4.74329L21.831 3.25671L11.831 12.2567L13.169 13.7433ZM22.5 3H12.5V5H22.5V3ZM11.8492 3.24074L1.34921 12.2407L2.65079 13.7593L13.1508 4.75926L11.8492 3.24074ZM1.01163 12.8479L0.0116283 19.3479L1.98837 19.6521L2.98837 13.1521L1.01163 12.8479ZM1 20.5H11.5V18.5H1V20.5ZM12.4884 19.6521L13.4884 13.1521L11.5116 12.8479L10.5116 19.3479L12.4884 19.6521ZM21.51 3.85858L20.51 10.8586L22.49 11.1414L23.49 4.14142L21.51 3.85858ZM20.8524 10.2381L10.8524 18.7381L12.1476 20.2619L22.1476 11.7619L20.8524 10.2381Z"
                    fill="currentColor"
                  />
                </svg>
              }
              disabled={!hadDrawSomething() || isInpaintingLoading}
              onClick={() => {
                if (!isInpaintingLoading && hadDrawSomething()) {
                  runInpainting()
                }
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}

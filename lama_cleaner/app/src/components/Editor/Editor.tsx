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
import { useRecoilState, useRecoilValue } from 'recoil'
import { useWindowSize, useKey, useKeyPressEvent } from 'react-use'
import inpaint from '../../adapters/inpainting'
import Button from '../shared/Button'
import Slider from './Slider'
import SizeSelector from './SizeSelector'
import {
  askWritePermission,
  copyCanvasImage,
  downloadImage,
  isMidClick,
  isRightClick,
  loadImage,
  srcToFile,
  useImage,
} from '../../utils'
import {
  croperState,
  fileState,
  isInpaintingState,
  isSDState,
  propmtState,
  runManuallyState,
  seedState,
  settingState,
  toastState,
} from '../../store/Atoms'
import useHotKey from '../../hooks/useHotkey'
import Croper from '../Croper/Croper'
import emitter, { EVENT_PROMPT } from '../../event'
import FileSelect from '../FileSelect/FileSelect'

const TOOLBAR_SIZE = 200
const MIN_BRUSH_SIZE = 10
const MAX_BRUSH_SIZE = 200
const BRUSH_COLOR = '#ffcc00bb'

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

export default function Editor() {
  const [file, setFile] = useRecoilState(fileState)
  const promptVal = useRecoilValue(propmtState)
  const settings = useRecoilValue(settingState)
  const [seedVal, setSeed] = useRecoilState(seedState)
  const croperRect = useRecoilValue(croperState)
  const [toastVal, setToastState] = useRecoilState(toastState)
  const [isInpainting, setIsInpainting] = useRecoilState(isInpaintingState)
  const runMannually = useRecoilValue(runManuallyState)
  const isSD = useRecoilValue(isSDState)

  const [brushSize, setBrushSize] = useState(40)
  const [original, isOriginalLoaded] = useImage(file)
  const [renders, setRenders] = useState<HTMLImageElement[]>([])
  const [context, setContext] = useState<CanvasRenderingContext2D>()
  const [maskCanvas] = useState<HTMLCanvasElement>(() => {
    return document.createElement('canvas')
  })
  const [lineGroups, setLineGroups] = useState<LineGroup[]>([])
  const [lastLineGroup, setLastLineGroup] = useState<LineGroup>([])
  const [curLineGroup, setCurLineGroup] = useState<LineGroup>([])
  const [{ x, y }, setCoords] = useState({ x: -1, y: -1 })
  const [showBrush, setShowBrush] = useState(false)
  const [showRefBrush, setShowRefBrush] = useState(false)
  const [isPanning, setIsPanning] = useState<boolean>(false)
  const [isChangingBrushSizeByMouse, setIsChangingBrushSizeByMouse] =
    useState<boolean>(false)
  const [changeBrushSizeByMouseInit, setChangeBrushSizeByMouseInit] = useState({
    x: -1,
    y: -1,
    brushSize: 20,
  })

  const [showOriginal, setShowOriginal] = useState(false)
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

  // redo 相关
  const [redoRenders, setRedoRenders] = useState<HTMLImageElement[]>([])
  const [redoCurLines, setRedoCurLines] = useState<Line[]>([])
  const [redoLineGroups, setRedoLineGroups] = useState<LineGroup[]>([])

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

  const drawLinesOnMask = useCallback(
    (_lineGroups: LineGroup[]) => {
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
    },
    [context, maskCanvas]
  )

  const hadDrawSomething = useCallback(() => {
    return curLineGroup.length !== 0
  }, [curLineGroup])

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

  const runInpainting = useCallback(
    async (prompt?: string, useLastLineGroup?: boolean) => {
      if (file === undefined) {
        return
      }
      // useLastLineGroup 的影响
      // 1. 使用上一次的 mask
      // 2. 结果替换当前 render
      console.log('runInpainting')

      let maskLineGroup = []
      if (useLastLineGroup === true) {
        if (lastLineGroup.length === 0) {
          return
        }
        maskLineGroup = lastLineGroup
      } else {
        if (!hadDrawSomething()) {
          return
        }

        setLastLineGroup(curLineGroup)
        maskLineGroup = curLineGroup
      }

      const newLineGroups = [...lineGroups, maskLineGroup]

      setCurLineGroup([])
      setIsDraging(false)
      setIsInpainting(true)
      if (settings.graduallyInpainting) {
        drawLinesOnMask([maskLineGroup])
      } else {
        drawLinesOnMask(newLineGroups)
      }

      let targetFile = file
      if (settings.graduallyInpainting === true) {
        if (useLastLineGroup === true) {
          // renders.length == 1 还是用原来的
          if (renders.length > 1) {
            const lastRender = renders[renders.length - 2]
            targetFile = await srcToFile(
              lastRender.currentSrc,
              file.name,
              file.type
            )
          }
        } else if (renders.length > 0) {
          console.info('gradually inpainting on last result')

          const lastRender = renders[renders.length - 1]
          targetFile = await srcToFile(
            lastRender.currentSrc,
            file.name,
            file.type
          )
        }
      }

      const sdSeed = settings.sdSeedFixed ? settings.sdSeed : -1

      try {
        const res = await inpaint(
          targetFile,
          maskCanvas.toDataURL(),
          settings,
          croperRect,
          prompt,
          sizeLimit.toString(),
          sdSeed
        )
        if (!res) {
          throw new Error('empty response')
        }
        const { blob, seed } = res
        console.log(seed)
        console.log(settings.sdSeedFixed)
        if (seed && !settings.sdSeedFixed) {
          setSeed(parseInt(seed, 10))
        }
        const newRender = new Image()
        await loadImage(newRender, blob)

        if (useLastLineGroup === true) {
          const prevRenders = renders.slice(0, -1)
          const newRenders = [...prevRenders, newRender]
          setRenders(newRenders)
        } else {
          const newRenders = [...renders, newRender]
          setRenders(newRenders)
        }

        draw(newRender, [])
        // Only append new LineGroup after inpainting success
        setLineGroups(newLineGroups)

        // clear redo stack
        resetRedoState()
      } catch (e: any) {
        setToastState({
          open: true,
          desc: e.message ? e.message : e.toString(),
          state: 'error',
          duration: 4000,
        })
        drawOnCurrentRender([])
      }
      setIsInpainting(false)
    },
    [
      lineGroups,
      curLineGroup,
      maskCanvas,
      settings.graduallyInpainting,
      settings,
      croperRect,
      sizeLimit,
      promptVal,
      drawOnCurrentRender,
      hadDrawSomething,
      drawLinesOnMask,
    ]
  )

  useEffect(() => {
    emitter.on(EVENT_PROMPT, () => {
      if (hadDrawSomething()) {
        runInpainting(promptVal)
      } else if (lastLineGroup.length !== 0) {
        runInpainting(promptVal, true)
      } else {
        setToastState({
          open: true,
          desc: 'Please draw mask on picture',
          state: 'error',
          duration: 1500,
        })
      }
    })
    return () => {
      emitter.off(EVENT_PROMPT)
    }
  }, [hadDrawSomething, runInpainting, prompt])

  const hadRunInpainting = () => {
    return renders.length !== 0
  }

  const handleMultiStrokeKeyDown = () => {
    if (isInpainting) {
      return
    }
    setIsMultiStrokeKeyPressed(true)
  }

  const handleMultiStrokeKeyup = () => {
    if (!isMultiStrokeKeyPressed) {
      return
    }
    if (isInpainting) {
      return
    }

    setIsMultiStrokeKeyPressed(false)

    if (!runMannually) {
      runInpainting()
    }
  }

  const predicate = (event: KeyboardEvent) => {
    return event.key === 'Control' || event.key === 'Meta'
  }

  useKey(predicate, handleMultiStrokeKeyup, { event: 'keyup' }, [
    isInpainting,
    isMultiStrokeKeyPressed,
    hadDrawSomething,
  ])

  useKey(
    predicate,
    handleMultiStrokeKeyDown,
    {
      event: 'keydown',
    },
    [isInpainting]
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

    if (context?.canvas) {
      context.canvas.width = original.naturalWidth
      context.canvas.height = original.naturalHeight
      drawOnCurrentRender([])
    }

    if (!initialCentered) {
      viewportRef.current?.centerView(s, 1)
      setInitialCentered(true)
      const imageSizeLimit = Math.max(original.width, original.height)
      setSizeLimit(imageSizeLimit)
    }
  }, [
    context?.canvas,
    viewportRef,
    original,
    isOriginalLoaded,
    windowSize,
    initialCentered,
    drawOnCurrentRender,
  ])

  // Zoom reset
  const resetZoom = useCallback(() => {
    if (!minScale || !original || !windowSize) {
      return
    }
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }
    const offsetX = (windowSize.width - original.width * minScale) / 2
    const offsetY = (windowSize.height - original.height * minScale) / 2
    viewport.setTransform(offsetX, offsetY, minScale, 200, 'easeOutQuad')
    viewport.state.scale = minScale

    setScale(minScale)
    setPanned(false)
  }, [
    viewportRef,
    windowSize,
    original,
    original.width,
    windowSize.height,
    minScale,
  ])

  const resetRedoState = () => {
    setRedoCurLines([])
    setRedoLineGroups([])
    setRedoRenders([])
  }

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

  useEffect(() => {
    window.addEventListener('blur', () => {
      setIsChangingBrushSizeByMouse(false)
    })
    return () => {
      window.removeEventListener('blur', () => {
        setIsChangingBrushSizeByMouse(false)
      })
    }
  }, [])

  const handleEscPressed = () => {
    if (isInpainting) {
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
      isInpainting,
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
    if (isChangingBrushSizeByMouse) {
      const initX = changeBrushSizeByMouseInit.x
      // move right: increase brush size
      const newSize = changeBrushSizeByMouseInit.brushSize + (x - initX)
      if (newSize <= MAX_BRUSH_SIZE && newSize >= MIN_BRUSH_SIZE) {
        setBrushSize(newSize)
      }
      return
    }
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
    if (isInpainting) {
      return
    }
    if (!isDraging) {
      return
    }

    if (isMultiStrokeKeyPressed) {
      setIsDraging(false)
      return
    }

    if (runMannually) {
      setIsDraging(false)
    } else {
      runInpainting()
    }
  }

  const isOutsideCroper = (clickPnt: { x: number; y: number }) => {
    if (clickPnt.x < croperRect.x) {
      return true
    }
    if (clickPnt.y < croperRect.y) {
      return true
    }
    if (clickPnt.x > croperRect.x + croperRect.width) {
      return true
    }
    if (clickPnt.y > croperRect.y + croperRect.height) {
      return true
    }
    return false
  }

  const onMouseDown = (ev: SyntheticEvent) => {
    if (isChangingBrushSizeByMouse) {
      return
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
    if (isInpainting) {
      return
    }

    if (isRightClick(ev)) {
      return
    }

    if (isMidClick(ev)) {
      setIsPanning(true)
      return
    }

    if (isSD && settings.showCroper && isOutsideCroper(mouseXY(ev))) {
      return
    }

    setIsDraging(true)

    let lineGroup: LineGroup = []
    if (isMultiStrokeKeyPressed || runMannually) {
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
    setLastLineGroup([])

    const lastLine = curLineGroup.pop()!
    const newRedoCurLines = [...redoCurLines, lastLine]
    setRedoCurLines(newRedoCurLines)

    const newLineGroup = [...curLineGroup]
    setCurLineGroup(newLineGroup)
    drawOnCurrentRender(newLineGroup)
  }, [curLineGroup, redoCurLines, drawOnCurrentRender])

  const undoRender = useCallback(() => {
    if (!renders.length) {
      return
    }

    // save line Group
    const latestLineGroup = lineGroups.pop()!
    setRedoLineGroups([...redoLineGroups, latestLineGroup])
    // If render is undo, clear strokes
    setRedoCurLines([])

    setLineGroups([...lineGroups])
    setCurLineGroup([])
    setIsDraging(false)

    // save render
    const lastRender = renders.pop()!
    setRedoRenders([...redoRenders, lastRender])

    const newRenders = [...renders]
    setRenders(newRenders)
    if (newRenders.length === 0) {
      draw(original, [])
    } else {
      draw(newRenders[newRenders.length - 1], [])
    }
  }, [draw, renders, redoRenders, redoLineGroups, lineGroups, original])

  const undo = () => {
    if (runMannually && curLineGroup.length !== 0) {
      undoStroke()
    } else {
      undoRender()
    }
  }

  // Handle Cmd+Z
  const undoPredicate = (event: KeyboardEvent) => {
    // TODO: fix prompt input ctrl+z
    const isCmdZ =
      (event.metaKey || event.ctrlKey) && !event.shiftKey && event.key === 'z'
    // Handle tab switch
    if (event.key === 'Tab') {
      event.preventDefault()
    }
    if (isCmdZ) {
      event.preventDefault()
      console.log('undo')
      return true
    }
    return false
  }

  useKey(undoPredicate, undo, undefined, [undoStroke, undoRender, isSD])

  const disableUndo = () => {
    if (isInpainting) {
      return true
    }
    if (renders.length > 0) {
      return false
    }

    if (runMannually) {
      if (curLineGroup.length === 0) {
        return true
      }
    } else if (renders.length === 0) {
      return true
    }

    return false
  }

  const redoStroke = useCallback(() => {
    if (redoCurLines.length === 0) {
      return
    }
    const line = redoCurLines.pop()!
    setRedoCurLines([...redoCurLines])

    const newLineGroup = [...curLineGroup, line]
    setCurLineGroup(newLineGroup)
    drawOnCurrentRender(newLineGroup)
  }, [curLineGroup, redoCurLines, drawOnCurrentRender])

  const redoRender = useCallback(() => {
    if (redoRenders.length === 0) {
      return
    }
    const lineGroup = redoLineGroups.pop()!
    setRedoLineGroups([...redoLineGroups])

    setLineGroups([...lineGroups, lineGroup])
    setCurLineGroup([])
    setIsDraging(false)

    const render = redoRenders.pop()!
    const newRenders = [...renders, render]
    setRenders(newRenders)
    draw(newRenders[newRenders.length - 1], [])
  }, [draw, renders, redoRenders, redoLineGroups, lineGroups, original])

  const redo = () => {
    if (runMannually && redoCurLines.length !== 0) {
      redoStroke()
    } else {
      redoRender()
    }
  }

  // Handle Cmd+shift+Z
  const redoPredicate = (event: KeyboardEvent) => {
    const isCmdZ =
      (event.metaKey || event.ctrlKey) &&
      event.shiftKey &&
      event.key.toLowerCase() === 'z'
    // Handle tab switch
    if (event.key === 'Tab') {
      event.preventDefault()
    }
    if (isCmdZ) {
      event.preventDefault()
      console.log('redo')
      return true
    }
    return false
  }

  useKey(redoPredicate, redo, undefined, [redoStroke, redoRender, isSD])

  const disableRedo = () => {
    if (isInpainting) {
      return true
    }
    if (redoRenders.length > 0) {
      return false
    }

    if (runMannually) {
      if (redoCurLines.length === 0) {
        return true
      }
    } else if (redoRenders.length === 0) {
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
    if (file === undefined) {
      return
    }
    const name = file.name.replace(/(\.[\w\d_-]+)$/i, '_cleanup$1')
    const curRender = renders[renders.length - 1]
    downloadImage(curRender.currentSrc, name)
    if (settings.downloadMask) {
      let maskFileName = file.name.replace(/(\.[\w\d_-]+)$/i, '_mask$1')
      maskFileName = maskFileName.replace(/\.[^/.]+$/, '.jpg')

      drawLinesOnMask(lineGroups)
      // Create a link
      const aDownloadLink = document.createElement('a')
      // Add the name of the file to the link
      aDownloadLink.download = maskFileName
      // Attach the data to the link
      aDownloadLink.href = maskCanvas.toDataURL('image/jpeg')
      // Get the code to click the download link
      aDownloadLink.click()
    }
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
  useHotKey('[', () => {
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

  useHotKey(']', () => {
    setBrushSize(currentBrushSize => {
      return currentBrushSize + 10
    })
  })

  // Manual Inpainting Hotkey
  useHotKey(
    'shift+r',
    () => {
      if (runMannually && hadDrawSomething()) {
        runInpainting()
      }
    },
    {},
    [runMannually, runInpainting, hadDrawSomething]
  )

  useHotKey(
    'ctrl+c, cmd+c',
    async () => {
      const hasPermission = await askWritePermission()
      if (hasPermission && renders.length > 0) {
        if (context?.canvas) {
          await copyCanvasImage(context?.canvas)
          setToastState({
            open: true,
            desc: 'Copy inpainting result to clipboard',
            state: 'success',
            duration: 3000,
          })
        }
      }
    },
    {},
    [renders, context]
  )

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

  useKeyPressEvent(
    'Alt',
    ev => {
      ev?.preventDefault()
      ev?.stopPropagation()
      setIsChangingBrushSizeByMouse(true)
      setChangeBrushSizeByMouseInit({ x, y, brushSize })
    },
    ev => {
      ev?.preventDefault()
      ev?.stopPropagation()
      setIsChangingBrushSizeByMouse(false)
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

  const renderFileSelect = () => {
    return (
      <div className="landing-file-selector">
        <FileSelect
          onSelection={async f => {
            setFile(f)
          }}
        />
      </div>
    )
  }

  const renderCanvas = () => {
    return (
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
        minScale={minScale * 0.6}
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
          contentClass={isInpainting ? 'editor-canvas-loading' : ''}
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
                transition: 'clip-path 300ms cubic-bezier(0.4, 0, 0.2, 1)',
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

          {settings.showCroper ? (
            <Croper
              maxHeight={original.naturalHeight}
              maxWidth={original.naturalWidth}
              minHeight={Math.min(256, original.naturalHeight)}
              minWidth={Math.min(256, original.naturalWidth)}
              scale={scale}
            />
          ) : (
            <></>
          )}
        </TransformComponent>
      </TransformWrapper>
    )
  }

  return (
    <div
      className="editor-container"
      aria-hidden="true"
      onMouseMove={onMouseMove}
      onMouseUp={onPointerUp}
    >
      {file === undefined ? renderFileSelect() : renderCanvas()}

      {showBrush && !isInpainting && !isPanning && (
        <div
          className="brush-shape"
          style={getBrushStyle(
            isChangingBrushSizeByMouse ? changeBrushSizeByMouseInit.x : x,
            isChangingBrushSizeByMouse ? changeBrushSizeByMouseInit.y : y
          )}
        />
      )}

      {showRefBrush && (
        <div
          className="brush-shape"
          style={getBrushStyle(windowCenterX, windowCenterY)}
        />
      )}

      <div className="editor-toolkit-panel">
        {isSD || file === undefined ? (
          <></>
        ) : (
          <SizeSelector
            onChange={onSizeLimitChange}
            originalWidth={original.naturalWidth}
            originalHeight={original.naturalHeight}
          />
        )}
        <Slider
          label="Brush"
          min={MIN_BRUSH_SIZE}
          max={MAX_BRUSH_SIZE}
          value={brushSize}
          onChange={handleSliderChange}
          onClick={() => setShowRefBrush(false)}
        />
        <div className="editor-toolkit-btns">
          <Button
            toolTip="Reset Zoom & Pan"
            tooltipPosition="top"
            icon={<ArrowsExpandIcon />}
            disabled={scale === minScale && panned === false}
            onClick={resetZoom}
          />
          <Button
            toolTip="Undo"
            tooltipPosition="top"
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
            toolTip="Redo"
            tooltipPosition="top"
            icon={
              <svg
                width="19"
                height="9"
                viewBox="0 0 19 9"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                style={{ transform: 'scale(-1,1)' }}
              >
                <path
                  d="M2 1C2 0.447715 1.55228 0 1 0C0.447715 0 0 0.447715 0 1H2ZM1 8H0V9H1V8ZM8 9C8.55228 9 9 8.55229 9 8C9 7.44771 8.55228 7 8 7V9ZM16.5963 7.42809C16.8327 7.92721 17.429 8.14016 17.9281 7.90374C18.4272 7.66731 18.6402 7.07103 18.4037 6.57191L16.5963 7.42809ZM16.9468 5.83205L17.8505 5.40396L16.9468 5.83205ZM0 1V8H2V1H0ZM1 9H8V7H1V9ZM1.66896 8.74329L6.66896 4.24329L5.33104 2.75671L0.331035 7.25671L1.66896 8.74329ZM16.043 6.26014L16.5963 7.42809L18.4037 6.57191L17.8505 5.40396L16.043 6.26014ZM6.65079 4.25926C9.67554 1.66661 14.3376 2.65979 16.043 6.26014L17.8505 5.40396C15.5805 0.61182 9.37523 -0.710131 5.34921 2.74074L6.65079 4.25926Z"
                  fill="currentColor"
                />
              </svg>
            }
            onClick={redo}
            disabled={disableRedo()}
          />
          <Button
            toolTip="Show Original"
            tooltipPosition="top"
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
              }, 300)
            }}
            disabled={renders.length === 0}
          />
          <Button
            toolTip="Save Image"
            tooltipPosition="top"
            icon={<DownloadIcon />}
            disabled={!renders.length}
            onClick={download}
          />

          {settings.runInpaintingManually && (
            <Button
              toolTip="Run Inpainting"
              tooltipPosition="top"
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
              disabled={!hadDrawSomething() || isInpainting}
              onClick={() => {
                if (!isInpainting && hadDrawSomething()) {
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

import React, {
  SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  CursorArrowRaysIcon,
  EyeIcon,
  ArrowsPointingOutIcon,
  ArrowDownTrayIcon,
} from '@heroicons/react/24/outline'
import {
  ReactZoomPanPinchRef,
  TransformComponent,
  TransformWrapper,
} from 'react-zoom-pan-pinch'
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil'
import { useWindowSize, useKey, useKeyPressEvent } from 'react-use'
import inpaint, { downloadToOutput, runPlugin } from '../../adapters/inpainting'
import Button from '../shared/Button'
import Slider from './Slider'
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
  appState,
  brushSizeState,
  croperState,
  enableFileManagerState,
  fileState,
  imageHeightState,
  imageWidthState,
  interactiveSegClicksState,
  isDiffusionModelsState,
  isEnableAutoSavingState,
  isInpaintingState,
  isInteractiveSegRunningState,
  isInteractiveSegState,
  isPix2PixState,
  isPluginRunningState,
  isProcessingState,
  negativePropmtState,
  propmtState,
  runManuallyState,
  seedState,
  settingState,
  toastState,
} from '../../store/Atoms'
import useHotKey from '../../hooks/useHotkey'
import Croper from '../Croper/Croper'
import emitter, {
  EVENT_PROMPT,
  EVENT_CUSTOM_MASK,
  EVENT_PAINT_BY_EXAMPLE,
  RERUN_LAST_MASK,
} from '../../event'
import FileSelect from '../FileSelect/FileSelect'
import InteractiveSeg from '../InteractiveSeg/InteractiveSeg'
import InteractiveSegConfirmActions from '../InteractiveSeg/ConfirmActions'
import InteractiveSegReplaceModal from '../InteractiveSeg/ReplaceModal'
import { PluginName } from '../Plugins/Plugins'
import MakeGIF from './MakeGIF'

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
  const negativePromptVal = useRecoilValue(negativePropmtState)
  const settings = useRecoilValue(settingState)
  const [seedVal, setSeed] = useRecoilState(seedState)
  const croperRect = useRecoilValue(croperState)
  const setToastState = useSetRecoilState(toastState)
  const [isInpainting, setIsInpainting] = useRecoilState(isInpaintingState)
  const setIsPluginRunning = useSetRecoilState(isPluginRunningState)
  const isProcessing = useRecoilValue(isProcessingState)
  const runMannually = useRecoilValue(runManuallyState)
  const isDiffusionModels = useRecoilValue(isDiffusionModelsState)
  const isPix2Pix = useRecoilValue(isPix2PixState)
  const [isInteractiveSeg, setIsInteractiveSeg] = useRecoilState(
    isInteractiveSegState
  )
  const setIsInteractiveSegRunning = useSetRecoilState(
    isInteractiveSegRunningState
  )

  const [showInteractiveSegModal, setShowInteractiveSegModal] = useState(false)
  const [interactiveSegMask, setInteractiveSegMask] =
    useState<HTMLImageElement | null>(null)
  // only used while interactive segmentation is on
  const [tmpInteractiveSegMask, setTmpInteractiveSegMask] =
    useState<HTMLImageElement | null>(null)
  const [prevInteractiveSegMask, setPrevInteractiveSegMask] = useState<
    HTMLImageElement | null | undefined
  >(null)

  const [clicks, setClicks] = useRecoilState(interactiveSegClicksState)

  const [brushSize, setBrushSize] = useRecoilState(brushSizeState)

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
  const enableFileManager = useRecoilValue(enableFileManagerState)
  const isEnableAutoSaving = useRecoilValue(isEnableAutoSavingState)

  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState)
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState)
  const app = useRecoilValue(appState)

  const draw = useCallback(
    (render: HTMLImageElement, lineGroup: LineGroup) => {
      if (!context) {
        return
      }
      console.log(
        `[draw] render size: ${render.width}x${render.height} image size: ${imageWidth}x${imageHeight} canvas size: ${context.canvas.width}x${context.canvas.height}`
      )

      context.clearRect(0, 0, context.canvas.width, context.canvas.height)
      context.drawImage(render, 0, 0, imageWidth, imageHeight)
      if (isInteractiveSeg && tmpInteractiveSegMask !== null) {
        context.drawImage(tmpInteractiveSegMask, 0, 0, imageWidth, imageHeight)
      }
      if (!isInteractiveSeg && interactiveSegMask !== null) {
        context.drawImage(interactiveSegMask, 0, 0, imageWidth, imageHeight)
      }
      drawLines(context, lineGroup)
    },
    [
      context,
      isInteractiveSeg,
      tmpInteractiveSegMask,
      interactiveSegMask,
      imageHeight,
      imageWidth,
    ]
  )

  const drawLinesOnMask = useCallback(
    (_lineGroups: LineGroup[], maskImage?: HTMLImageElement | null) => {
      if (!context?.canvas.width || !context?.canvas.height) {
        throw new Error('canvas has invalid size')
      }
      maskCanvas.width = context?.canvas.width
      maskCanvas.height = context?.canvas.height
      const ctx = maskCanvas.getContext('2d')
      if (!ctx) {
        throw new Error('could not retrieve mask canvas')
      }

      if (maskImage !== undefined && maskImage !== null) {
        // TODO: check whether draw yellow mask works on backend
        ctx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
      }

      _lineGroups.forEach(lineGroup => {
        drawLines(ctx, lineGroup, 'white')
      })

      if (
        (maskImage === undefined || maskImage === null) &&
        _lineGroups.length === 1 &&
        _lineGroups[0].length === 0 &&
        isPix2Pix
      ) {
        // For InstructPix2Pix without mask
        drawLines(
          ctx,
          [
            {
              size: 9999999999,
              pts: [
                { x: 0, y: 0 },
                { x: imageWidth, y: 0 },
                { x: imageWidth, y: imageHeight },
                { x: 0, y: imageHeight },
              ],
            },
          ],
          'white'
        )
      }
    },
    [context, maskCanvas, isPix2Pix, imageWidth, imageHeight]
  )

  const hadDrawSomething = useCallback(() => {
    if (isPix2Pix) {
      return true
    }
    return curLineGroup.length !== 0
  }, [curLineGroup, isPix2Pix])

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
    async (
      useLastLineGroup?: boolean,
      customMask?: File,
      maskImage?: HTMLImageElement | null,
      paintByExampleImage?: File
    ) => {
      // customMask: mask uploaded by user
      // maskImage: mask from interactive segmentation
      if (file === undefined) {
        return
      }
      const useCustomMask = customMask !== undefined && customMask !== null
      const useMaskImage = maskImage !== undefined && maskImage !== null
      // useLastLineGroup 的影响
      // 1. 使用上一次的 mask
      // 2. 结果替换当前 render
      console.log('runInpainting')
      console.log({
        useCustomMask,
        useMaskImage,
      })

      let maskLineGroup: LineGroup = []
      if (useLastLineGroup === true) {
        if (lastLineGroup.length === 0) {
          return
        }
        maskLineGroup = lastLineGroup
      } else if (!useCustomMask) {
        if (!hadDrawSomething() && !useMaskImage) {
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
        drawLinesOnMask([maskLineGroup], maskImage)
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

      try {
        const res = await inpaint(
          targetFile,
          settings,
          croperRect,
          promptVal,
          negativePromptVal,
          seedVal,
          useCustomMask ? undefined : maskCanvas.toDataURL(),
          useCustomMask ? customMask : undefined,
          paintByExampleImage
        )
        if (!res) {
          throw new Error('Something went wrong on server side.')
        }
        const { blob, seed } = res
        if (seed) {
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
      setPrevInteractiveSegMask(maskImage)
      setTmpInteractiveSegMask(null)
      setInteractiveSegMask(null)
    },
    [
      lineGroups,
      curLineGroup,
      maskCanvas,
      settings.graduallyInpainting,
      settings,
      croperRect,
      promptVal,
      negativePromptVal,
      drawOnCurrentRender,
      hadDrawSomething,
      drawLinesOnMask,
      seedVal,
    ]
  )

  useEffect(() => {
    emitter.on(EVENT_PROMPT, () => {
      if (hadDrawSomething() || interactiveSegMask) {
        runInpainting(false, undefined, interactiveSegMask)
      } else if (lastLineGroup.length !== 0) {
        // 使用上一次手绘的 mask 生成
        runInpainting(true, undefined, prevInteractiveSegMask)
      } else if (prevInteractiveSegMask) {
        // 使用上一次 IS 的 mask 生成
        runInpainting(false, undefined, prevInteractiveSegMask)
      } else if (isPix2Pix) {
        runInpainting(false, undefined, null)
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
  }, [
    hadDrawSomething,
    runInpainting,
    promptVal,
    interactiveSegMask,
    prevInteractiveSegMask,
  ])

  useEffect(() => {
    emitter.on(EVENT_CUSTOM_MASK, (data: any) => {
      // TODO: not work with paint by example
      runInpainting(false, data.mask)
    })

    return () => {
      emitter.off(EVENT_CUSTOM_MASK)
    }
  }, [runInpainting])

  useEffect(() => {
    emitter.on(EVENT_PAINT_BY_EXAMPLE, (data: any) => {
      if (hadDrawSomething() || interactiveSegMask) {
        runInpainting(false, undefined, interactiveSegMask, data.image)
      } else if (lastLineGroup.length !== 0) {
        // 使用上一次手绘的 mask 生成
        runInpainting(true, undefined, prevInteractiveSegMask, data.image)
      } else if (prevInteractiveSegMask) {
        // 使用上一次 IS 的 mask 生成
        runInpainting(false, undefined, prevInteractiveSegMask, data.image)
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
      emitter.off(EVENT_PAINT_BY_EXAMPLE)
    }
  }, [runInpainting])

  useEffect(() => {
    emitter.on(RERUN_LAST_MASK, () => {
      if (lastLineGroup.length !== 0) {
        // 使用上一次手绘的 mask 生成
        runInpainting(true, undefined, prevInteractiveSegMask)
      } else if (prevInteractiveSegMask) {
        // 使用上一次 IS 的 mask 生成
        runInpainting(false, undefined, prevInteractiveSegMask)
      } else {
        setToastState({
          open: true,
          desc: 'No mask to reuse',
          state: 'error',
          duration: 1500,
        })
      }
    })
    return () => {
      emitter.off(RERUN_LAST_MASK)
    }
  }, [runInpainting])

  const getCurrentRender = useCallback(async () => {
    let targetFile = file
    if (renders.length > 0) {
      const lastRender = renders[renders.length - 1]
      targetFile = await srcToFile(lastRender.currentSrc, file.name, file.type)
    }
    return targetFile
  }, [file, renders])

  useEffect(() => {
    emitter.on(PluginName.InteractiveSeg, () => {
      setIsInteractiveSeg(true)
      if (interactiveSegMask !== null) {
        setShowInteractiveSegModal(true)
      }
    })
    return () => {
      emitter.off(PluginName.InteractiveSeg)
    }
  })

  const runRenderablePlugin = useCallback(
    async (name: string, data?: any) => {
      if (isProcessing) {
        return
      }
      try {
        // TODO 要不要加 undoCurrentLine？？
        const start = new Date()
        setIsPluginRunning(true)
        const targetFile = await getCurrentRender()
        const res = await runPlugin(name, targetFile, data?.upscale)
        if (!res) {
          throw new Error('Something went wrong on server side.')
        }
        const { blob } = res
        const newRender = new Image()
        await loadImage(newRender, blob)
        setImageHeight(newRender.height)
        setImageWidth(newRender.width)
        const newRenders = [...renders, newRender]
        setRenders(newRenders)
        const newLineGroups = [...lineGroups, []]
        setLineGroups(newLineGroups)

        const end = new Date()
        const time = end.getTime() - start.getTime()

        setToastState({
          open: true,
          desc: `Run ${name} successfully in ${time / 1000}s`,
          state: 'success',
          duration: 3000,
        })

        const rW = windowSize.width / newRender.width
        const rH = (windowSize.height - TOOLBAR_SIZE) / newRender.height
        let s = 1.0
        if (rW < 1 || rH < 1) {
          s = Math.min(rW, rH)
        }
        setMinScale(s)
        setScale(s)
        viewportRef.current?.centerView(s, 1)
      } catch (e: any) {
        setToastState({
          open: true,
          desc: e.message ? e.message : e.toString(),
          state: 'error',
          duration: 3000,
        })
      } finally {
        setIsPluginRunning(false)
      }
    },
    [
      renders,
      setRenders,
      getCurrentRender,
      setIsPluginRunning,
      isProcessing,
      setImageHeight,
      setImageWidth,
      lineGroups,
      viewportRef,
      windowSize,
      setLineGroups,
    ]
  )

  useEffect(() => {
    emitter.on(PluginName.RemoveBG, () => {
      runRenderablePlugin(PluginName.RemoveBG)
    })
    return () => {
      emitter.off(PluginName.RemoveBG)
    }
  }, [runRenderablePlugin])

  useEffect(() => {
    emitter.on(PluginName.GFPGAN, () => {
      runRenderablePlugin(PluginName.GFPGAN)
    })
    return () => {
      emitter.off(PluginName.GFPGAN)
    }
  }, [runRenderablePlugin])

  useEffect(() => {
    emitter.on(PluginName.RestoreFormer, () => {
      runRenderablePlugin(PluginName.RestoreFormer)
    })
    return () => {
      emitter.off(PluginName.RestoreFormer)
    }
  }, [runRenderablePlugin])

  useEffect(() => {
    emitter.on(PluginName.RealESRGAN, (data: any) => {
      runRenderablePlugin(PluginName.RealESRGAN, data)
    })
    return () => {
      emitter.off(PluginName.RealESRGAN)
    }
  }, [runRenderablePlugin])

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

  const getCurrentWidthHeight = useCallback(() => {
    let width = 512
    let height = 512
    if (!isOriginalLoaded) {
      return [width, height]
    }
    if (renders.length === 0) {
      width = original.naturalWidth
      height = original.naturalHeight
    } else if (renders.length !== 0) {
      width = renders[renders.length - 1].width
      height = renders[renders.length - 1].height
    }

    return [width, height]
  }, [original, isOriginalLoaded, renders])

  // Draw once the original image is loaded
  useEffect(() => {
    if (!isOriginalLoaded) {
      return
    }

    const [width, height] = getCurrentWidthHeight()
    setImageWidth(width)
    setImageHeight(height)

    const rW = windowSize.width / width
    const rH = (windowSize.height - TOOLBAR_SIZE) / height

    let s = 1.0
    if (rW < 1 || rH < 1) {
      s = Math.min(rW, rH)
    }
    setMinScale(s)
    setScale(s)

    console.log(
      `[on file load] image size: ${width}x${height}, canvas size: ${context?.canvas.width}x${context?.canvas.height} scale: ${s}, initialCentered: ${initialCentered}`
    )

    if (context?.canvas) {
      context.canvas.width = width
      context.canvas.height = height
      console.log('[on file load] set canvas size && drawOnCurrentRender')
      drawOnCurrentRender([])
    }

    if (!initialCentered) {
      // 防止每次擦除以后图片 zoom 还原
      viewportRef.current?.centerView(s, 1)
      console.log('[on file load] centerView')
      setInitialCentered(true)
    }
  }, [
    context?.canvas,
    viewportRef,
    original,
    isOriginalLoaded,
    windowSize,
    initialCentered,
    drawOnCurrentRender,
    getCurrentWidthHeight,
  ])

  useEffect(() => {
    console.log('[useEffect] centerView')
    // render 改变尺寸以后，undo/redo 重新 center
    viewportRef?.current?.centerView(minScale, 1)
  }, [context?.canvas.height, context?.canvas.width, viewportRef, minScale])

  // Zoom reset
  const resetZoom = useCallback(() => {
    if (!minScale || !windowSize) {
      return
    }
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }
    const offsetX = (windowSize.width - imageWidth * minScale) / 2
    const offsetY = (windowSize.height - imageHeight * minScale) / 2
    viewport.setTransform(offsetX, offsetY, minScale, 200, 'easeOutQuad')
    viewport.state.scale = minScale

    setScale(minScale)
    setPanned(false)
  }, [
    viewportRef,
    windowSize,
    imageHeight,
    imageWidth,
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

  const onInteractiveCancel = useCallback(() => {
    setIsInteractiveSeg(false)
    setIsInteractiveSegRunning(false)
    setClicks([])
    setTmpInteractiveSegMask(null)
  }, [])

  const handleEscPressed = () => {
    if (isProcessing) {
      return
    }

    if (isInteractiveSeg) {
      onInteractiveCancel()
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
      isInteractiveSeg,
      onInteractiveCancel,
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
    if (isInteractiveSeg) {
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

  const runInteractiveSeg = async (newClicks: number[][]) => {
    if (!file) {
      return
    }

    setIsInteractiveSegRunning(true)
    const targetFile = await getCurrentRender()
    const prevMask = null
    // prev_mask seems to be not working better
    // if (tmpInteractiveSegMask !== null) {
    //   prevMask = await srcToFile(
    //     tmpInteractiveSegMask.currentSrc,
    //     'prev_mask.jpg',
    //     'image/jpeg'
    //   )
    // }

    try {
      const res = await runPlugin(
        PluginName.InteractiveSeg,
        targetFile,
        undefined,
        prevMask,
        newClicks
      )
      if (!res) {
        throw new Error('Something went wrong on server side.')
      }
      const { blob } = res
      const img = new Image()
      img.onload = () => {
        setTmpInteractiveSegMask(img)
      }
      img.src = blob
    } catch (e: any) {
      setToastState({
        open: true,
        desc: e.message ? e.message : e.toString(),
        state: 'error',
        duration: 4000,
      })
    }
    setIsInteractiveSegRunning(false)
  }

  const onPointerUp = (ev: SyntheticEvent) => {
    if (isMidClick(ev)) {
      setIsPanning(false)
    }
    if (isInteractiveSeg) {
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

  const onCanvasMouseUp = (ev: SyntheticEvent) => {
    if (isInteractiveSeg) {
      const xy = mouseXY(ev)
      const newClicks: number[][] = [...clicks]
      if (isRightClick(ev)) {
        newClicks.push([xy.x, xy.y, 0, newClicks.length])
      } else {
        newClicks.push([xy.x, xy.y, 1, newClicks.length])
      }
      runInteractiveSeg(newClicks)
      setClicks(newClicks)
    }
  }

  const onMouseDown = (ev: SyntheticEvent) => {
    if (isProcessing) {
      return
    }
    if (isInteractiveSeg) {
      return
    }
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

    if (isRightClick(ev)) {
      return
    }

    if (isMidClick(ev)) {
      setIsPanning(true)
      return
    }

    if (
      isDiffusionModels &&
      settings.showCroper &&
      isOutsideCroper(mouseXY(ev))
    ) {
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
    // if (newRenders.length === 0) {
    //   draw(original, [])
    // } else {
    //   draw(newRenders[newRenders.length - 1], [])
    // }
  }, [
    draw,
    renders,
    redoRenders,
    redoLineGroups,
    lineGroups,
    original,
    context,
  ])

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
      return true
    }
    return false
  }

  useKey(undoPredicate, undo, undefined, [
    undoStroke,
    undoRender,
    runMannually,
    curLineGroup,
    context?.canvas,
    renders,
  ])

  const disableUndo = () => {
    if (isProcessing) {
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
    // draw(newRenders[newRenders.length - 1], [])
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
      return true
    }
    return false
  }

  useKey(redoPredicate, redo, undefined, [
    redoStroke,
    redoRender,
    runMannually,
    redoCurLines,
  ])

  const disableRedo = () => {
    if (isProcessing) {
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
    if ((enableFileManager || isEnableAutoSaving) && renders.length > 0) {
      try {
        downloadToOutput(renders[renders.length - 1], file.name, file.type)
        setToastState({
          open: true,
          desc: `Save image success`,
          state: 'success',
          duration: 2000,
        })
      } catch (e: any) {
        setToastState({
          open: true,
          desc: e.message ? e.message : e.toString(),
          state: 'error',
          duration: 2000,
        })
      }
      return
    }

    // TODO: download to output directory
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
    setBrushSize((currentBrushSize: number) => {
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
    setBrushSize((currentBrushSize: number) => {
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
      if (!app.disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        setShowBrush(false)
        setIsPanning(true)
      }
    },
    ev => {
      if (!app.disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        setShowBrush(true)
        setIsPanning(false)
      }
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

  const renderInteractiveSegCursor = () => {
    return (
      <div
        className="interactive-seg-cursor"
        style={{
          left: `${x}px`,
          top: `${y}px`,
          // transform: 'translate(-50%, -50%)',
        }}
      >
        <CursorArrowRaysIcon />
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
          contentClass={isProcessing ? 'editor-canvas-loading' : ''}
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
              onMouseUp={onCanvasMouseUp}
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
                width: `${imageWidth}px`,
                height: `${imageHeight}px`,
              }}
            >
              {showOriginal && (
                <>
                  <div
                    className="editor-slider"
                    style={{
                      marginRight: `${sliderPos}%`,
                    }}
                  />
                  <img
                    className="original-image"
                    src={original.src}
                    alt="original"
                    style={{
                      width: `${imageWidth}px`,
                      height: `${imageHeight}px`,
                    }}
                  />
                </>
              )}
            </div>
          </div>

          <Croper
            maxHeight={imageHeight}
            maxWidth={imageWidth}
            minHeight={Math.min(256, imageHeight)}
            minWidth={Math.min(256, imageWidth)}
            scale={scale}
            show={isDiffusionModels && settings.showCroper}
          />

          {isInteractiveSeg ? <InteractiveSeg /> : <></>}
        </TransformComponent>
      </TransformWrapper>
    )
  }

  const onInteractiveAccept = () => {
    setInteractiveSegMask(tmpInteractiveSegMask)
    setTmpInteractiveSegMask(null)

    if (!runMannually && tmpInteractiveSegMask) {
      runInpainting(false, undefined, tmpInteractiveSegMask)
    }
  }

  return (
    <div
      className="editor-container"
      aria-hidden="true"
      onMouseMove={onMouseMove}
      onMouseUp={onPointerUp}
    >
      <MakeGIF renders={renders} />
      <InteractiveSegConfirmActions
        onAcceptClick={onInteractiveAccept}
        onCancelClick={onInteractiveCancel}
      />
      {file === undefined ? renderFileSelect() : renderCanvas()}

      {showBrush &&
        !isInpainting &&
        !isPanning &&
        (isInteractiveSeg ? (
          renderInteractiveSegCursor()
        ) : (
          <div
            className="brush-shape"
            style={getBrushStyle(
              isChangingBrushSizeByMouse ? changeBrushSizeByMouseInit.x : x,
              isChangingBrushSizeByMouse ? changeBrushSizeByMouseInit.y : y
            )}
          />
        ))}

      {showRefBrush && (
        <div
          className="brush-shape"
          style={getBrushStyle(windowCenterX, windowCenterY)}
        />
      )}

      <div className="editor-toolkit-panel">
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
            icon={<ArrowsPointingOutIcon />}
            disabled={scale === minScale && panned === false}
            onClick={resetZoom}
          />
          <Button
            toolTip="Undo"
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
            icon={<ArrowDownTrayIcon />}
            disabled={!renders.length}
            onClick={download}
          />

          {settings.runInpaintingManually && !isDiffusionModels && (
            <Button
              toolTip="Run Inpainting"
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
              disabled={
                isProcessing ||
                (!hadDrawSomething() && interactiveSegMask === null)
              }
              onClick={() => {
                // ensured by disabled
                runInpainting(false, undefined, interactiveSegMask)
              }}
            />
          )}
        </div>
      </div>
      <InteractiveSegReplaceModal
        show={showInteractiveSegModal}
        onClose={() => {
          onInteractiveCancel()
          setShowInteractiveSegModal(false)
        }}
        onCleanClick={() => {
          onInteractiveCancel()
          setInteractiveSegMask(null)
        }}
        onReplaceClick={() => {
          setShowInteractiveSegModal(false)
          setIsInteractiveSeg(true)
        }}
      />
    </div>
  )
}

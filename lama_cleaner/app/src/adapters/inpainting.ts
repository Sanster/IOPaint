import { Rect, Settings } from '../store/Atoms'
import { dataURItoBlob, loadImage, srcToFile } from '../utils'

export const API_ENDPOINT = `${process.env.REACT_APP_INPAINTING_URL}`

export default async function inpaint(
  imageFile: File,
  settings: Settings,
  croperRect: Rect,
  prompt?: string,
  negativePrompt?: string,
  sizeLimit?: string,
  seed?: number,
  maskBase64?: string,
  customMask?: File,
  paintByExampleImage?: File
) {
  // 1080, 2000, Original
  const fd = new FormData()
  fd.append('image', imageFile)
  if (maskBase64 !== undefined) {
    fd.append('mask', dataURItoBlob(maskBase64))
  } else if (customMask !== undefined) {
    fd.append('mask', customMask)
  }

  const hdSettings = settings.hdSettings[settings.model]
  fd.append('ldmSteps', settings.ldmSteps.toString())
  fd.append('ldmSampler', settings.ldmSampler.toString())
  fd.append('zitsWireframe', settings.zitsWireframe.toString())
  fd.append('hdStrategy', hdSettings.hdStrategy)
  fd.append('hdStrategyCropMargin', hdSettings.hdStrategyCropMargin.toString())
  fd.append(
    'hdStrategyCropTrigerSize',
    hdSettings.hdStrategyCropTrigerSize.toString()
  )
  fd.append(
    'hdStrategyResizeLimit',
    hdSettings.hdStrategyResizeLimit.toString()
  )

  fd.append('prompt', prompt === undefined ? '' : prompt)
  fd.append(
    'negativePrompt',
    negativePrompt === undefined ? '' : negativePrompt
  )
  fd.append('croperX', croperRect.x.toString())
  fd.append('croperY', croperRect.y.toString())
  fd.append('croperHeight', croperRect.height.toString())
  fd.append('croperWidth', croperRect.width.toString())
  fd.append('useCroper', settings.showCroper ? 'true' : 'false')

  fd.append('sdMaskBlur', settings.sdMaskBlur.toString())
  fd.append('sdStrength', settings.sdStrength.toString())
  fd.append('sdSteps', settings.sdSteps.toString())
  fd.append('sdGuidanceScale', settings.sdGuidanceScale.toString())
  fd.append('sdSampler', settings.sdSampler.toString())
  fd.append('sdSeed', seed ? seed.toString() : '-1')
  fd.append('sdMatchHistograms', settings.sdMatchHistograms ? 'true' : 'false')
  fd.append('sdScale', (settings.sdScale / 100).toString())

  fd.append('cv2Radius', settings.cv2Radius.toString())
  fd.append('cv2Flag', settings.cv2Flag.toString())

  fd.append('paintByExampleSteps', settings.paintByExampleSteps.toString())
  fd.append(
    'paintByExampleGuidanceScale',
    settings.paintByExampleGuidanceScale.toString()
  )
  fd.append('paintByExampleSeed', seed ? seed.toString() : '-1')
  fd.append(
    'paintByExampleMaskBlur',
    settings.paintByExampleMaskBlur.toString()
  )
  fd.append(
    'paintByExampleMatchHistograms',
    settings.paintByExampleMatchHistograms ? 'true' : 'false'
  )
  // TODO: resize image's shortest_edge to 224 before pass to backend, save network time?
  // https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPImageProcessor
  if (paintByExampleImage) {
    fd.append('paintByExampleImage', paintByExampleImage)
  }

  // InstructPix2Pix
  fd.append('p2pSteps', settings.p2pSteps.toString())
  fd.append('p2pImageGuidanceScale', settings.p2pImageGuidanceScale.toString())
  fd.append('p2pGuidanceScale', settings.p2pGuidanceScale.toString())

  if (sizeLimit === undefined) {
    fd.append('sizeLimit', '1080')
  } else {
    fd.append('sizeLimit', sizeLimit)
  }

  try {
    const res = await fetch(`${API_ENDPOINT}/inpaint`, {
      method: 'POST',
      body: fd,
    })
    if (res.ok) {
      const blob = await res.blob()
      const newSeed = res.headers.get('x-seed')
      return { blob: URL.createObjectURL(blob), seed: newSeed }
    }
    const errMsg = await res.text()
    throw new Error(errMsg)
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

export function getIsDisableModelSwitch() {
  return fetch(`${API_ENDPOINT}/is_disable_model_switch`, {
    method: 'GET',
  })
}

export function getEnableFileManager() {
  return fetch(`${API_ENDPOINT}/is_enable_file_manager`, {
    method: 'GET',
  })
}

export function switchModel(name: string) {
  const fd = new FormData()
  fd.append('name', name)
  return fetch(`${API_ENDPOINT}/model`, {
    method: 'POST',
    body: fd,
  })
}

export function currentModel() {
  return fetch(`${API_ENDPOINT}/model`, {
    method: 'GET',
  })
}

export function isDesktop() {
  return fetch(`${API_ENDPOINT}/is_desktop`, {
    method: 'GET',
  })
}

export function modelDownloaded(name: string) {
  return fetch(`${API_ENDPOINT}/model_downloaded/${name}`, {
    method: 'GET',
  })
}

export async function postInteractiveSeg(
  imageFile: File,
  maskFile: File | null,
  clicks: number[][]
) {
  const fd = new FormData()
  fd.append('image', imageFile)
  fd.append('clicks', JSON.stringify(clicks))
  if (maskFile !== null) {
    fd.append('mask', maskFile)
  }

  try {
    const res = await fetch(`${API_ENDPOINT}/interactive_seg`, {
      method: 'POST',
      body: fd,
    })
    if (res.ok) {
      const blob = await res.blob()
      return { blob: URL.createObjectURL(blob) }
    }
    const errMsg = await res.text()
    throw new Error(errMsg)
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

export async function getMediaFile(tab: string, filename: string) {
  const res = await fetch(
    `${API_ENDPOINT}/media/${tab}/${encodeURIComponent(filename)}`,
    {
      method: 'GET',
    }
  )
  if (res.ok) {
    const blob = await res.blob()
    const file = new File([blob], filename)
    return file
  }
  const errMsg = await res.text()
  throw new Error(errMsg)
}

export async function getMedias(tab: string) {
  const res = await fetch(`${API_ENDPOINT}/medias/${tab}`, {
    method: 'GET',
  })
  if (res.ok) {
    const filenames = await res.json()
    return filenames
  }
  const errMsg = await res.text()
  throw new Error(errMsg)
}

export async function downloadToOutput(
  image: HTMLImageElement,
  filename: string,
  mimeType: string
) {
  const file = await srcToFile(image.src, filename, mimeType)
  const fd = new FormData()
  fd.append('image', file)
  fd.append('filename', filename)

  try {
    const res = await fetch(`${API_ENDPOINT}/save_image`, {
      method: 'POST',
      body: fd,
    })
    if (!res.ok) {
      const errMsg = await res.text()
      throw new Error(errMsg)
    }
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

export async function makeGif(
  originFile: File,
  cleanImage: HTMLImageElement,
  filename: string,
  mimeType: string
) {
  const cleanFile = await srcToFile(cleanImage.src, filename, mimeType)
  const fd = new FormData()
  fd.append('origin_img', originFile)
  fd.append('clean_img', cleanFile)
  fd.append('filename', filename)

  try {
    const res = await fetch(`${API_ENDPOINT}/make_gif`, {
      method: 'POST',
      body: fd,
    })
    if (!res.ok) {
      const errMsg = await res.text()
      throw new Error(errMsg)
    }

    const blob = await res.blob()
    const newImage = new Image()
    await loadImage(newImage, URL.createObjectURL(blob))
    return newImage
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

import { Settings } from '../store/Atoms'
import { dataURItoBlob } from '../utils'

export const API_ENDPOINT = `${process.env.REACT_APP_INPAINTING_URL}`

export default async function inpaint(
  imageFile: File,
  maskBase64: string,
  settings: Settings,
  sizeLimit?: string
) {
  // 1080, 2000, Original
  const fd = new FormData()
  fd.append('image', imageFile)
  const mask = dataURItoBlob(maskBase64)
  fd.append('mask', mask)

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

  if (sizeLimit === undefined) {
    fd.append('sizeLimit', '1080')
  } else {
    fd.append('sizeLimit', sizeLimit)
  }

  const res = await fetch(`${API_ENDPOINT}/inpaint`, {
    method: 'POST',
    body: fd,
  }).then(async r => {
    if (r.ok) {
      return r.blob()
    }
    throw new Error('Something went wrong on server side.')
  })

  return URL.createObjectURL(res)
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

export function modelDownloaded(name: string) {
  return fetch(`${API_ENDPOINT}/model_downloaded/${name}`, {
    method: 'GET',
  })
}

import { Setting } from '../store/Atoms'
import { dataURItoBlob } from '../utils'

export const API_ENDPOINT = `${process.env.REACT_APP_INPAINTING_URL}`

export default async function inpaint(
  imageFile: File,
  maskBase64: string,
  settings: Setting,
  sizeLimit?: string
) {
  // 1080, 2000, Original
  const fd = new FormData()
  fd.append('image', imageFile)
  const mask = dataURItoBlob(maskBase64)
  fd.append('mask', mask)

  fd.append('ldmSteps', settings.ldmSteps.toString())
  fd.append('hdStrategy', settings.hdStrategy)
  fd.append('hdStrategyCropMargin', settings.hdStrategyCropMargin.toString())
  fd.append(
    'hdStrategyCropTrigerSize',
    settings.hdStrategyCropTrigerSize.toString()
  )
  fd.append('hdStrategyResizeLimit', settings.hdStrategyResizeLimit.toString())

  if (sizeLimit === undefined) {
    fd.append('sizeLimit', '1080')
  } else {
    fd.append('sizeLimit', sizeLimit)
  }

  const res = await fetch(`${API_ENDPOINT}/inpaint`, {
    method: 'POST',
    body: fd,
  }).then(async r => {
    return r.blob()
  })

  return URL.createObjectURL(res)
}

export function switchModel(name: string) {
  const fd = new FormData()
  fd.append('name', name)
  return fetch(`${API_ENDPOINT}/switch_model`, {
    method: 'POST',
    body: fd,
  })
}

import { dataURItoBlob } from '../utils'

export const API_ENDPOINT = `${process.env.REACT_APP_INPAINTING_URL}/inpaint`

export default async function inpaint(
  imageFile: File,
  maskBase64: string,
  sizeLimit?: string
) {
  // 1080, 2000, Original
  const fd = new FormData()
  fd.append('image', imageFile)
  const mask = dataURItoBlob(maskBase64)
  fd.append('mask', mask)

  if (sizeLimit === undefined) {
    fd.append('sizeLimit', '1080')
  } else {
    fd.append('sizeLimit', sizeLimit)
  }

  const res = await fetch(API_ENDPOINT, {
    method: 'POST',
    body: fd,
  }).then(async r => {
    return r.blob()
  })

  return URL.createObjectURL(res)
}

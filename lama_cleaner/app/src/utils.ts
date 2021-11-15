import { useEffect, useState } from 'react'

export function dataURItoBlob(dataURI: string) {
  const mime = dataURI.split(',')[0].split(':')[1].split(';')[0]
  const binary = atob(dataURI.split(',')[1])
  const array = []
  for (let i = 0; i < binary.length; i += 1) {
    array.push(binary.charCodeAt(i))
  }
  return new Blob([new Uint8Array(array)], { type: mime })
}

// const dataURItoBlob = (dataURI: string) => {
//   const bytes =
//     dataURI.split(',')[0].indexOf('base64') >= 0
//       ? atob(dataURI.split(',')[1])
//       : unescape(dataURI.split(',')[1])
//   const mime = dataURI.split(',')[0].split(':')[1].split(';')[0]
//   const max = bytes.length
//   const ia = new Uint8Array(max)
//   for (var i = 0; i < max; i++) ia[i] = bytes.charCodeAt(i)
//   return new Blob([ia], { type: mime })
// }

export function downloadImage(uri: string, name: string) {
  const link = document.createElement('a')
  link.href = uri
  link.download = name

  // this is necessary as link.click() does not work on the latest firefox
  link.dispatchEvent(
    new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      view: window,
    })
  )

  setTimeout(() => {
    // For Firefox it is necessary to delay revoking the ObjectURL
    // window.URL.revokeObjectURL(base64)
    link.remove()
  }, 100)
}

export function shareImage(base64: string, name: string) {
  const blob = dataURItoBlob(base64)
  const filesArray = [new File([blob], name, { type: 'image/jpeg' })]
  const shareData = {
    files: filesArray,
  }
  // eslint-disable-nextline
  const nav: any = navigator
  const canShare = nav.canShare && nav.canShare(shareData)
  const userAgent = navigator.userAgent || navigator.vendor
  const isMobile = /android|iPad|iPhone|iPod/i.test(userAgent)
  if (canShare && isMobile) {
    navigator.share(shareData)
    return true
  }
  return false
}

export function loadImage(image: HTMLImageElement, src: string) {
  return new Promise((resolve, reject) => {
    const initSRC = image.src
    const img = image
    img.onload = resolve
    img.onerror = err => {
      img.src = initSRC
      reject(err)
    }
    img.src = src
  })
}

export function useImage(file: File): [HTMLImageElement, boolean] {
  const [image] = useState(new Image())
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    image.onload = () => {
      setIsLoaded(true)
    }
    setIsLoaded(false)
    image.src = URL.createObjectURL(file)
    return () => {
      image.onload = null
    }
  }, [file, image])

  return [image, isLoaded]
}

// https://stackoverflow.com/questions/23945494/use-html5-to-resize-an-image-before-upload
interface ResizeImageFileResult {
  file: File
  resized: boolean
  originalWidth?: number
  originalHeight?: number
}
export function resizeImageFile(
  file: File,
  maxSize: number
): Promise<ResizeImageFileResult> {
  const reader = new FileReader()
  const image = new Image()
  const canvas = document.createElement('canvas')

  const resize = (): ResizeImageFileResult => {
    let { width, height } = image

    if (width > height) {
      if (width > maxSize) {
        height *= maxSize / width
        width = maxSize
      }
    } else if (height > maxSize) {
      width *= maxSize / height
      height = maxSize
    }

    if (width === image.width && height === image.height) {
      return { file, resized: false }
    }

    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      throw new Error('could not get context')
    }
    canvas.getContext('2d')?.drawImage(image, 0, 0, width, height)
    const dataUrl = canvas.toDataURL('image/jpeg')
    const blob = dataURItoBlob(dataUrl)
    const f = new File([blob], file.name, {
      type: file.type,
    })
    return {
      file: f,
      resized: true,
      originalWidth: image.width,
      originalHeight: image.height,
    }
  }

  return new Promise((resolve, reject) => {
    if (!file.type.match(/image.*/)) {
      reject(new Error('Not an image'))
      return
    }
    reader.onload = (readerEvent: any) => {
      image.onload = () => resolve(resize())
      image.src = readerEvent.target.result
    }
    reader.readAsDataURL(file)
  })
}

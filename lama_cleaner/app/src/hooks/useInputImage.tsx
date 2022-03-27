import { useCallback, useEffect, useState } from 'react'

export default function useInputImage() {
  const [inputImage, setInputImage] = useState<File>()

  const fetchInputImage = useCallback(() => {
    const headers = new Headers()
    headers.append('pragma', 'no-cache')
    headers.append('cache-control', 'no-cache')

    fetch('/inputimage', { headers })
      .then(res => res.blob())
      .then(data => {
        if (data && data.type.startsWith('image')) {
          const userInput = new File([data], 'inputImage')
          setInputImage(userInput)
        }
      })
  }, [setInputImage])

  useEffect(() => {
    fetchInputImage()
  }, [fetchInputImage])

  return inputImage
}

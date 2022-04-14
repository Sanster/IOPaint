import { useCallback, useEffect, useState } from 'react'

export default function useInputImage() {
  const [inputImage, setInputImage] = useState<File>()

  const fetchInputImage = useCallback(() => {
    const headers = new Headers()
    headers.append('pragma', 'no-cache')
    headers.append('cache-control', 'no-cache')

    fetch('/inputimage', { headers }).then(async res => {
      const filename = res.headers
        .get('content-disposition')
        ?.split('filename=')[1]
        .split(';')[0]

      const data = await res.blob()
      if (data && data.type.startsWith('image')) {
        const userInput = new File(
          [data],
          filename !== undefined ? filename : 'inputImage'
        )
        setInputImage(userInput)
      }
    })
  }, [setInputImage])

  useEffect(() => {
    fetchInputImage()
  }, [fetchInputImage])

  return inputImage
}

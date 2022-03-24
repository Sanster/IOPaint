import { useCallback, useEffect, useState } from 'react'

export default function useInputImage() {
  const [inputImage, setInputImage] = useState()

  const fetchInputImage = useCallback(() => {
    fetch('/inputimage')
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
  }, [])

  return inputImage
}

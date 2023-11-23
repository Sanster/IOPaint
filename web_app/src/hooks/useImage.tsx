import { useEffect, useState } from "react"

function useImage(file: File | null): [HTMLImageElement, boolean] {
  const [image] = useState(new Image())
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    if (!file) {
      return
    }
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

export { useImage }

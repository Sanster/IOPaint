import { useCallback, useEffect, useState } from 'react'

const useResolution = () => {
  const [width, setWidth] = useState(window.innerWidth)

  const windowSizeHandler = useCallback(() => {
    setWidth(window.innerWidth)
  }, [])

  useEffect(() => {
    window.addEventListener('resize', windowSizeHandler)

    return () => {
      window.removeEventListener('resize', windowSizeHandler)
    }
  })

  if (width < 768) {
    return 'mobile'
  }

  if (width >= 768 && width < 1224) {
    return 'tablet'
  }

  if (width >= 1224) {
    return 'desktop'
  }
}

export default useResolution

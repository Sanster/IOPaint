import React from 'react'
import { useRecoilValue } from 'recoil'
import { imageHeightState, imageWidthState } from '../../store/Atoms'

const ImageSize = () => {
  const imageWidth = useRecoilValue(imageWidthState)
  const imageHeight = useRecoilValue(imageHeightState)

  if (!imageWidth || !imageHeight) {
    return null
  }

  return (
    <div className="imageSize">
      {imageWidth}x{imageHeight}
    </div>
  )
}

export default ImageSize

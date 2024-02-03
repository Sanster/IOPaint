import { useStore } from "@/lib/states"

const ImageSize = () => {
  const [imageWidth, imageHeight] = useStore((state) => [
    state.imageWidth,
    state.imageHeight,
  ])

  if (!imageWidth || !imageHeight) {
    return null
  }

  return (
    <div className="border rounded-lg px-2 py-[6px] z-10 bg-background">
      {imageWidth}x{imageHeight}
    </div>
  )
}

export default ImageSize

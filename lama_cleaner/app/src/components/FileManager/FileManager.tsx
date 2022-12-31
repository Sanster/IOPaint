import React, { ReactNode, useEffect, useMemo, useState } from 'react'
import PhotoAlbum, { RenderPhoto } from 'react-photo-album'
import Modal from '../shared/Modal'

interface Photo {
  src: string
  height: number
  width: number
}

interface Filename {
  name: string
  height: number
  width: number
}

const renderPhoto: RenderPhoto = ({
  layout,
  layoutOptions,
  imageProps: { alt, style, ...restImageProps },
}) => (
  <div
    style={{
      boxSizing: 'content-box',
      alignItems: 'center',
    }}
  >
    <img
      alt={alt}
      style={{ ...style, width: '100%', padding: 0 }}
      {...restImageProps}
    />
  </div>
)

interface Props {
  show: boolean
  onClose: () => void
  onPhotoClick: (filename: string) => void
}

export default function FileManager(props: Props) {
  const { show, onClose, onPhotoClick } = props
  const [filenames, setFileNames] = useState<Filename[]>([])

  const onClick = ({ index }: { index: number }) => {
    onPhotoClick(filenames[index].name)
  }

  useEffect(() => {
    const fetchData = async () => {
      const res = await fetch('/medias')
      if (res.ok) {
        const newFilenames = await res.json()
        setFileNames(newFilenames)
      }
    }

    fetchData()
  }, [])

  const photos = useMemo(() => {
    return filenames.map((filename: Filename) => {
      const width = 256
      const height = filename.height * (width / filename.width)
      const src = `/media_thumbnail/${filename.name}?width=${width}&height=${height}`
      return { src, height, width }
    })
  }, [filenames])

  return (
    <Modal
      onClose={onClose}
      title="Files"
      className="file-manager-modal"
      show={show}
    >
      <div className="file-manager">
        <PhotoAlbum
          layout="columns"
          photos={photos}
          renderPhoto={renderPhoto}
          spacing={6}
          padding={4}
          onClick={onClick}
        />
      </div>
    </Modal>
  )
}

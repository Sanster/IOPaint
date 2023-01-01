import React, {
  SyntheticEvent,
  useEffect,
  useMemo,
  useState,
  useCallback,
} from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import PhotoAlbum, { RenderPhoto } from 'react-photo-album'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import Modal from '../shared/Modal'
import { toastState } from '../../store/Atoms'
import { getMedias } from '../../adapters/inpainting'

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
  onPhotoClick(filename: string): void
  photoWidth: number
}

export default function FileManager(props: Props) {
  const { show, onClose, onPhotoClick, photoWidth } = props
  const [filenames, setFileNames] = useState<Filename[]>([])
  const [scrollTop, setScrollTop] = useState(0)
  const [closeScrollTop, setCloseScrollTop] = useState(0)
  const [toastVal, setToastState] = useRecoilState(toastState)

  useEffect(() => {
    if (!show) {
      setCloseScrollTop(scrollTop)
    }
  }, [show, scrollTop])

  const onRefChange = useCallback(
    (node: HTMLDivElement) => {
      if (node !== null) {
        if (show) {
          setTimeout(() => {
            // TODO: without timeout, scrollTo not work, why?
            node.scrollTo({ top: closeScrollTop, left: 0 })
          }, 100)
        }
      }
    },
    [show, closeScrollTop]
  )

  const onClick = ({ index }: { index: number }) => {
    onPhotoClick(filenames[index].name)
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        const newFilenames = await getMedias()
        setFileNames(newFilenames)
      } catch (e: any) {
        setToastState({
          open: true,
          desc: e.message ? e.message : e.toString(),
          state: 'error',
          duration: 2000,
        })
      }
    }
    if (show) {
      fetchData()
    }
  }, [show])

  const onScroll = (event: SyntheticEvent) => {
    setScrollTop(event.currentTarget.scrollTop)
  }

  const photos = useMemo(() => {
    return filenames.map((filename: Filename) => {
      const width = photoWidth
      const height = filename.height * (width / filename.width)
      const src = `/media_thumbnail/${filename.name}?width=${width}&height=${height}`
      return { src, height, width }
    })
  }, [filenames])

  return (
    <Modal
      onClose={onClose}
      title={`Files(${photos.length})`}
      className="file-manager-modal"
      show={show}
    >
      <ScrollArea.Root className="ScrollAreaRoot">
        <ScrollArea.Viewport
          className="ScrollAreaViewport"
          onScroll={onScroll}
          ref={onRefChange}
        >
          <PhotoAlbum
            layout="columns"
            photos={photos}
            renderPhoto={renderPhoto}
            spacing={8}
            padding={8}
            onClick={onClick}
          />
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar
          className="ScrollAreaScrollbar"
          orientation="vertical"
        >
          <ScrollArea.Thumb className="ScrollAreaThumb" />
        </ScrollArea.Scrollbar>
        <ScrollArea.Scrollbar
          className="ScrollAreaScrollbar"
          orientation="horizontal"
        >
          <ScrollArea.Thumb className="ScrollAreaThumb" />
        </ScrollArea.Scrollbar>
        <ScrollArea.Corner className="ScrollAreaCorner" />
      </ScrollArea.Root>
    </Modal>
  )
}

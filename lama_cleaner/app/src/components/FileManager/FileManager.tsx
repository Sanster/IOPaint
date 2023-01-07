import React, {
  SyntheticEvent,
  useEffect,
  useMemo,
  useState,
  useCallback,
  useRef,
  FormEvent,
} from 'react'
import _ from 'lodash'
import { useSetRecoilState } from 'recoil'
import PhotoAlbum from 'react-photo-album'
import { BarsArrowDownIcon, BarsArrowUpIcon } from '@heroicons/react/24/outline'
import { MagnifyingGlassIcon } from '@radix-ui/react-icons'
import { useDebounce } from 'react-use'
import { Id, Index, IndexSearchResult } from 'flexsearch'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import Modal from '../shared/Modal'
import Flex from '../shared/Layout'
import { toastState } from '../../store/Atoms'
import { getMedias } from '../../adapters/inpainting'
import Selector from '../shared/Selector'
import Button from '../shared/Button'
import TextInput from '../shared/Input'
import { useAsyncMemo } from '../../hooks/useAsyncMemo'

interface Photo {
  src: string
  height: number
  width: number
}

interface Filename {
  name: string
  height: number
  width: number
  ctime: number
}

enum SortOrder {
  DESCENDING = 'desc',
  ASCENDING = 'asc',
}

enum SortBy {
  NAME = 'name',
  CTIME = 'ctime',
}

const SORT_BY_NAME = 'Name'
const SORT_BY_CREATED_TIME = 'Created time'

const SortByMap = {
  [SortBy.NAME]: SORT_BY_NAME,
  [SortBy.CTIME]: SORT_BY_CREATED_TIME,
}

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
  const setToastState = useSetRecoilState(toastState)
  const [sortBy, setSortBy] = useState<SortBy>(SortBy.CTIME)
  const [sortOrder, setSortOrder] = useState<SortOrder>(SortOrder.DESCENDING)
  const ref = useRef(null)
  const [searchText, setSearchText] = useState('')
  const [debouncedSearchText, setDebouncedSearchText] = useState('')

  const [, cancel] = useDebounce(
    () => {
      setDebouncedSearchText(searchText)
    },
    500,
    [searchText]
  )

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
  }, [show, setToastState])

  const onScroll = (event: SyntheticEvent) => {
    setScrollTop(event.currentTarget.scrollTop)
  }

  const filteredFilenames: Filename[] | undefined = useAsyncMemo(async () => {
    if (!debouncedSearchText) {
      return filenames
    }

    const index = new Index()
    filenames.forEach((filename: Filename, id: number) =>
      index.add(id, filename.name)
    )
    const results: IndexSearchResult = await index.searchAsync(
      debouncedSearchText
    )
    return results.map((id: Id) => filenames[id as number])
  }, [filenames, debouncedSearchText])

  const photos: Photo[] = useMemo(() => {
    return _.orderBy(filteredFilenames, sortBy, sortOrder).map(
      (filename: Filename) => {
        const width = photoWidth
        const height = filename.height * (width / filename.width)
        const src = `/media_thumbnail/${filename.name}?width=${width}&height=${height}`
        return { src, height, width }
      }
    )
  }, [filteredFilenames, photoWidth, sortBy, sortOrder])

  return (
    <Modal
      onClose={onClose}
      title={`Images (${photos.length})`}
      className="file-manager-modal"
      show={show}
    >
      <Flex style={{ justifyContent: 'end', gap: 8 }}>
        <Flex
          style={{
            position: 'relative',
            justifyContent: 'start',
          }}
        >
          <MagnifyingGlassIcon style={{ position: 'absolute', left: 8 }} />
          <TextInput
            ref={ref}
            value={searchText}
            className="file-search-input"
            tabIndex={-1}
            onInput={(evt: FormEvent<HTMLInputElement>) => {
              evt.preventDefault()
              evt.stopPropagation()
              const target = evt.target as HTMLInputElement
              setSearchText(target.value)
            }}
            placeholder="Search by file name"
          />
        </Flex>
        <Flex style={{ gap: 8 }}>
          <Selector
            width={130}
            value={SortByMap[sortBy]}
            options={Object.values(SortByMap)}
            onChange={val => {
              if (val === SORT_BY_CREATED_TIME) {
                setSortBy(SortBy.CTIME)
              } else {
                setSortBy(SortBy.NAME)
              }
            }}
            chevronDirection="down"
          />
          <Button
            icon={<BarsArrowDownIcon />}
            toolTip="Descending order"
            onClick={() => {
              setSortOrder(SortOrder.DESCENDING)
            }}
            className={
              sortOrder !== SortOrder.DESCENDING ? 'sort-btn-inactive' : ''
            }
          />
          <Button
            icon={<BarsArrowUpIcon />}
            toolTip="Ascending order"
            onClick={() => {
              setSortOrder(SortOrder.ASCENDING)
            }}
            className={
              sortOrder !== SortOrder.ASCENDING ? 'sort-btn-inactive' : ''
            }
          />
        </Flex>
      </Flex>
      <ScrollArea.Root className="ScrollAreaRoot">
        <ScrollArea.Viewport
          className="ScrollAreaViewport"
          onScroll={onScroll}
          ref={onRefChange}
        >
          <PhotoAlbum
            layout="masonry"
            photos={photos}
            spacing={8}
            padding={0}
            onClick={onClick}
          />
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar
          className="ScrollAreaScrollbar"
          orientation="vertical"
        >
          <ScrollArea.Thumb className="ScrollAreaThumb" />
        </ScrollArea.Scrollbar>
        {/* <ScrollArea.Scrollbar
          className="ScrollAreaScrollbar"
          orientation="horizontal"
        >
          <ScrollArea.Thumb className="ScrollAreaThumb" />
        </ScrollArea.Scrollbar> */}
        <ScrollArea.Corner className="ScrollAreaCorner" />
      </ScrollArea.Root>
    </Modal>
  )
}

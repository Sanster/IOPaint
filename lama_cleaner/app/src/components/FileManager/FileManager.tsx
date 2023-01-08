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
import * as Tabs from '@radix-ui/react-tabs'
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
  name: string
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

const IMAGE_TAB = 'image'
const OUTPUT_TAB = 'output'

const SortByMap = {
  [SortBy.NAME]: SORT_BY_NAME,
  [SortBy.CTIME]: SORT_BY_CREATED_TIME,
}

interface Props {
  show: boolean
  onClose: () => void
  onPhotoClick(tab: string, filename: string): void
  photoWidth: number
}

export default function FileManager(props: Props) {
  const { show, onClose, onPhotoClick, photoWidth } = props
  const [scrollTop, setScrollTop] = useState(0)
  const [closeScrollTop, setCloseScrollTop] = useState(0)
  const setToastState = useSetRecoilState(toastState)
  const [sortBy, setSortBy] = useState<SortBy>(SortBy.CTIME)
  const [sortOrder, setSortOrder] = useState<SortOrder>(SortOrder.DESCENDING)
  const ref = useRef(null)
  const [searchText, setSearchText] = useState('')
  const [debouncedSearchText, setDebouncedSearchText] = useState('')
  const [tab, setTab] = useState(IMAGE_TAB)
  const [photos, setPhotos] = useState<Photo[]>([])

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

  useEffect(() => {
    if (!show) {
      return
    }
    const fetchData = async () => {
      try {
        const filenames = await getMedias(tab)
        let filteredFilenames = filenames
        if (debouncedSearchText) {
          const index = new Index()
          filenames.forEach((filename: Filename, id: number) =>
            index.add(id, filename.name)
          )
          const results: IndexSearchResult = index.search(debouncedSearchText)
          filteredFilenames = results.map((id: Id) => filenames[id as number])
        }

        filteredFilenames = _.orderBy(filteredFilenames, sortBy, sortOrder)

        const newPhotos = filteredFilenames.map((filename: Filename) => {
          const width = photoWidth
          const height = filename.height * (width / filename.width)
          const src = `/media_thumbnail/${tab}/${filename.name}?width=${width}&height=${height}`
          return { src, height, width, name: filename.name }
        })
        setPhotos(newPhotos)
      } catch (e: any) {
        setToastState({
          open: true,
          desc: e.message ? e.message : e.toString(),
          state: 'error',
          duration: 2000,
        })
      }
    }
    fetchData()
  }, [
    setToastState,
    tab,
    debouncedSearchText,
    sortBy,
    sortOrder,
    photoWidth,
    show,
  ])

  const onScroll = (event: SyntheticEvent) => {
    setScrollTop(event.currentTarget.scrollTop)
  }

  const onClick = ({ index }: { index: number }) => {
    onPhotoClick(tab, photos[index].name)
  }

  return (
    <Modal
      onClose={onClose}
      title={`Images (${photos.length})`}
      className="file-manager-modal"
      show={show}
    >
      <Flex style={{ justifyContent: 'space-between', gap: 8 }}>
        <Tabs.Root
          className="TabsRoot"
          defaultValue={tab}
          onValueChange={val => setTab(val)}
        >
          <Tabs.List className="TabsList" aria-label="Manage your account">
            <Tabs.Trigger className="TabsTrigger" value={IMAGE_TAB}>
              Image Directory
            </Tabs.Trigger>
            <Tabs.Trigger className="TabsTrigger" value={OUTPUT_TAB}>
              Output Directory
            </Tabs.Trigger>
          </Tabs.List>
        </Tabs.Root>
        <Flex style={{ gap: 8 }}>
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

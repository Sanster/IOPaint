import {
  SyntheticEvent,
  useEffect,
  useState,
  useCallback,
  useRef,
  FormEvent,
} from "react"
import _ from "lodash"
import { useRecoilState } from "recoil"
import PhotoAlbum from "react-photo-album"
import {
  BarsArrowDownIcon,
  BarsArrowUpIcon,
  FolderIcon,
} from "@heroicons/react/24/outline"
import {
  MagnifyingGlassIcon,
  ViewHorizontalIcon,
  ViewGridIcon,
} from "@radix-ui/react-icons"
import { useDebounce, useToggle } from "react-use"
import FlexSearch from "flexsearch/dist/flexsearch.bundle.js"
import {
  fileManagerLayout,
  fileManagerSearchText,
  fileManagerSortBy,
  fileManagerSortOrder,
  SortBy,
  SortOrder,
} from "@/lib/store"
import { useToast } from "@/components/ui/use-toast"
import { API_ENDPOINT, getMedias } from "@/lib/api"
import { IconButton } from "./ui/button"
import { Input } from "./ui/input"
import { Dialog, DialogContent, DialogTitle } from "./ui/dialog"
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"
import { ScrollArea } from "./ui/scroll-area"
import { DialogTrigger } from "@radix-ui/react-dialog"
import { useHotkeys } from "react-hotkeys-hook"

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
  mtime: number
}

const SORT_BY_NAME = "Name"
const SORT_BY_CREATED_TIME = "Created time"
const SORT_BY_MODIFIED_TIME = "Modified time"

const IMAGE_TAB = "image"
const OUTPUT_TAB = "output"

const SortByMap = {
  [SortBy.NAME]: SORT_BY_NAME,
  [SortBy.CTIME]: SORT_BY_CREATED_TIME,
  [SortBy.MTIME]: SORT_BY_MODIFIED_TIME,
}

interface Props {
  onPhotoClick(tab: string, filename: string): void
  photoWidth: number
}

export default function FileManager(props: Props) {
  const { onPhotoClick, photoWidth } = props
  const [open, toggleOpen] = useToggle(false)

  useHotkeys("f", () => {
    toggleOpen()
  })

  const { toast } = useToast()
  const [scrollTop, setScrollTop] = useState(0)
  const [closeScrollTop, setCloseScrollTop] = useState(0)

  const [sortBy, setSortBy] = useRecoilState<SortBy>(fileManagerSortBy)
  const [sortOrder, setSortOrder] = useRecoilState(fileManagerSortOrder)
  const [layout, setLayout] = useRecoilState(fileManagerLayout)
  const [debouncedSearchText, setDebouncedSearchText] = useRecoilState(
    fileManagerSearchText
  )
  const ref = useRef(null)
  const [searchText, setSearchText] = useState(debouncedSearchText)
  const [tab, setTab] = useState(IMAGE_TAB)
  const [photos, setPhotos] = useState<Photo[]>([])

  const [, cancel] = useDebounce(
    () => {
      setDebouncedSearchText(searchText)
    },
    300,
    [searchText]
  )

  useEffect(() => {
    if (!open) {
      setCloseScrollTop(scrollTop)
    }
  }, [open, scrollTop])

  const onRefChange = useCallback(
    (node: HTMLDivElement) => {
      if (node !== null) {
        if (open) {
          setTimeout(() => {
            // TODO: without timeout, scrollTo not work, why?
            node.scrollTo({ top: closeScrollTop, left: 0 })
          }, 100)
        }
      }
    },
    [open, closeScrollTop]
  )

  useEffect(() => {
    if (!open) {
      return
    }
    const fetchData = async () => {
      try {
        const filenames = await getMedias(tab)
        let filteredFilenames = filenames
        if (debouncedSearchText) {
          const index = new FlexSearch.Index({
            tokenize: "forward",
            minlength: 1,
          })
          filenames.forEach((filename: Filename, id: number) =>
            index.add(id, filename.name)
          )
          const results: FlexSearch.IndexSearchResult =
            index.search(debouncedSearchText)
          filteredFilenames = results.map(
            (id: FlexSearch.Id) => filenames[id as number]
          )
        }

        filteredFilenames = _.orderBy(filteredFilenames, sortBy, sortOrder)

        const newPhotos = filteredFilenames.map((filename: Filename) => {
          const width = photoWidth
          const height = filename.height * (width / filename.width)
          const src = `${API_ENDPOINT}/media_thumbnail/${tab}/${filename.name}?width=${width}&height=${height}`
          return { src, height, width, name: filename.name }
        })
        setPhotos(newPhotos)
      } catch (e: any) {
        toast({
          variant: "destructive",
          title: "Uh oh! Something went wrong.",
          description: e.message ? e.message : e.toString(),
        })
      }
    }
    fetchData()
  }, [tab, debouncedSearchText, sortBy, sortOrder, photoWidth, open])

  const onScroll = (event: SyntheticEvent) => {
    setScrollTop(event.currentTarget.scrollTop)
  }

  const onClick = ({ index }: { index: number }) => {
    toggleOpen()
    onPhotoClick(tab, photos[index].name)
  }

  const renderTitle = () => {
    return (
      <div className="flex justify-start items-center gap-[12px]">
        <div>{`Images (${photos.length})`}</div>
        <div className="flex">
          <IconButton
            tooltip="Rows layout"
            onClick={() => {
              setLayout("rows")
            }}
          >
            <ViewHorizontalIcon
              className={layout !== "rows" ? "opacity-50" : ""}
            />
          </IconButton>
          <IconButton
            tooltip="Grid layout"
            onClick={() => {
              setLayout("masonry")
            }}
            className={layout !== "masonry" ? "opacity-50" : ""}
          >
            <ViewGridIcon />
          </IconButton>
        </div>
      </div>
    )
  }

  return (
    <Dialog open={open} onOpenChange={toggleOpen}>
      <DialogTrigger>
        <IconButton tooltip="File Manager">
          <FolderIcon />
        </IconButton>
      </DialogTrigger>
      <DialogContent className="h-4/5 max-w-6xl">
        <DialogTitle>{renderTitle()}</DialogTitle>
        <div className="flex justify-between gap-8 items-center">
          <div className="flex relative justify-start items-center">
            <MagnifyingGlassIcon className="absolute left-[8px]" />
            <Input
              ref={ref}
              value={searchText}
              className="w-[250px] pl-[30px]"
              tabIndex={-1}
              onInput={(evt: FormEvent<HTMLInputElement>) => {
                evt.preventDefault()
                evt.stopPropagation()
                const target = evt.target as HTMLInputElement
                setSearchText(target.value)
              }}
              placeholder="Search by file name"
            />
          </div>

          <Tabs defaultValue={tab} onValueChange={(val) => setTab(val)}>
            <TabsList aria-label="Manage your account">
              <TabsTrigger value={IMAGE_TAB}>Image Directory</TabsTrigger>
              <TabsTrigger value={OUTPUT_TAB}>Output Directory</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="flex gap-2">
            <div className="flex gap-1">
              <Select
                value={SortByMap[sortBy]}
                onValueChange={(val) => {
                  switch (val) {
                    case SORT_BY_NAME:
                      setSortBy(SortBy.NAME)
                      break
                    case SORT_BY_CREATED_TIME:
                      setSortBy(SortBy.CTIME)
                      break
                    case SORT_BY_MODIFIED_TIME:
                      setSortBy(SortBy.MTIME)
                      break
                    default:
                      break
                  }
                }}
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.values(SortByMap).map((val) => {
                    return (
                      <SelectItem value={val} key={val}>
                        {val}
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>

              {sortOrder === SortOrder.DESCENDING ? (
                <IconButton
                  tooltip="Descending Order"
                  onClick={() => {
                    setSortOrder(SortOrder.ASCENDING)
                  }}
                >
                  <BarsArrowDownIcon />
                </IconButton>
              ) : (
                <IconButton
                  tooltip="Ascending Order"
                  onClick={() => {
                    setSortOrder(SortOrder.DESCENDING)
                  }}
                >
                  <BarsArrowUpIcon />
                </IconButton>
              )}
            </div>
          </div>
        </div>

        <ScrollArea
          className="w-full h-full rounded-md"
          onScroll={onScroll}
          ref={onRefChange}
        >
          <PhotoAlbum
            layout={layout}
            photos={photos}
            spacing={12}
            padding={0}
            onClick={onClick}
          />
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}

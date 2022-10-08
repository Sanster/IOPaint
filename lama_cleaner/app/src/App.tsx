import React, { useEffect, useMemo } from 'react'
import { useRecoilState } from 'recoil'
import { nanoid } from 'nanoid'
import useInputImage from './hooks/useInputImage'
import LandingPage from './components/LandingPage/LandingPage'
import { themeState } from './components/Header/ThemeChanger'
import Workspace from './components/Workspace'
import { fileState, toastState } from './store/Atoms'
import { keepGUIAlive } from './utils'
import Header from './components/Header/Header'
import useHotKey from './hooks/useHotkey'

const SUPPORTED_FILE_TYPE = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
  'image/tiff',
]

// Keeping GUI Window Open
keepGUIAlive()

function App() {
  const [file, setFile] = useRecoilState(fileState)
  const [theme, setTheme] = useRecoilState(themeState)
  const [toastVal, setToastState] = useRecoilState(toastState)
  const userInputImage = useInputImage()

  // Set Input Image
  useEffect(() => {
    setFile(userInputImage)
  }, [userInputImage, setFile])

  // Dark Mode Hotkey
  useHotKey(
    'shift+d',
    () => {
      const newTheme = theme === 'light' ? 'dark' : 'light'
      setTheme(newTheme)
    },
    {},
    [theme]
  )

  useEffect(() => {
    document.body.setAttribute('data-theme', theme)
  }, [theme])

  const workspaceId = useMemo(() => {
    return nanoid()
  }, [file])

  ///

  const [isDragging, setIsDragging] = React.useState(false)
  const dragCounter = React.useRef(0)

  const handleDrag = React.useCallback(event => {
    event.preventDefault()
    event.stopPropagation()
  }, [])
  const handleDragIn = React.useCallback(event => {
    event.preventDefault()
    event.stopPropagation()
    dragCounter.current += 1
    if (event.dataTransfer.items && event.dataTransfer.items.length > 0) {
      setIsDragging(true)
    }
  }, [])
  const handleDragOut = React.useCallback(event => {
    event.preventDefault()
    event.stopPropagation()
    dragCounter.current -= 1
    if (dragCounter.current > 0) return
    setIsDragging(false)
  }, [])
  const handleDrop = React.useCallback(event => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      if (event.dataTransfer.files.length > 1) {
        setToastState({
          open: true,
          desc: 'Please drag and drop only one file',
          state: 'error',
          duration: 3000,
        })
      } else {
        const dragFile = event.dataTransfer.files[0]
        const fileType = dragFile.type
        if (SUPPORTED_FILE_TYPE.includes(fileType)) {
          setFile(dragFile)
        } else {
          setToastState({
            open: true,
            desc: 'Please drag and drop an image file',
            state: 'error',
            duration: 3000,
          })
        }
      }
      event.dataTransfer.clearData()
    }
  }, [])

  React.useEffect(() => {
    window.addEventListener('dragenter', handleDragIn)
    window.addEventListener('dragleave', handleDragOut)
    window.addEventListener('dragover', handleDrag)
    window.addEventListener('drop', handleDrop)
    return function cleanUp() {
      window.removeEventListener('dragenter', handleDragIn)
      window.removeEventListener('dragleave', handleDragOut)
      window.removeEventListener('dragover', handleDrag)
      window.removeEventListener('drop', handleDrop)
    }
  })

  ///

  return (
    <div className="lama-cleaner">
      <Header />
      <Workspace key={workspaceId} />
    </div>
  )
}

export default App

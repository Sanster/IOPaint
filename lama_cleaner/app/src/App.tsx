import React, { useEffect } from 'react'
import { useKeyPressEvent } from 'react-use'
import { useRecoilState } from 'recoil'
import useInputImage from './hooks/useInputImage'
import LandingPage from './components/LandingPage/LandingPage'
import { themeState } from './components/Header/ThemeChanger'
import Workspace from './components/Workspace'
import { fileState } from './store/Atoms'
import { keepGUIAlive } from './utils'
import Header from './components/Header/Header'

// Keeping GUI Window Open
keepGUIAlive()

function App() {
  const [file, setFile] = useRecoilState(fileState)
  const [theme, setTheme] = useRecoilState(themeState)
  const userInputImage = useInputImage()

  // Set Input Image
  useEffect(() => {
    setFile(userInputImage)
  }, [userInputImage, setFile])

  // Dark Mode Hotkey
  useKeyPressEvent('D', ev => {
    ev?.preventDefault()
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
  })

  return (
    <div className="lama-cleaner" data-theme={theme}>
      <Header />
      {file ? <Workspace file={file} /> : <LandingPage />}
    </div>
  )
}

export default App

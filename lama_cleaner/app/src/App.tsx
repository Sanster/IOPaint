import React, { useEffect } from 'react'
import { useKeyPressEvent } from 'react-use'
import { useRecoilState } from 'recoil'
import useInputImage from './hooks/useInputImage'
import LandingPage from './components/LandingPage/LandingPage'
import { ThemeChanger, themeState } from './components/shared/ThemeChanger'
import Workspace from './components/Workspace'
import { fileState } from './store/Atoms'

// Keeping GUI Window Open
async function getRequest(url = '') {
  const response = await fetch(url, {
    method: 'GET',
    cache: 'no-cache',
  })
  return response.json()
}

if (!process.env.NODE_ENV || process.env.NODE_ENV === 'production') {
  document.addEventListener('DOMContentLoaded', function () {
    const url = document.location
    const route = '/flaskwebgui-keep-server-alive'
    const intervalRequest = 3 * 1000
    function keepAliveServer() {
      getRequest(url + route).then(data => console.log(data))
    }
    setInterval(keepAliveServer, intervalRequest)
  })
}

function App() {
  const [file, setFile] = useRecoilState(fileState)
  const [theme, setTheme] = useRecoilState(themeState)
  const userInputImage = useInputImage()

  // Set Input Image
  useEffect(() => {
    setFile(userInputImage)
  }, [userInputImage, setFile])

  // Dark Mode Hotkey
  useKeyPressEvent('D', () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
  })

  return (
    <div className="lama-cleaner" data-theme={theme}>
      <ThemeChanger />
      {file ? <Workspace file={file} /> : <LandingPage />}
    </div>
  )
}

export default App

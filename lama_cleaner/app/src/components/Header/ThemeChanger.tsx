import React, { useEffect } from 'react'
import { atom, useRecoilState } from 'recoil'
import { SunIcon, MoonIcon } from '@heroicons/react/outline'

export const themeState = atom({
  key: 'themeState',
  default: 'dark',
})

export const ThemeChanger = () => {
  const [theme, setTheme] = useRecoilState(themeState)

  const themeSwitchHandler = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
  }

  return (
    <div className="theme-toggle-ui">
      <div
        className="theme-btn"
        onClick={themeSwitchHandler}
        role="button"
        tabIndex={0}
        aria-hidden="true"
      >
        {theme === 'light' ? (
          <MoonIcon />
        ) : (
          <SunIcon style={{ color: '#ffcc00' }} />
        )}
      </div>
    </div>
  )
}

import React from 'react'
import { atom, useRecoilState } from 'recoil'

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
    <button
      type="button"
      className="theme-changer"
      onClick={themeSwitchHandler}
      aria-label="Switch Theme"
    />
  )
}

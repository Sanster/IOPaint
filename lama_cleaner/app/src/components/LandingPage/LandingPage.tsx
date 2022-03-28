import React from 'react'
import { useSetRecoilState } from 'recoil'
import { fileState } from '../../store/Atoms'
import FileSelect from '../FileSelect/FileSelect'

const LandingPage = () => {
  const setFile = useSetRecoilState(fileState)

  return (
    <div className="landing-page">
      <h1>
        Image inpainting powered by 🦙
        <a href="https://github.com/saic-mdal/lama">LaMa</a>
      </h1>
      <div className="landing-file-selector">
        <FileSelect
          onSelection={async f => {
            setFile(f)
          }}
        />
      </div>
    </div>
  )
}

export default LandingPage

import { ArrowLeftIcon } from '@heroicons/react/outline'
import React from 'react'
import { useSetRecoilState } from 'recoil'
import { fileState } from '../../store/Atoms'
import Button from '../shared/Button'
import Shortcuts from '../Shortcuts/Shortcuts'
import useResolution from '../../hooks/useResolution'

const Header = () => {
  const setFile = useSetRecoilState(fileState)
  const resolution = useResolution()

  const renderHeader = () => {
    return (
      <header>
        <Button
          icon={<ArrowLeftIcon className="w-6 h-6" />}
          onClick={() => {
            setFile(undefined)
          }}
        >
          {resolution === 'desktop' ? 'Start New' : undefined}
        </Button>
        <Shortcuts />
      </header>
    )
  }
  return renderHeader()
}

export default Header

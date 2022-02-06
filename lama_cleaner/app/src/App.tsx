import { ArrowLeftIcon } from '@heroicons/react/outline'
import React, { useState } from 'react'
import { useWindowSize } from 'react-use'
import Button from './components/Button'
import FileSelect from './components/FileSelect'
import Editor from './Editor'

function App() {
  const [file, setFile] = useState<File>()
  const windowSize = useWindowSize()

  return (
    <div className="h-full full-visible-h-safari flex flex-col">
      <header className="absolute z-10 flex w-full p-2 justify-center sm:justify-between items-center sm:items-start  bg-white backdrop-blur backdrop-filter bg-opacity-30">
        {file ? (
          <Button
            icon={<ArrowLeftIcon className="w-6 h-6" />}
            onClick={() => {
              setFile(undefined)
            }}
          >
            {windowSize.width > 640 ? 'Start new' : undefined}
          </Button>
        ) : (
          <></>
        )}
      </header>

      <main
        className={[
          'h-full flex flex-1 flex-col sm:items-center sm:justify-center overflow-hidden',
          'items-center justify-center',
        ].join(' ')}
      >
        {file ? (
          <Editor file={file} />
        ) : (
          <>
            <div
              className={[
                'flex flex-col sm:flex-row items-center',
                'space-y-5 sm:space-y-0 sm:space-x-6 p-5 pt-0 pb-10',
              ].join(' ')}
            >
              <div className="max-w-xl flex flex-col items-center sm:items-start p-0 m-0 space-y-5">
                <h1 className="text-center sm:text-left text-xl sm:text-3xl">
                  Image inpainting powered by 🦙
                  <u>
                    <a href="https://github.com/saic-mdal/lama">LaMa</a>
                  </u>
                </h1>
              </div>
            </div>

            <div
              className="h-20 sm:h-52 px-4 w-full"
              style={{ maxWidth: '800px' }}
            >
              <FileSelect
                onSelection={async f => {
                  setFile(f)
                }}
              />
            </div>
          </>
        )}
      </main>
    </div>
  )
}

export default App

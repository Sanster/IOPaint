import * as React from 'react'
import * as Progress from '@radix-ui/react-progress'
import { useRecoilValue } from 'recoil'
import io from 'socket.io-client'
import { isInpaintingState, isSDState, settingState } from '../../store/Atoms'

const isDev = process.env.NODE_ENV === 'development'
const socket = isDev ? io(`http://localhost:8080`) : io()

const SDProgress = () => {
  const isSD = useRecoilValue(isSDState)
  const isInpainting = useRecoilValue(isInpaintingState)
  const [isConnected, setIsConnected] = React.useState(false)
  const [step, setStep] = React.useState(0)
  const setting = useRecoilValue(settingState)
  const maxStep = Math.max(setting.sdSteps, 1)

  const progress = Math.min(Math.round((step / maxStep) * 100), 100)

  React.useEffect(() => {
    if (!isSD) return

    socket.on('connect', () => {
      setIsConnected(true)
    })

    socket.on('disconnect', () => {
      setIsConnected(false)
    })

    socket.on('diffusion_progress', data => {
      console.log(`step: ${data.step + 1}`)
      if (data) {
        setStep(data.step + 1)
      }
    })

    socket.on('diffusion_finish', data => {
      setStep(0)
    })

    return () => {
      socket.off('connect')
      socket.off('disconnect')
      socket.off('diffusion_progress')
      socket.off('diffusion_finish')
    }
  }, [isSD])

  return (
    <div
      className="ProgressWrapper"
      style={{
        visibility: isInpainting && isConnected && isSD ? 'visible' : 'hidden',
      }}
    >
      <Progress.Root value={progress} className="ProgressRoot">
        <Progress.Indicator
          className="ProgressIndicator"
          style={{ transform: `translateX(-${100 - progress}%)` }}
        />
      </Progress.Root>
      <div
        style={{
          width: 45,
          display: 'flex',
          justifyContent: 'center',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {progress}%
      </div>
    </div>
  )
}

export default SDProgress

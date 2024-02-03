import * as React from "react"
import io from "socket.io-client"
import { Progress } from "./ui/progress"
import { useStore } from "@/lib/states"

export const API_ENDPOINT = import.meta.env.DEV
  ? import.meta.env.VITE_BACKEND
  : ""
const socket = io(API_ENDPOINT)

const DiffusionProgress = () => {
  const [settings, isInpainting, isSD] = useStore((state) => [
    state.settings,
    state.isInpainting,
    state.isSD(),
  ])

  const [isConnected, setIsConnected] = React.useState(false)
  const [step, setStep] = React.useState(0)

  const progress = Math.min(Math.round((step / settings.sdSteps) * 100), 100)

  React.useEffect(() => {
    socket.on("connect", () => {
      setIsConnected(true)
    })

    socket.on("disconnect", () => {
      setIsConnected(false)
    })

    socket.on("diffusion_progress", (data) => {
      if (data) {
        setStep(data.step + 1)
      }
    })

    socket.on("diffusion_finish", () => {
      setStep(0)
    })

    return () => {
      socket.off("connect")
      socket.off("disconnect")
      socket.off("diffusion_progress")
      socket.off("diffusion_finish")
    }
  }, [])

  return (
    <div
      className="z-10 fixed bg-background w-[220px] left-1/2 -translate-x-1/2 top-[68px] h-[32px] flex justify-center items-center gap-[18px] border-[1px] border-[solid] rounded-[14px] pl-[8px] pr-[8px]"
      style={{
        visibility: isConnected && isInpainting && isSD ? "visible" : "hidden",
      }}
    >
      <Progress value={progress} />
      <div className="w-[45px] flex justify-center font-nums">{progress}%</div>
    </div>
  )
}

export default DiffusionProgress

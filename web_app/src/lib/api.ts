import { Filename, ModelInfo, PowerPaintTask, Rect } from "@/lib/types"
import { Settings } from "@/lib/states"
import { convertToBase64, srcToFile } from "@/lib/utils"
import axios from "axios"

export const API_ENDPOINT = import.meta.env.VITE_BACKEND
  ? import.meta.env.VITE_BACKEND
  : "/api/v1"

const api = axios.create({
  baseURL: API_ENDPOINT,
})

export default async function inpaint(
  imageFile: File,
  settings: Settings,
  croperRect: Rect,
  extenderState: Rect,
  mask: File | Blob,
  paintByExampleImage: File | null = null
) {
  const imageBase64 = await convertToBase64(imageFile)
  const maskBase64 = await convertToBase64(mask)
  const exampleImageBase64 = paintByExampleImage
    ? await convertToBase64(paintByExampleImage)
    : null
  try {
    const res = await fetch(`${API_ENDPOINT}/inpaint`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: imageBase64,
        mask: maskBase64,
        ldm_steps: settings.ldmSteps,
        ldm_sampler: settings.ldmSampler,
        zits_wireframe: settings.zitsWireframe,
        cv2_flag: settings.cv2Flag,
        cv2_radius: settings.cv2Radius,
        hd_strategy: "Crop",
        hd_strategy_crop_triger_size: 640,
        hd_strategy_crop_margin: 128,
        hd_trategy_resize_imit: 2048,
        prompt: settings.prompt,
        negative_prompt: settings.negativePrompt,
        use_croper: settings.showCropper,
        croper_x: croperRect.x,
        croper_y: croperRect.y,
        croper_height: croperRect.height,
        croper_width: croperRect.width,
        use_extender: settings.showExtender,
        extender_x: extenderState.x,
        extender_y: extenderState.y,
        extender_height: extenderState.height,
        extender_width: extenderState.width,
        sd_mask_blur: settings.sdMaskBlur,
        sd_strength: settings.sdStrength,
        sd_steps: settings.sdSteps,
        sd_guidance_scale: settings.sdGuidanceScale,
        sd_sampler: settings.sdSampler,
        sd_seed: settings.seedFixed ? settings.seed : -1,
        sd_match_histograms: settings.sdMatchHistograms,
        sd_freeu: settings.enableFreeu,
        sd_freeu_config: settings.freeuConfig,
        sd_lcm_lora: settings.enableLCMLora,
        paint_by_example_example_image: exampleImageBase64,
        p2p_image_guidance_scale: settings.p2pImageGuidanceScale,
        enable_controlnet: settings.enableControlnet,
        controlnet_conditioning_scale: settings.controlnetConditioningScale,
        controlnet_method: settings.controlnetMethod
          ? settings.controlnetMethod
          : "",
        powerpaint_task: settings.showExtender
          ? PowerPaintTask.outpainting
          : settings.powerpaintTask,
      }),
    })
    const blob = await res.blob()
    return {
      blob: URL.createObjectURL(blob),
      seed: res.headers.get("X-Seed"),
    }
  } catch (error: any) {
    throw new Error(`Something went wrong: ${JSON.stringify(error.message)}`)
  }
}

export function getServerConfig() {
  return fetch(`${API_ENDPOINT}/server-config`, {
    method: "GET",
  })
}

export function switchModel(name: string) {
  return axios.post(`${API_ENDPOINT}/model`, { name })
}

export function currentModel() {
  return fetch(`${API_ENDPOINT}/model`, {
    method: "GET",
  })
}

export function fetchModelInfos(): Promise<ModelInfo[]> {
  return api.get("/models").then((response) => response.data)
}

export function modelDownloaded(name: string) {
  return fetch(`${API_ENDPOINT}/model_downloaded/${name}`, {
    method: "GET",
  })
}

export async function runPlugin(
  name: string,
  imageFile: File,
  upscale?: number,
  clicks?: number[][]
) {
  const fd = new FormData()
  fd.append("name", name)
  fd.append("image", imageFile)
  if (upscale) {
    fd.append("upscale", upscale.toString())
  }
  if (clicks) {
    fd.append("clicks", JSON.stringify(clicks))
  }

  try {
    const res = await fetch(`${API_ENDPOINT}/run_plugin`, {
      method: "POST",
      body: fd,
    })
    if (res.ok) {
      const blob = await res.blob()
      return { blob: URL.createObjectURL(blob) }
    }
    const errMsg = await res.text()
    throw new Error(errMsg)
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

export async function getMediaFile(tab: string, filename: string) {
  const res = await fetch(
    `${API_ENDPOINT}/media_file?tab=${tab}&filename=${encodeURIComponent(
      filename
    )}`,
    {
      method: "GET",
    }
  )
  if (res.ok) {
    const blob = await res.blob()
    const file = new File([blob], filename, {
      type: res.headers.get("Content-Type") ?? "image/png",
    })
    return file
  }
  const errMsg = await res.text()
  throw new Error(errMsg)
}

export async function getMedias(tab: string): Promise<Filename[]> {
  const res = await axios.get(`${API_ENDPOINT}/medias`, { params: { tab } })
  return res.data
}

export async function downloadToOutput(
  image: HTMLImageElement,
  filename: string,
  mimeType: string
) {
  const file = await srcToFile(image.src, filename, mimeType)
  const fd = new FormData()
  fd.append("file", file)

  try {
    const res = await fetch(`${API_ENDPOINT}/save_image`, {
      method: "POST",
      body: fd,
    })
    if (!res.ok) {
      const errMsg = await res.text()
      throw new Error(errMsg)
    }
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

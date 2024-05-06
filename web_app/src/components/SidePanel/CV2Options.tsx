import { useStore } from "@/lib/states"
import { LabelTitle, RowContainer } from "./LabelTitle"
import { NumberInput } from "../ui/input"
import { Slider } from "../ui/slider"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select"
import { CV2Flag } from "@/lib/types"

const CV2Options = () => {
  const [settings, updateSettings] = useStore((state) => [
    state.settings,
    state.updateSettings,
  ])

  return (
    <div className="flex flex-col gap-4 mt-4">
      <RowContainer>
        <LabelTitle
          text="CV2 Flag"
          url="https://docs.opencv.org/4.8.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca892824c38e258feb5e72f308a358d52e"
        />
        <Select
          value={settings.cv2Flag as string}
          onValueChange={(value) => {
            const flag = value as CV2Flag
            updateSettings({ cv2Flag: flag })
          }}
        >
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Select flag" />
          </SelectTrigger>
          <SelectContent align="end">
            <SelectGroup>
              {Object.values(CV2Flag).map((flag) => (
                <SelectItem key={flag as string} value={flag as string}>
                  {flag as string}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </RowContainer>
      <LabelTitle
        text="CV2 Radius"
        url="https://docs.opencv.org/4.8.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca892824c38e258feb5e72f308a358d52e"
      />
      <RowContainer>
        <Slider
          className="w-[180px]"
          defaultValue={[5]}
          min={1}
          max={100}
          step={1}
          value={[Math.floor(settings.cv2Radius)]}
          onValueChange={(vals) => updateSettings({ cv2Radius: vals[0] })}
        />
        <NumberInput
          id="cv2-radius"
          className="w-[50px] rounded-full"
          numberValue={settings.cv2Radius}
          allowFloat={false}
          onNumberValueChange={(val) => {
            updateSettings({ cv2Radius: val })
          }}
        />
      </RowContainer>
    </div>
  )
}

export default CV2Options

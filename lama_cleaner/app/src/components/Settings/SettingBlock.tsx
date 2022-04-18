import React, { ReactNode } from 'react'

interface SettingBlockProps {
  title: string
  desc?: string
  input: ReactNode
  optionDesc?: ReactNode
  className?: string
}

function SettingBlock(props: SettingBlockProps) {
  const { title, desc, input, optionDesc, className } = props
  return (
    <div className={`setting-block ${className}`}>
      <div className="setting-block-content">
        <div className="setting-block-content-title">
          <span>{title}</span>
          {desc && <span className="setting-block-desc">{desc}</span>}
        </div>
        {input}
      </div>
      {optionDesc && <div className="option-desc">{optionDesc}</div>}
    </div>
  )
}

export default SettingBlock

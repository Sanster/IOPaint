import React, { ReactNode } from 'react'
import Tooltip from '../shared/Tooltip'

interface SettingBlockProps {
  title: string
  desc?: string
  titleSuffix?: ReactNode
  input: ReactNode
  optionDesc?: ReactNode
  className?: string
}

function SettingBlock(props: SettingBlockProps) {
  const { title, titleSuffix, desc, input, optionDesc, className } = props
  return (
    <div className={`setting-block ${className}`}>
      <div className="setting-block-content">
        <div className="setting-block-content-title">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {desc ? (
              <Tooltip content={<div style={{ maxWidth: 400 }}>{desc}</div>}>
                <span>{title}</span>
              </Tooltip>
            ) : (
              <span>{title}</span>
            )}
            {titleSuffix}
          </div>
        </div>
        {input}
      </div>
      {optionDesc && <div className="option-desc">{optionDesc}</div>}
    </div>
  )
}

export default SettingBlock

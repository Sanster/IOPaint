import React, { ReactNode } from 'react'
import Tooltip from '../shared/Tooltip'

interface SettingBlockProps {
  title: string
  desc?: string
  titleSuffix?: ReactNode
  input: ReactNode
  optionDesc?: ReactNode
  className?: string
  layout?: string
}

function SettingBlock(props: SettingBlockProps) {
  const { title, titleSuffix, desc, input, optionDesc, className, layout } =
    props
  const contentClass =
    layout === 'h' ? 'setting-block-content' : 'setting-block-content-v'

  return (
    <div className={`setting-block ${className}`}>
      <div className={contentClass}>
        <div className="setting-block-content-title">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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

SettingBlock.defaultProps = {
  layout: 'h',
}

export default SettingBlock

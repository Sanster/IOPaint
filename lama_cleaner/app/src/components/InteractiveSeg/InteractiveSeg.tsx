import { nanoid } from 'nanoid'
import React, { useEffect, useState } from 'react'
import { useRecoilValue } from 'recoil'
import { interactiveSegClicksState } from '../../store/Atoms'

interface ItemProps {
  x: number
  y: number
  positive: boolean
}

const Item = (props: ItemProps) => {
  const { x, y, positive } = props
  const name = positive ? 'click-item-positive' : 'click-item-negative'
  return <div className={`click-item ${name}`} style={{ left: x, top: y }} />
}

const InteractiveSeg = () => {
  const clicks = useRecoilValue<number[][]>(interactiveSegClicksState)

  return (
    <div className="interactive-seg-wrapper">
      {clicks.map(click => {
        return (
          <Item
            key={click[3]}
            x={click[0]}
            y={click[1]}
            positive={click[2] === 1}
          />
        )
      })}
    </div>
  )
}

export default InteractiveSeg

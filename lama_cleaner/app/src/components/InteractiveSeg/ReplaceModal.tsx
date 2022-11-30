import React, { useEffect, useState } from 'react'
import Button from '../shared/Button'
import Modal from '../shared/Modal'

interface Props {
  show: boolean
  onClose: () => void
  onCleanClick: () => void
  onReplaceClick: () => void
}

const InteractiveSegReplaceModal = (props: Props) => {
  const { show, onClose, onCleanClick, onReplaceClick } = props

  return (
    <Modal
      onClose={onClose}
      title="Mask exists"
      className="modal-setting"
      show={show}
      showCloseIcon
    >
      <h4 style={{ lineHeight: '24px' }}>
        Do you want to remove it or create a new one?
      </h4>
      <div
        style={{
          display: 'flex',
          width: '100%',
          justifyContent: 'flex-end',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        <Button
          onClick={() => {
            onClose()
            onCleanClick()
          }}
        >
          Remove
        </Button>
        <Button onClick={onReplaceClick} border>
          Create a new one
        </Button>
      </div>
    </Modal>
  )
}

export default InteractiveSegReplaceModal

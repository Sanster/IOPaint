import 'hacktimer'
import React from 'react'
import ReactDOM from 'react-dom'
import './styles/_index.scss'
import { RecoilRoot } from 'recoil'
import App from './App'

ReactDOM.render(
  <React.StrictMode>
    <RecoilRoot>
      <App />
    </RecoilRoot>
  </React.StrictMode>,
  document.getElementById('root')
)

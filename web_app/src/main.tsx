import React from "react"
import ReactDOM from "react-dom/client"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import "inter-ui/inter.css"
import App from "./App.tsx"
import "./globals.css"
import { ThemeProvider } from "next-themes"

const queryClient = new QueryClient()

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" disableTransitionOnChange>
        <App />
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>
)

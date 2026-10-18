import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Any request to /api/... from React goes to our Python backend on port 8002
    proxy: {
      '/api': 'http://localhost:8002',
    },
  },
})

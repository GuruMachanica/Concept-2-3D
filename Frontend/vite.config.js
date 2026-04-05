import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    // Bind IPv4 explicitly so localhost/ws resolution is stable on Windows.
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    hmr: {
      host: 'localhost',
      protocol: 'ws',
      port: 5173,
      clientPort: 5173,
    },
    // Proxy API and model asset requests to the backend during development.
    // Set BACKEND env var to override the default host:port.
    proxy: {
      '/models': {
        target: process.env.BACKEND || 'http://localhost:8011',
        changeOrigin: true,
        secure: false,
      },
      '/api': {
        target: process.env.BACKEND || 'http://localhost:8011',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})

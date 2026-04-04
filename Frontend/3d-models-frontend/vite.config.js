import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'permissions-policy-header',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          res.setHeader('Permissions-Policy', 'accelerometer=(), gyroscope=()');
          next();
        });
      },
    },
  ],
  server: {
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

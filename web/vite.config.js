import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  envPrefix: 'REACT_APP_',
  publicDir: 'public',
  plugins: [react()],
  optimizeDeps: {
    include: ['leader-line-new'],
  },
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
  server: {
    port: 80,
    host: '0.0.0.0',
    // Allow requests from any host (nginx proxies with the external hostname)
    allowedHosts: 'all',
    hmr: {
      // Browser connects back through nginx over wss
      protocol: 'wss',
      host: process.env.HOST || 'localhost',
      clientPort: parseInt(process.env.HTTPS_PORT || '8443'),
    },
    watch: {
      usePolling: true,
      interval: 1000,
    },
  },
  build: {
    outDir: 'build',
    target: 'esnext',
  },
  appType: 'spa',
})


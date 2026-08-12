import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/sync': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: '../src/surreal_memory/server/static/dist',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-query': ['@tanstack/react-query'],
          'vendor-recharts': ['recharts'],
          'vendor-motion': ['framer-motion'],
          'vendor-ui': ['class-variance-authority', 'clsx', 'tailwind-merge'],
          'vendor-icons': ['@phosphor-icons/react'],
          // three.js is the heaviest dependency in the app and is only reachable
          // from the Graph route, which is already React.lazy — keeping it in its
          // own chunk means every other page stops paying for it.
          'vendor-3d': ['three', '3d-force-graph'],
        },
      },
    },
  },
})

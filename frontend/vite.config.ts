import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({ plugins: [react()], server: { proxy: { '/api': 'http://127.0.0.1:8001' } }, build: { rollupOptions: { output: { manualChunks(id) { if (id.includes('node_modules/react-leaflet') || id.includes('node_modules/leaflet')) return 'map'; if (id.includes('node_modules/recharts') || id.includes('node_modules/d3-')) return 'charts'; if (id.includes('node_modules/react') || id.includes('node_modules/scheduler')) return 'react-vendor' } } } } })

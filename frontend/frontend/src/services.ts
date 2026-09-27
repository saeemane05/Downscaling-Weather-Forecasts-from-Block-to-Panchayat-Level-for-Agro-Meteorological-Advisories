import type { BlockData, LocationOption } from './types'
const apiBase=(import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/,'')

function apiPath(path:string){ return `${apiBase}${path}` }

export async function getLocations(): Promise<LocationOption[]> {
  const r=await fetch(apiPath('/api/locations')); if(!r.ok) throw new Error('Could not load available locations'); return r.json()
}
export async function getGPLocation(id:string):Promise<LocationOption>{
  const r=await fetch(apiPath(`/api/gp-location?id=${encodeURIComponent(id)}`)); if(!r.ok) throw new Error('No GP forecast found for this id'); return r.json()
}
export async function getBlock(state:string,district:string,block:string):Promise<BlockData>{
  const q=new URLSearchParams({state,district,block}); const r=await fetch(apiPath(`/api/block?${q}`)); if(!r.ok) throw new Error(r.status===404?'Forecast data is unavailable for this block':'Could not load block weather'); return r.json()
}

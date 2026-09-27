export interface LocationOption { state: string; district: string; block: string; gpCount: number; forecastDates: string[] }
export interface ForecastDay { date: string; day: number; temperature: number|null; humidity: number|null; rainfall: number|null; wind: number|null }
export interface GPWeather { id: string; name: string; lat: number|null; lon: number|null; area: number|null; elevation: number|null; slope: number|null; soilClay: number|null; landCover: string|null; forecast: ForecastDay[] }
export interface BlockData { location: LocationOption; gps: GPWeather[]; gpGeometry: GeoJSON.FeatureCollection; blockGeometry: GeoJSON.FeatureCollection|null; updatedAt: string|null }
export interface Advisory { id:string; type:'rainfall'|'humidity'|'wind'|'temperature'; title:string; period:string; severity:'low'|'medium'|'high'; signal:string; description:string; action:string }

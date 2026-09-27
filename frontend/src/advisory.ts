import type { Advisory, ForecastDay } from './types'
const avg=(a:(number|null)[])=>{const v=a.filter((x):x is number=>x!==null&&Number.isFinite(x));return v.length?v.reduce((x,y)=>x+y,0)/v.length:null}
const RULES={rainfallMm:10,humidityPct:80,windMs:7,temperatureAboveWeeklyMeanC:3}
export function buildAdvisories(days:ForecastDay[]):Advisory[]{
 const out:Advisory[]=[]; const range=(key:'rainfall'|'humidity'|'wind'|'temperature', threshold:number, mode:'above'|'below'='above')=>{
  const available=days.filter(d=>d[key]!==null).length, idx=days.filter(d=>d[key]!==null&&(mode==='above'?d[key]!>=threshold:d[key]!<=threshold)).map(d=>d.day);
  if(!idx.length)return null; const first=Math.min(...idx),last=Math.max(...idx);return {period:first===last?`D${first}`:`D${first}–D${last}`,matches:idx.length,available}
 };
 const rain=range('rainfall',RULES.rainfallMm);if(rain)out.push({id:'rain',type:'rainfall',title:'Rainfall watch',period:rain.period,severity:'high',signal:`${rain.matches}/${rain.available} days`,description:'Higher daily rainfall is present in the forecast for this period.',action:'Review irrigation plans and check field drainage.'});
 const humid=range('humidity',RULES.humidityPct);if(humid)out.push({id:'humid',type:'humidity',title:'Humidity watch',period:humid.period,severity:'medium',signal:`${humid.matches}/${humid.available} days`,description:'Forecast humidity may favor disease pressure in some crops.',action:'Monitor crops and follow local, crop-specific guidance.'});
 const wind=range('wind',RULES.windMs);if(wind)out.push({id:'wind',type:'wind',title:'Wind watch',period:wind.period,severity:'medium',signal:`${wind.matches}/${wind.available} days`,description:'Relatively strong winds appear in the forecast.',action:'Review planned spraying conditions and secure vulnerable structures.'});
 const temps=days.map(d=>d.temperature),mean=avg(temps);if(mean!==null&&temps.some(t=>t!==null&&t>mean+RULES.temperatureAboveWeeklyMeanC)){const ix=days.filter(d=>d.temperature!==null&&d.temperature>mean+RULES.temperatureAboveWeeklyMeanC).map(d=>d.day);out.push({id:'heat',type:'temperature',title:'Temperature watch',period:`D${Math.min(...ix)}–D${Math.max(...ix)}`,severity:'medium',signal:`${ix.length}/7 available days`,description:'Some forecast days are warmer than this week’s average.',action:'Monitor soil moisture and crop water stress.'})}
 return out
}
export function forecastInsight(days:ForecastDay[]){const rain=days.filter(d=>d.rainfall!==null).sort((a,b)=>(b.rainfall??0)-(a.rainfall??0))[0];if(!rain)return 'Forecast details are unavailable for this location.';if((rain.rainfall??0)>=8)return `Rainfall is highest around Day ${rain.day} in the available forecast. Review irrigation and drainage plans.`;return 'No large rainfall peak is apparent in the available forecast. Continue to monitor local conditions.'}

import"./modulepreload-polyfill-B5Qt9EMX.js";import*as F from"https://unpkg.com/maplibre-gl@6.12.0/dist/maplibre-gl.mjs";import*as B from"https://cdn.jsdelivr.net/npm/@turf/turf@7.2.0/+esm";const x=(t,r)=>({version:8,sources:{basemap:{type:"raster",tiles:t,tileSize:256,attribution:r}},layers:[{id:"basemap",type:"raster",source:"basemap"}]}),h={liberty:x(["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],"© OpenStreetMap contributors"),positron:x(["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],"© OpenStreetMap contributors"),dark:x(["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],"© OpenStreetMap contributors"),fiord:x(["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],"© OpenStreetMap contributors")},G=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],R=["https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"],O=["https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}"],D=[73.995,19.845],H=10,_={temperature:{title:"Temperature",unit:"Â°C",min:15,max:40,colors:["#fff04a","#ffad1f","#ed401b"],expression:["interpolate",["linear"],["get","temperature"],12,"#fff04a",24,"#ffad1f",42,"#ed401b"]},humidity:{title:"Humidity",unit:"%",min:30,max:100,colors:["#efff9a","#41c9a4","#1556c0"],expression:["interpolate",["linear"],["get","humidity"],0,"#efff9a",50,"#41c9a4",100,"#1556c0"]},rainfall:{title:"Rainfall",unit:"mm",min:0,max:60,colors:["#e8f5ff","#318ad8","#092e87"],expression:["interpolate",["linear"],["get","rainfall"],0,"#e8f5ff",15,"#318ad8",60,"#092e87"]},wind:{title:"Wind Speed",unit:"km/h",min:0,max:60,colors:["#dcfce7","#16a34a","#633bb4"],expression:["interpolate",["linear"],["get","wind_speed"],0,"#dcfce7",25,"#16a34a",90,"#633bb4"]}};let y={type:"FeatureCollection",features:[{type:"Feature",id:"gp001",properties:{id:"gp001",name:"Wavi",temperature:34.7,humidity:68,rainfall:11.9,wind_speed:20,wind_direction:145},geometry:{type:"Point",coordinates:[73.995,19.845]}},{type:"Feature",id:"gp002",properties:{id:"gp002",name:"Sinnar East",temperature:31.5,humidity:73,rainfall:18.5,wind_speed:14,wind_direction:220},geometry:{type:"Point",coordinates:[74.035,19.86]}},{type:"Feature",id:"gp003",properties:{id:"gp003",name:"Sinnar West",temperature:27.2,humidity:82,rainfall:31.4,wind_speed:23,wind_direction:45},geometry:{type:"Point",coordinates:[73.955,19.83]}},{type:"Feature",id:"gp004",properties:{id:"gp004",name:"Example GP 4",temperature:22.5,humidity:91,rainfall:46,wind_speed:8,wind_direction:315},geometry:{type:"Point",coordinates:[74.01,19.89]}}]},u={type:"FeatureCollection",features:[]},w={type:"FeatureCollection",features:[]},n=!0,m=!0,d="temperature",T=!1,A="satellite",b=null,e;e=new F.Map({container:"map",style:h.liberty,center:D,zoom:H,attributionControl:!0,maxZoom:18});window.addEventListener("message",V);e.addControl(new F.NavigationControl({visualizePitch:!0}),"bottom-left");e.addControl(new F.FullscreenControl,"bottom-left");e.on("load",async()=>{await k(),I(),f(),N(),v(),z("satellite"),window.parent.postMessage({type:"hyperweather:map-ready"},"*")});async function k(){var r;const t=(r=e.getStyle().layers.find(i=>i.type==="symbol"))==null?void 0:r.id;e.getSource("satellite")||e.addSource("satellite",{type:"raster",tiles:G,tileSize:256,minzoom:0,maxzoom:19,attribution:"Imagery Â© Esri, Maxar, Earthstar Geographics, and others"}),e.getSource("satellite-labels")||e.addSource("satellite-labels",{type:"raster",tiles:R,tileSize:256,minzoom:0,maxzoom:19,attribution:"Â© Esri"}),e.getSource("satellite-roads")||e.addSource("satellite-roads",{type:"raster",tiles:O,tileSize:256,minzoom:0,maxzoom:19,attribution:"Â© Esri"}),e.getLayer("satellite-layer")||e.addLayer({id:"satellite-layer",type:"raster",source:"satellite",layout:{visibility:"none"},paint:{"raster-opacity":.86,"raster-saturation":-.1,"raster-contrast":.05}},t),e.getLayer("satellite-label-layer")||e.addLayer({id:"satellite-label-layer",type:"raster",source:"satellite-labels",layout:{visibility:"none"},paint:{"raster-opacity":.85}},t),e.getLayer("satellite-road-layer")||e.addLayer({id:"satellite-road-layer",type:"raster",source:"satellite-roads",layout:{visibility:"none"},paint:{"raster-opacity":.96,"raster-contrast":.12,"raster-brightness-max":.98}},t)}function I(){e.getSource("weather")||e.addSource("weather",{type:"geojson",data:y,promoteId:"id"}),e.getSource("gp-boundaries")||e.addSource("gp-boundaries",{type:"geojson",data:u}),e.getSource("block-boundaries")||e.addSource("block-boundaries",{type:"geojson",data:w}),e.getLayer("block-boundary-casing")||e.addLayer({id:"block-boundary-casing",type:"line",source:"block-boundaries",paint:{"line-color":"#10261f","line-width":5,"line-opacity":.92,"line-dasharray":[2,1]}}),e.getLayer("block-boundary-line")||e.addLayer({id:"block-boundary-line",type:"line",source:"block-boundaries",paint:{"line-color":"#f4c542","line-width":2.5,"line-dasharray":[2,1],"line-opacity":1}}),e.getLayer("gp-boundary-fill")||e.addLayer({id:"gp-boundary-fill",type:"fill",source:"gp-boundaries",paint:{"fill-color":"#FFFFFF","fill-opacity":.03}}),e.getLayer("gp-boundary-casing")||e.addLayer({id:"gp-boundary-casing",type:"line",source:"gp-boundaries",paint:{"line-color":"#10231c","line-width":3.8,"line-opacity":.96}}),e.getLayer("gp-boundary-line")||e.addLayer({id:"gp-boundary-line",type:"line",source:"gp-boundaries",paint:{"line-color":"#FFFFFF","line-width":1.7,"line-opacity":.98}}),e.getLayer("selected-gp-fill")||e.addLayer({id:"selected-gp-fill",type:"fill",source:"gp-boundaries",filter:["==",["get","gp_id"],""],paint:{"fill-color":"#168cff","fill-opacity":.34}}),e.getLayer("selected-gp-line")||e.addLayer({id:"selected-gp-line",type:"line",source:"gp-boundaries",filter:["==",["get","gp_id"],""],paint:{"line-color":"#0878f9","line-width":3.5,"line-opacity":1}}),e.getLayer("weather-glow")||e.addLayer({id:"weather-glow",type:"circle",source:"weather",paint:{"circle-radius":["interpolate",["linear"],["zoom"],7,9,10,12,14,19],"circle-color":"#FFFFFF","circle-opacity":.14,"circle-blur":1}}),e.getLayer("weather-points")||e.addLayer({id:"weather-points",type:"circle",source:"weather",paint:{"circle-radius":["interpolate",["linear"],["zoom"],7,4,10,6,13,8,16,11],"circle-color":_.temperature.expression,"circle-opacity":.94,"circle-stroke-color":"#FFFFFF","circle-stroke-width":1.4}}),e.getLayer("selected-gp")||e.addLayer({id:"selected-gp",type:"circle",source:"weather",filter:["==",["get","id"],""],paint:{"circle-radius":["interpolate",["linear"],["zoom"],8,12,12,18,16,24],"circle-color":"rgba(255,255,255,0)","circle-stroke-color":"#FFFFFF","circle-stroke-width":3}}),e.getLayer("wind-arrows")||e.addLayer({id:"wind-arrows",type:"symbol",source:"weather",layout:{"text-field":"âž¤","text-size":["interpolate",["linear"],["zoom"],7,10,12,15,16,20],"text-rotate":["coalesce",["get","wind_direction"],0],"text-rotation-alignment":"map","text-allow-overlap":!0,visibility:"none"},paint:{"text-color":"#FFFFFF","text-halo-color":"#222222","text-halo-width":1}}),e.getLayer("gp-labels")||e.addLayer({id:"gp-labels",type:"symbol",source:"gp-boundaries",layout:{"text-field":["coalesce",["get","gp_name"],["get","name"]],"text-font":["Noto Sans Regular"],"text-size":["interpolate",["linear"],["zoom"],8,8,10,10,13,12],"text-anchor":"center","text-max-width":9,"text-padding":3,"text-allow-overlap":!1,"text-ignore-placement":!1,visibility:n?"visible":"none"},paint:{"text-color":"#ffffff","text-halo-color":"#14251e","text-halo-width":2,"text-halo-blur":.4}}),j(),E(),e.on("click","weather-points",Z),e.on("mouseenter","weather-points",()=>{e.getCanvas().style.cursor="pointer"}),e.on("mouseleave","weather-points",()=>{e.getCanvas().style.cursor=""});for(const[t,r]of[["gp-boundary-fill",n],["selected-gp-fill",n],["selected-gp-line",n],["gp-boundary-casing",n],["gp-boundary-line",n],["gp-labels",n],["block-boundary-casing",m],["block-boundary-line",m]])e.getLayer(t)&&e.setLayoutProperty(t,"visibility",r?"visible":"none")}function j(){var i;const t=new Set(["gp-labels","wind-arrows"]),r=(i=e.getStyle().layers.find(a=>a.type==="symbol"&&!t.has(a.id)))==null?void 0:i.id;r&&["block-boundary-casing","block-boundary-line","gp-boundary-fill","gp-boundary-casing","gp-boundary-line","weather-glow","weather-points","selected-gp","wind-arrows"].forEach(a=>{e.getLayer(a)&&e.moveLayer(a,r)})}function E(){const t=["liberty","positron","fiord","agriculture"].includes(A);e.getLayer("gp-labels")&&(e.setPaintProperty("gp-labels","text-color",t?"#142a20":"#ffffff"),e.setPaintProperty("gp-labels","text-halo-color",t?"#f8fff0":"#13231c"),e.setPaintProperty("gp-labels","text-halo-width",2.8)),e.getLayer("gp-boundary-line")&&e.setPaintProperty("gp-boundary-line","line-color",t?"#176b50":"#ffffff"),e.getLayer("gp-boundary-casing")&&e.setPaintProperty("gp-boundary-casing","line-color",t?"#f7fff3":"#10231c"),e.getLayer("block-boundary-line")&&e.setPaintProperty("block-boundary-line","line-color",t?"#00885d":"#ffd44f")}function c(t){const r=Number(t);return t==null||t===""||!Number.isFinite(r)?"Not available":r.toFixed(4)}function Z(t){if(!t.features||!t.features.length)return;const r=t.features[0],i=r.properties,a=r.geometry.coordinates.slice();L(i.id),window.parent.postMessage({type:"hyperweather:select-gp",id:String(i.id)},"*"),new F.Popup({closeButton:!0,closeOnClick:!0}).setLngLat(a).setHTML(`

        <div class="popup">

            <div class="popup-title">

                ${i.name}

            </div>


            <div class="popup-row">

                <span>ðŸŒ¡ Temperature</span>

                <strong>
                    ${c(i.temperature)} Â°C
                </strong>

            </div>


            <div class="popup-row">

                <span>ðŸ’§ Humidity</span>

                <strong>
                    ${c(i.humidity)} %
                </strong>

            </div>


            <div class="popup-row">

                <span>ðŸŒ§ Rainfall</span>

                <strong>
                    ${c(i.rainfall)} mm
                </strong>

            </div>


            <div class="popup-row">

                <span>ðŸ’¨ Wind</span>

                <strong>
                    ${c(i.wind_speed)} km/h
                </strong>

            </div>

        </div>

    `).addTo(e)}function V(t){var i;const r=t.data;if(!(!r||typeof r!="object")){if(r.type==="hyperweather:data"){b=r.selectedId==null?null:String(r.selectedId);const a=Array.isArray(r.gps)?r.gps:[],o=Number(r.dayIndex||0);y={type:"FeatureCollection",features:a.filter(l=>Number.isFinite(l.lon)&&Number.isFinite(l.lat)).map(l=>{var S;const s=((S=l.forecast)==null?void 0:S[o])||{};return{type:"Feature",id:String(l.id),properties:{id:String(l.id),gp_id:String(l.id),name:l.name,temperature:s.temperature,humidity:s.humidity,rainfall:s.rainfall,wind_speed:s.wind==null?null:s.wind*3.6,wind_direction:s.wind_direction},geometry:{type:"Point",coordinates:[l.lon,l.lat]}}})},u=r.gpGeometry||{type:"FeatureCollection",features:[]},w=r.blockGeometry||{type:"FeatureCollection",features:[]},e.isStyleLoaded()&&e.getSource("weather")&&(e.getSource("weather").setData(y),e.getSource("gp-boundaries").setData(u),(i=e.getSource("block-boundaries"))==null||i.setData(w),f(),r.selectedId?L(r.selectedId):v())}else if(r.type==="hyperweather:mode")T=r.mode==="difference",f();else if(r.type==="hyperweather:layer")d=String(r.layer||"temperature").toLowerCase(),f();else if(r.type==="hyperweather:selected")L(r.id);else if(r.type==="hyperweather:basemap")z(r.base);else if(r.type==="hyperweather:fit")v();else if(r.type==="hyperweather:boundaries"){n=!!r.gp,m=!!r.block;for(const[a,o]of[["gp-boundary-fill",n],["selected-gp-fill",n],["selected-gp-line",n],["gp-boundary-casing",n],["gp-boundary-line",n],["gp-labels",n],["block-boundary-casing",m],["block-boundary-line",m]])e.getLayer(a)&&e.setLayoutProperty(a,"visibility",o?"visible":"none")}}}function L(t){b=String(t),e.getLayer("selected-gp")&&e.setFilter("selected-gp",["==",["get","id"],t]);for(const o of["selected-gp-fill","selected-gp-line"])e.getLayer(o)&&e.setFilter(o,["==",["to-string",["get","gp_id"]],String(t)]);const r=y.features.find(o=>String(o.properties.id)===String(t));if(!r)return;const i=(u.features||[]).filter(o=>{var l,s;return String(((l=o.properties)==null?void 0:l.gp_id)??((s=o.properties)==null?void 0:s.id)??"")===String(t)});if(i.length){const o=B.bbox({type:"FeatureCollection",features:i});e.fitBounds([[o[0],o[1]],[o[2],o[3]]],{padding:70,maxZoom:11,duration:850})}else e.easeTo({center:r.geometry.coordinates,zoom:Math.max(e.getZoom(),10),duration:850});const a=r.properties;document.getElementById("selectedCard").innerHTML=`

        <div class="selected-name">

            ${a.name}

        </div>


        <div class="selected-grid">

            <div class="metric">

                ðŸŒ¡ Temperature

                <strong>
                    ${c(a.temperature)} Â°C
                </strong>

            </div>


            <div class="metric">

                ðŸ’§ Humidity

                <strong>
                    ${c(a.humidity)} %
                </strong>

            </div>


            <div class="metric">

                ðŸŒ§ Rainfall

                <strong>
                    ${c(a.rainfall)} mm
                </strong>

            </div>


            <div class="metric">

                ðŸ’¨ Wind

                <strong>
                    ${c(a.wind_speed)} km/h
                </strong>

            </div>

        </div>

    `}function f(){const t=_[d];if(!e.getLayer("weather-points"))return;const r=d==="wind"?"wind_speed":d,i=y.features.map(p=>p.properties[r]).filter(p=>p!=null&&Number.isFinite(Number(p))).map(Number);let a=i.length?Math.min(...i):t.min,o=i.length?Math.max(...i):t.max;o===a&&(a-=.5,o+=.5);let l=d==="temperature"?["#fff04a","#ffad1f","#ed401b"]:d==="rainfall"?["#e8f5ff","#318ad8","#092e87"]:d==="humidity"?["#efff9a","#41c9a4","#1556c0"]:["#dcfce7","#16a34a","#633bb4"],s=a+(o-a)/2;if(T&&i.length){const p=Math.max(.1,...i.map(g=>Math.abs(g)));a=-p,s=0,o=p,l=["#c85539","#f4f3e9","#176dc1"]}const S=["interpolate",["linear"],["coalesce",["get",r],a],a,l[0],s,l[1],o,l[2]];if(e.setPaintProperty("weather-points","circle-color",S),e.setPaintProperty("weather-points","circle-radius",["interpolate",["linear"],["zoom"],7,7,10,10,13,13,16,16]),e.setPaintProperty("weather-points","circle-opacity",i.length?1:0),e.setPaintProperty("gp-boundary-fill","fill-color",["interpolate",["linear"],["coalesce",["get","weatherValue"],a],a,l[0],s,l[1],o,l[2]]),e.setPaintProperty("gp-boundary-fill","fill-opacity",i.length?["interpolate",["linear"],["zoom"],7,.26,12,.38,15,.44]:.03),e.getSource("gp-boundaries")){const p={...u,features:(u.features||[]).map(g=>{var C,M;const $=String(((C=g.properties)==null?void 0:C.gp_id)??((M=g.properties)==null?void 0:M.id)??""),P=y.features.find(W=>String(W.properties.id)===$);return{...g,properties:{...g.properties,weatherValue:(P==null?void 0:P.properties[r])??a}}})};e.getSource("gp-boundaries").setData(p)}e.getLayer("wind-arrows")&&e.setLayoutProperty("wind-arrows","visibility",d==="wind"?"visible":"none"),N()}function N(){const t=_[d];document.getElementById("legendTitle").textContent=t.title,document.getElementById("legendMin").textContent=`${t.min}${t.unit}`,document.getElementById("legendMax").textContent=`${t.max}${t.unit}`,document.getElementById("legendGradient").style.background=`linear-gradient(
            to right,
            ${t.colors.join(",")}
        )`}async function z(t){if(t==="bright"&&(t="liberty"),A=t,t==="satellite"||t==="satellite-labels"){e.getLayer("satellite-layer")&&e.setLayoutProperty("satellite-layer","visibility","visible"),e.getLayer("satellite-label-layer")&&e.setLayoutProperty("satellite-label-layer","visibility",t==="satellite-labels"?"visible":"none"),e.getLayer("satellite-road-layer")&&e.setLayoutProperty("satellite-road-layer","visibility","visible"),E();return}if(e.getLayer("satellite-layer")&&e.setLayoutProperty("satellite-layer","visibility","none"),e.getLayer("satellite-label-layer")&&e.setLayoutProperty("satellite-label-layer","visibility","none"),e.getLayer("satellite-road-layer")&&e.setLayoutProperty("satellite-road-layer","visibility","none"),t==="agriculture"){await U();return}h[t]&&(e.setStyle(h[t]),e.once("style.load",async()=>{await k(),I(),f(),b&&L(b),E()}))}async function U(){try{const r=await(await fetch(h.liberty)).json();for(const i of r.layers)i.type==="background"&&(i.paint={...i.paint||{},"background-color":"#dce9cc"}),i.type==="fill"&&i["source-layer"]&&String(i["source-layer"]).toLowerCase().includes("water")&&(i.paint={...i.paint||{},"fill-color":"#9dd8e8","fill-opacity":.85}),i.type==="fill"&&i["source-layer"]&&/forest|wood|park|landcover/i.test(i["source-layer"])&&(i.paint={...i.paint||{},"fill-color":"#78a96b"}),i.type==="fill"&&i["source-layer"]&&/landuse/i.test(i["source-layer"])&&(i.paint={...i.paint||{},"fill-color":"#c7ddb2","fill-opacity":.75}),i.type==="line"&&i["source-layer"]&&/transportation|road/i.test(i["source-layer"])&&(i.paint={...i.paint||{},"line-color":"#d8a15d"}),i.type==="symbol"&&i.layout&&i.layout["text-field"]&&(i.paint={...i.paint||{},"text-color":"#142a20","text-halo-color":"#f8fff0","text-halo-width":1.8,"text-halo-blur":.2,"text-opacity":1});e.setStyle(r),e.once("style.load",async()=>{await k(),I(),f(),b&&L(b)})}catch(t){console.error("Custom style failed:",t),e.setStyle(h.liberty)}}function v(){var i,a;if(!y.features.length)return;const t=(i=w.features)!=null&&i.length?w:(a=u.features)!=null&&a.length?u:y,r=B.bbox(t);e.fitBounds([[r[0],r[1]],[r[2],r[3]]],{padding:52,maxZoom:11,duration:700})}document.getElementById("fitButton").addEventListener("click",()=>{v()});document.getElementById("baseMapSelect").addEventListener("change",t=>{z(t.target.value)});document.getElementById("weatherSelect").addEventListener("change",t=>{d=t.target.value,f(),window.parent.postMessage({type:"hyperweather:layer",layer:d},"*")});window.addEventListener("resize",()=>{e.resize()});

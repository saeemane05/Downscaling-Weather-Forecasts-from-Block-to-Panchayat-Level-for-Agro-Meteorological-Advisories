# HyperWeather UI V2 Changes

This version preserves the existing data/API/model integration and focuses on the desktop dashboard presentation.

## Changes

- Main dashboard optimized for a 1366×768 desktop viewport.
- Reduced unnecessary page/card whitespace.
- Increased primary typography and weather-value hierarchy.
- Reduced header/sidebar vertical footprint.
- Reduced map/card heights while keeping the map prominent.
- Compact 7-day forecast so D1–D7 fit in one frame.
- Compact farm-advisory cards while keeping advisory text readable.
- Compact environmental, spatial-variation and crop-advisory sections.
- Added desktop height-aware layout rules for short desktop viewports.
- Preserved normal scrolling for secondary/mobile pages.
- Forecast graph wind values now display in km/h consistently with the dashboard cards; underlying model values remain unchanged.
- Existing location discovery, map, GP selection, forecast, advisory and data behavior are preserved.

## Verification

TypeScript project compilation (`tsc -b`) passed on the updated source.

A full Vite production build could not be executed in this Linux packaging environment because the uploaded Windows `node_modules` lacks the Linux Rollup optional binary. This is an environment/dependency issue, not a TypeScript source error.

On the user's Windows project, run:

```bash
npm install
npm run build
npm run dev
```

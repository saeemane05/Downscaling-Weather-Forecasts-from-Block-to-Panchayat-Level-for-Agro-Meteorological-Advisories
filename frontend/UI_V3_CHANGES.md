# HyperWeather UI V3 — Information-First Dashboard

## What changed

The previous dashboard had two navigation systems that duplicated each other:

- the top Dashboard / Insights / About links
- the left Home / Map / Forecast / Advisory / Insights / About sidebar

For the main dashboard, these were removed because the dashboard already exposes the map, forecast and advisory together.

### New header

The header now communicates useful status/context instead of duplicate navigation:

- Weather intelligence for resilient farms
- Latest available model output
- 7-day horizon
- How it works
- Data & methodology

### New dashboard information strip

A compact strip below the hero heading now shows:

- number of Gram Panchayats covered
- forecast horizon
- model output run date
- number of weather variables

These values are tied to the current selected block where applicable.

### Layout

The dashboard now uses the available desktop viewport more effectively:

- wider full-page content because the sidebar is removed
- larger map
- larger selected-GP weather card
- larger typography
- compact but readable 7-day forecast
- compact Farm Advisory
- compact Environmental Context
- Spatial Weather Variation
- Crop Advisory
- taller layouts on larger desktop screens to reduce the large empty lower area

### Responsive behavior

- Desktop: information-dense one-frame dashboard.
- Mobile/tablet: normal responsive behavior and scrolling where required.
- Secondary pages can still scroll normally.

## Important

No forecasting/data logic was intentionally changed.

The existing data adapter, model outputs, GP boundaries, advisories and wind conversion behavior remain the source of truth.

## Run

```bash
npm install
npm run build
npm run dev
```


## V3.1 — Weather layer visibility fix

The Weather Layer controls were being clipped below the fixed location-card height.
The four layer buttons are now arranged in a compact 2×2 grid, and the location
selectors use slightly tighter vertical spacing on desktop so all four weather
layers remain visible without changing the data or map behavior.

# HyperWeather Frontend — Codex Implementation Specification

## 1. Objective

Build a polished, responsive frontend for **HyperWeather**, a hyperlocal weather intelligence and farm advisory platform.

Core product story:

> **Turn block-level weather information into actionable Gram Panchayat-level weather intelligence for farmers.**

The UI should feel like a real production **weather intelligence / GIS / agricultural decision-support dashboard**, not an ML demo and not a generic weather app.

The uploaded UI image is the primary visual reference. Recreate its structure and visual language as actual components; do not embed the screenshot.

---

## 2. Core UX principle

The primary user journey is:

**Location → Hyperlocal Weather → 7-Day Forecast → Local Conditions → Farm Advisory**

Do NOT make the main UI revolve around:

- Stage 1
- Stage 2
- residual correction
- ExtraTrees
- model training
- feature engineering
- train/validation/test splits
- RMSE

Those belong only in technical/analytics/insights sections if needed.

A farmer or SIH judge should understand the main dashboard within seconds.

---

## 3. Visual direction

Match the uploaded reference image:

- clean white/light background
- dark navy/green typography
- agricultural green as primary accent
- blue weather accents
- rounded cards
- subtle borders and shadows
- compact but readable information density
- map as the central visual element
- left-side location/weather controls
- right-side selected-GP information
- 7-day forecast below the map
- prominent Farm Advisory
- modern SaaS/GIS dashboard aesthetic

Recommended font: **Inter** or the existing project font.

Recommended icons: **Lucide React**.

Do not mix unrelated icon libraries.

---

## 4. Technology

If the repository already has a frontend stack, preserve it unless there is a strong reason to change it.

Preferred if starting from scratch:

- React
- TypeScript
- Vite or Next.js
- Tailwind CSS
- Lucide React
- Leaflet or MapLibre GL JS
- Recharts

Keep the architecture modular.

Do not rewrite the existing ML/data pipeline just to make the frontend work.

---

# 5. Application navigation

Create:

- Dashboard
- Map
- Forecast
- Farm Advisory
- Insights
- About

Optional:

- Analytics

Default route:

`/dashboard`

Suggested sidebar:

```text
Home
Map
Forecast
Advisory
Analytics
Resources
```

On mobile, collapse this into a compact navigation/bottom navigation.

---

# 6. Header

Sticky top header.

Left:

```text
☁ HyperWeather
Hyperlocal Weather Intelligence for Every Gram Panchayat
```

Right/center:

```text
Dashboard
Insights
About
```

Far right:

```text
Notifications
User/avatar
```

Show current context:

```text
Maharashtra • Nashik • Sinnar
```

Keep header compact.

---

# 7. Location selection

Create hierarchical selectors:

```text
State
[ Maharashtra ▼ ]

District
[ Nashik ▼ ]

Block
[ Sinnar ▼ ]

Gram Panchayat
[ Sinnar GP ▼ ]
```

Must be dynamic.

Do NOT hard-code Sinnar as the only location.

The system should support examples such as:

```text
Maharashtra → Nashik → Sinnar
Maharashtra → Pune → Baramati
Goa → North Goa → Bicholim
```

When block changes:

- load that block's boundary
- load its GPs
- update weather markers
- reset/select a valid GP
- update forecast
- update advisory
- update environmental context

---

# 8. Main dashboard layout

Desktop target:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ HEADER                                                               │
├──────────────┬───────────────────────────────────────┬───────────────┤
│ LOCATION     │                                       │ SELECTED GP   │
│ CONTROLS     │               MAP                     │ WEATHER       │
│              │                                       │               │
│ WEATHER      │                                       │               │
│ LAYERS       │                                       │               │
├──────────────┴───────────────────────────────────────┴───────────────┤
│                         7-DAY FORECAST                               │
├───────────────────────────────────────────┬──────────────────────────┤
│ LOCAL ENVIRONMENT / SPATIAL VARIATION     │ FARM ADVISORY            │
└───────────────────────────────────────────┴──────────────────────────┘
```

Map should be the visual centerpiece.

---

# 9. Interactive map

Use Leaflet or MapLibre.

Required functionality:

- pan
- zoom
- block boundary
- GP boundaries
- GP markers
- selected GP highlighting
- weather-layer switching
- base-map/satellite option if available
- legend
- fullscreen control if practical

Initial load should only load the currently selected block and its GPs.

Do not load every GP boundary in the entire country.

---

# 10. Weather layers

Provide:

```text
Temperature
Rainfall
Humidity
Wind
```

Only one should be active at a time.

Temperature layer:
- marker/area appearance based on temperature.

Rainfall layer:
- appearance based on rainfall.

Humidity layer:
- appearance based on humidity.

Wind layer:
- appearance based on wind speed.

Use intuitive weather semantics, and do not rely on color alone.

---

# 11. GP markers

Every available GP should have a marker.

Clicking a marker must:

1. select GP
2. highlight GP boundary if available
3. update selected-GP card
4. update 7-day forecast
5. update farm advisory
6. update environmental context
7. update charts

Popup example:

```text
Sinnar GP

28.4°C
Humidity 72%
Rainfall 4.2 mm
Wind 12 km/h

View details →
```

Keep popups concise.

---

# 12. Selected Gram Panchayat card

Right side of map.

Example structure:

```text
Selected Gram Panchayat

Sinnar GP (GP-01)
Sinnar Block, Nashik, Maharashtra

☀ 28.4°C

Humidity       72%
Rainfall       4.2 mm
Wind Speed     12 km/h
Wind Direction NW

Last updated: <real timestamp>
```

Do not invent values or timestamps.

Use actual backend/data values.

---

# 13. Weather Insight

Below selected-GP metrics:

```text
Weather Insight

Rainfall is expected to increase around Day 3–4.
Review irrigation scheduling accordingly.
```

Generate this from actual forecast data where possible.

Use cautious language.

Avoid absolute claims unless supported by the data.

---

# 14. 7-Day forecast

Below map:

```text
7-Day Forecast

D1     D2     D3     D4     D5     D6     D7
date   date   date   date   date   date   date

icon   icon   icon   icon   icon   icon   icon

temp   temp   temp   temp   temp   temp   temp

rain   rain   rain   rain   rain   rain   rain

wind   wind   wind   wind   wind   wind   wind
```

Use actual forecast values.

Add:

```text
View as Graph
```

Allow chart variable switching:

- Temperature
- Rainfall
- Humidity
- Wind

Use Recharts or equivalent.

Mobile: horizontal scrolling is acceptable.

---

# 15. Farm Advisory — core feature

Farm Advisory must be prominent.

Section title:

```text
🌱 Farm Advisory
Based on 7-day forecast
```

Advisories should be forecast-triggered.

Each card contains:

- condition
- period
- recommended action
- confidence

Examples:

### Rainfall

```text
Rainfall Advisory
D3–D4

Moderate to heavy rainfall expected.

Recommended action:
Review irrigation scheduling and check field drainage.

Confidence: High
```

### Humidity

```text
Disease Management
D3–D5

Higher humidity may create conditions favorable
for some fungal diseases.

Recommended action:
Monitor crops and follow crop-specific guidance.

Confidence: Medium
```

### Wind

```text
Wind Advisory
D4

Relatively strong winds expected.

Recommended action:
Avoid spraying during unsuitable wind conditions
and secure vulnerable farm structures.

Confidence: Medium
```

### Temperature

```text
Temperature Advisory
D6–D7

Higher temperatures expected.

Recommended action:
Monitor soil moisture and crop water stress.

Confidence: Medium
```

Do not present advisory text as guaranteed agronomic instructions.

Prefer:

> Review planned irrigation.

instead of:

> Do not irrigate.

Prefer:

> Monitor for disease-favorable conditions.

instead of:

> Disease will occur.

---

# 16. Advisory engine

Create a separate rules/service module.

Suggested type:

```ts
interface FarmAdvisory {
  id: string;
  type:
    | "rainfall"
    | "humidity"
    | "wind"
    | "temperature"
    | "general";
  title: string;
  period: string;
  severity: "low" | "medium" | "high";
  confidence: "low" | "medium" | "high";
  description: string;
  action: string;
}
```

Architecture:

```text
forecast data
    ↓
advisory rules
    ↓
FarmAdvisory objects
    ↓
UI cards
```

Keep rules explainable and configuration-driven.

If rules are prototype rules, make that clear in About/Advisory documentation.

---

# 17. Crop profile

Add:

```text
Crop Advisory

Crop
[ Sugarcane ▼ ]

Growth Stage
[ Vegetative ▼ ]

Soil Type
[ Black Soil ▼ ]
```

Then:

```text
Crop-specific insight

Current conditions may be favorable for vegetative growth.
Monitor disease-favorable conditions during high humidity.
```

Crop rules should eventually be backend/configuration driven.

Do not pretend prototype recommendations are officially validated agronomic advice.

Use a note such as:

> Prototype weather-based advisory — verify with local agricultural guidance.

---

# 18. Local environmental context

Show:

```text
Local Environmental Context

Elevation       612 m
Slope           4.8°
Land Cover      Cropland
Soil Type       Black Soil
```

Use actual spatial data.

If unavailable:

```text
Not available
```

Never invent values.

---

# 19. Spatial weather variation

Add a card showing why hyperlocal information matters.

Example:

```text
Spatial Weather Variation

Block Average
27.1°C

GP Forecast Range
24.8°C — 29.4°C

Local variation
4.6°C
```

Calculate dynamically from current block/forecast data.

Do not hard-code example values.

This is an important storytelling element:

> Different Gram Panchayats inside the same block can have different expected conditions.

---

# 20. GP detail route

Create:

`/gp/:gpId`

Include:

- GP name/location
- current weather
- 7-day forecast
- forecast charts
- farm advisory
- environmental context
- map location
- spatial comparison

Map marker → GP detail page should be easy.

---

# 21. Block overview route

Create:

`/block/:blockId`

Include:

- block boundary
- GP count
- weather summary
- GP weather distribution
- forecast summary
- advisory summary

Example structure:

```text
Sinnar Block

113 Gram Panchayats

Average Temperature   <dynamic>
Average Humidity      <dynamic>
Expected Rainfall     <dynamic>

Temperature range
<min> ───────── <max>
```

Calculate all metrics dynamically.

---

# 22. Insights page

Use this page for explaining the science behind the system.

Include the previously generated explanatory graphs:

- coarse → hyperlocal resolution
- temperature vs elevation
- precipitation/topography
- predictor relevance
- static vs dynamic predictors
- information layers

Main message:

```text
Why can weather differ within a block?

Weather conditions are influenced by
atmospheric state + terrain + land characteristics
+ location + time.
```

Do not describe these graphs as measured project model results unless they actually are.

---

# 23. Analytics page

Optional technical page.

Possible content:

- forecast vs reference
- spatial variation
- distributions
- model metrics
- historical performance

When evaluation uses ERA5-Land, label it accurately:

```text
Evaluation against ERA5-Land reference
```

Do not call ERA5-Land "ground truth" unless the actual validation methodology supports that wording.

---

# 24. About page

Explain:

## What is HyperWeather?

Suggested wording:

> HyperWeather converts broader-area weather information into more location-specific Gram Panchayat weather intelligence using atmospheric conditions and spatial environmental information.

Show:

```text
Broad Weather Information
          +
Terrain
          +
Land Cover
          +
Soil
          +
Location
          ↓
Hyperlocal Weather Intelligence
          ↓
Farm Advisory
```

Keep it understandable.

---

# 25. Data architecture

Do not tightly couple UI components to CSV parsing.

Use a data/service layer.

Suggested structure:

```text
src/
├── components/
│   ├── Header.tsx
│   ├── Sidebar.tsx
│   ├── LocationSelector.tsx
│   ├── WeatherMap.tsx
│   ├── GPWeatherCard.tsx
│   ├── Forecast7Day.tsx
│   ├── ForecastChart.tsx
│   ├── FarmAdvisory.tsx
│   ├── CropAdvisory.tsx
│   ├── EnvironmentalContext.tsx
│   └── SpatialVariation.tsx
│
├── pages/
│   ├── Dashboard.tsx
│   ├── MapPage.tsx
│   ├── ForecastPage.tsx
│   ├── AdvisoryPage.tsx
│   ├── InsightsPage.tsx
│   ├── AnalyticsPage.tsx
│   └── AboutPage.tsx
│
├── services/
│   ├── weatherService.ts
│   ├── gpService.ts
│   ├── advisoryService.ts
│   └── locationService.ts
│
├── types/
│   ├── weather.ts
│   ├── gp.ts
│   ├── advisory.ts
│   └── location.ts
│
└── App.tsx
```

Adapt to the repository instead of blindly replacing an existing architecture.

---

# 26. Backend integration

Preferred architecture:

```text
Frontend
    ↓
API/data service
    ↓
Existing Python/model outputs
```

If an API does not exist yet, create a clean development data adapter.

Do NOT create a second forecasting system inside React.

The frontend should consume existing forecasts.

---

# 27. Dynamic location support

Never build the UI around only Sinnar.

It must support all available state/district/block/GP combinations.

Use dynamic discovery from backend/data where possible.

---

# 28. Loading/error/empty states

Every data-dependent section needs a loading state.

Example:

```text
Loading GP weather...
```

If data is unavailable:

```text
Weather data unavailable

We could not load the forecast for this location.
Please try again.
```

If GP boundary is unavailable:

```text
GP boundary unavailable

Weather information is still available.
```

If no GP is selected:

```text
Select a Gram Panchayat

Choose a GP from the map or location selector
to view its hyperlocal forecast.
```

One missing layer must not crash the entire dashboard.

---

# 29. Map performance

The existing project has potentially large GeoJSON files.

Do not send massive geometry payloads unnecessarily.

Use:

- simplified geometries
- only selected block
- only its GPs
- lazy loading
- caching

Never load every state/block geometry on initial dashboard load.

---

# 30. Responsive design

Desktop is the primary SIH demo target.

Desktop:

```text
Sidebar | Main map/content | Selected GP
```

Tablet:

```text
Two-column layout
```

Mobile:

```text
Header
Location
Map
Selected GP
Forecast
Advisory
Environmental context
```

7-day forecast may horizontally scroll.

No horizontal page overflow.

---

# 31. Accessibility

Use:

- sufficient contrast
- keyboard-accessible controls
- accessible dropdowns
- meaningful tooltips
- labels for icons
- text labels alongside color-based warnings

Never communicate severity through color alone.

Example:

```text
Rainfall: High
```

not merely a red marker.

---

# 32. Data integrity

Never invent:

- weather values
- GP coordinates
- GP boundaries
- elevation
- soil type
- land cover
- crop conditions
- confidence
- timestamps
- forecast probabilities

If unavailable, show:

```text
Not available
```

or hide the optional component gracefully.

---

# 33. Demo mode

Optional but recommended.

Default demo:

```text
Maharashtra
→ Nashik
→ Sinnar
→ first available GP
```

This makes the SIH demonstration smooth.

Keep demo data clearly separated from production/backend data.

---

# 34. Suggested visual design tokens

Primary:
- agricultural green

Secondary:
- weather blue

Background:
- very light blue/neutral

Text:
- dark navy/charcoal

Cards:
- white
- subtle border
- subtle shadow
- 10–16px radius

Use restrained color.

Do not make every card brightly colored.

---

# 35. Important UX hierarchy

The interface should answer questions in this order:

### 1. WHERE AM I?
State → District → Block → GP

### 2. WHAT IS THE WEATHER?
Map + current conditions

### 3. WHAT WILL HAPPEN?
7-day forecast

### 4. WHAT DOES IT MEAN LOCALLY?
Spatial variation + environmental context

### 5. WHAT SHOULD I CONSIDER?
Farm Advisory

### 6. WHY DOES THIS WORK?
Insights/scientific explanation

This hierarchy is more important than adding extra features.

---

# 36. Implementation order

## Phase 1 — Repository audit

Before coding, inspect:

- existing frontend
- Python backend
- model output locations
- forecast files
- GP boundary files
- current dashboard
- available APIs
- existing assets

Do not immediately rewrite existing code.

## Phase 2 — Data contract

Define TypeScript interfaces for:

- Location
- GramPanchayat
- CurrentWeather
- WeatherForecast
- EnvironmentalContext
- FarmAdvisory
- CropProfile

## Phase 3 — Core shell

Build:

- routing
- header
- sidebar
- theme
- responsive cards

## Phase 4 — Map/location

Build:

- location selectors
- map
- block boundary
- GP boundaries
- markers
- selected GP

## Phase 5 — Weather

Build:

- current weather
- 7-day forecast
- charts
- weather layers

## Phase 6 — Farm Advisory

Build:

- advisory engine
- advisory cards
- confidence
- crop selector
- crop-specific context

## Phase 7 — Insights

Add scientific explanatory graphs.

## Phase 8 — Polish

Test:

- responsive layouts
- loading
- errors
- empty states
- accessibility
- map performance
- visual consistency

---

# 37. Acceptance criteria

The implementation is complete when:

- [ ] Dashboard visually follows the uploaded reference.
- [ ] State → District → Block → GP selection works.
- [ ] Sinnar is not hard-coded as the only location.
- [ ] Interactive map works.
- [ ] Block boundary works.
- [ ] GP boundaries work when available.
- [ ] GP markers work.
- [ ] GP selection updates the dashboard.
- [ ] Weather layer switching works.
- [ ] Selected-GP weather card works.
- [ ] 7-day forecast works.
- [ ] Forecast charts work.
- [ ] Farm Advisory works.
- [ ] Advisory rules are separated from UI components.
- [ ] Crop selection works.
- [ ] Environmental context works when data exists.
- [ ] Spatial variation is calculated dynamically.
- [ ] Loading states exist.
- [ ] Error states exist.
- [ ] Empty states exist.
- [ ] Desktop UI is polished.
- [ ] Mobile UI is usable.
- [ ] No fabricated weather/environment values.
- [ ] Existing backend/model outputs are reused.
- [ ] Map payload is kept reasonably small.
- [ ] Stage 1/Stage 2 terminology does not dominate the user interface.

---

# 38. Final instruction to Codex

Do not treat this as a request to reproduce a static screenshot.

Build a **functional HyperWeather application** inspired by the uploaded reference.

The final product should feel like:

> **A professional hyperlocal agricultural weather intelligence platform.**

The central story is:

```text
Broad Weather Information
          ↓
Spatial Intelligence
          ↓
Gram Panchayat Forecast
          ↓
Farm Advisory
          ↓
Better-Informed Farm Decisions
```

Prioritize:

1. correctness of data
2. map usability
3. clear weather information
4. useful farm advisory
5. visual polish
6. responsive design
7. maintainable architecture

Do not sacrifice data correctness for visual effects.

If a requested UI element cannot be backed by real project data, create the component architecture but display a clear unavailable state rather than inventing data.

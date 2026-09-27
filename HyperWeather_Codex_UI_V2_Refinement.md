# HyperWeather Frontend — UI V2 Refinement Instructions

## Goal

The current HyperWeather frontend is functionally successful. This task is a **focused visual refinement only**.

Do not rewrite the forecasting pipeline, model outputs, API, data adapter, or advisory logic unless a small UI-related change is necessary.

The main issues to fix are:

1. The primary dashboard currently requires vertical scrolling.
2. The complete dashboard should fit into one desktop viewport.
3. There is too much unused whitespace inside and between some cards.
4. Typography is too small relative to the available space.
5. Important information should be larger and easier to read.
6. The final composition should closely follow the newly supplied V2 reference image.

---

## 1. USE THE NEW IMAGE AS THE PRIMARY VISUAL REFERENCE

**Yes: use the newly supplied image together with the original reference image.**

The new image is the target for this refinement.

Use it to guide:

- overall proportions
- card arrangement
- typography scale
- information density
- map size
- sidebar width
- forecast layout
- advisory placement
- environmental-context placement
- spatial-variation placement
- crop-advisory placement
- spacing and visual hierarchy

Do not embed the screenshot into the application. Recreate it with real React components.

The image is a **visual reference only**. Do not copy its example weather values or dates unless those values actually exist in the application data.

---

# 2. PRIMARY REQUIREMENT: ONE-VIEW DESKTOP DASHBOARD

Target desktop viewport:

```text
1366 × 768 px
```

Also test:

```text
1440 × 900
1536 × 864
1920 × 1080
```

At 1366×768, the main dashboard must show all important information without browser-level vertical scrolling:

```text
Header
+
Sidebar
+
Hero/title
+
Location selector
+
Interactive map
+
Selected GP weather
+
7-day forecast
+
Farm advisory
+
Local environmental context
+
Spatial weather variation
+
Crop advisory
```

Do not solve this simply by shrinking everything.

Instead:

- remove unnecessary whitespace
- reduce excessive padding
- reduce unnecessary vertical gaps
- use horizontal space more effectively
- restructure cards where useful
- increase important typography
- keep secondary labels smaller
- increase information density without making it cramped

---

# 3. TARGET COMPOSITION

Use approximately this structure:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER                                                                      │
├────────┬──────────────────────────────────────────────┬────────────────────┤
│        │                                              │                    │
│ SIDE   │                 MAIN MAP                     │ SELECTED GP        │
│ BAR    │                                              │ WEATHER            │
│        │                                              │                    │
├────────┴───────────────────────────┬──────────────────┴────────────────────┤
│ 7-DAY FORECAST                      │ FARM ADVISORY                         │
├────────────────────────────────────┼───────────────────────────────────────┤
│ LOCAL ENVIRONMENTAL CONTEXT        │ SPATIAL VARIATION + CROP ADVISORY     │
└────────────────────────────────────┴───────────────────────────────────────┘
```

Adapt the exact proportions to the existing application.

---

# 4. HEADER

Keep the existing header.

Recommended scale:

- product name: 22–26px
- subtitle: 11–13px
- navigation: 14–15px
- controls: comfortable clickable size

Target header height:

```text
~60–68px
```

Do not allow the header to consume excessive vertical space.

---

# 5. SIDEBAR

Keep the dark vertical sidebar.

Target width:

```text
~70–88px
```

Items:

```text
Home
Map
Forecast
Advisory
Insights
About
```

Use icon + label.

Increase readability slightly but keep the sidebar compact.

Do not let the sidebar consume unnecessary horizontal space.

---

# 6. MAIN CONTENT SPACING

Current UI has too much whitespace.

Reduce:

- outer page padding
- card gaps
- top/bottom margins
- internal card padding
- empty vertical areas

Suggested starting values:

```text
Page horizontal padding: 18–28px
Card gap: 10–16px
Card padding: 14–20px
```

Adjust after visual testing.

The content should visually fill the desktop viewport.

---

# 7. HERO TITLE

Keep:

```text
Good decisions start with your local forecast.
```

Use green emphasis on:

```text
your local forecast
```

Recommended:

```text
Heading: 26–32px
Subtitle: 14–16px
```

Keep the breadcrumb aligned to the right.

The title area should consume approximately:

```text
70–80px
```

of vertical space, not more.

---

# 8. LOCATION PANEL

Keep:

```text
State
District
Block
Gram Panchayat
```

Use readable controls.

Suggested:

```text
Labels: 11–12px
Control text: 14–15px
Control height: ~38–42px
```

Reduce excessive gaps between selectors.

Weather layer buttons should remain clearly clickable.

---

# 9. MAP

The map is the primary visual element.

It should show:

- GP markers
- selected GP
- GP boundaries
- block boundary when available
- zoom controls
- legend
- weather-layer control

Increase usable map height compared with the current UI.

Remove any large blank area underneath the map.

If useful, calculate map height based on available viewport height rather than using an unnecessarily fixed small height.

Do not break dropdowns, overlays, or mobile behavior by globally applying `overflow:hidden`.

---

# 10. SELECTED GP CARD

Increase visual hierarchy.

Example:

```text
ADVALPAL (GP-254346)

26.5°C
```

Suggested sizes:

```text
GP name: 18–22px
Primary temperature: 40–48px
Metric values: 14–16px
Metric labels: 11–12px
```

Metrics:

```text
Humidity
Rainfall
Wind speed
Wind direction
```

Use actual data.

Do not invent timestamps.

Keep the card vertically compact.

---

# 11. WEATHER INSIGHT

Keep the green Weather Insight card.

Recommended:

```text
Title: 14–15px
Body: 12–14px
```

Remove unnecessary empty space beneath it.

Generate the text from actual forecast data where possible.

Use cautious language rather than absolute claims.

---

# 12. 7-DAY FORECAST

The seven forecast columns must fit horizontally in one card.

Each day:

```text
D1
Date
Weather icon
Temperature
Rainfall
Wind
```

Suggested sizes:

```text
Day label: 12–13px
Date: 11–12px
Temperature: 20–24px
Rainfall/wind: 11–13px
```

The forecast should become more readable, not smaller.

Use the actual data.

Do not copy incorrect labels from the visual reference. The real sequence must remain:

```text
D1 D2 D3 D4 D5 D6 D7
```

with actual dates.

---

# 13. FARM ADVISORY

Farm Advisory must remain visible in the same viewport.

Keep triggered advisory cards such as:

```text
Rainfall watch
Humidity watch
Wind advisory
Temperature advisory
```

Only show advisories when their existing rules trigger.

Use a compact structure:

```text
Rainfall watch       D4–D5
Higher daily rainfall is present.
Consider: Review irrigation and field drainage.
                         2/7 Days
                         Rule Match
```

Increase text readability while reducing unnecessary vertical padding.

IMPORTANT:

Do not call rule-match counts "confidence".

Continue using terminology such as:

```text
Rule Match
2/7 Days
```

unless a genuine calibrated confidence value exists.

---

# 14. BOTTOM INFORMATION AREA

Make the bottom section compact and information-dense.

Preferred conceptual layout:

```text
┌──────────────────────────────┬───────────────────┬──────────────────────┐
│ Local Environmental Context  │ Spatial Variation │ Crop Advisory        │
└──────────────────────────────┴───────────────────┴──────────────────────┘
```

Adapt this if needed so Farm Advisory remains prominent and the dashboard does not become crowded.

All important bottom information should be visible without page scrolling at 1366×768.

---

# 15. LOCAL ENVIRONMENTAL CONTEXT

Use compact tiles:

```text
Elevation
46 m

Slope
6.3°

Land cover
Trees

Soil clay
32.6%
```

Suggested:

```text
Values: 16–20px
Labels: 11–12px
```

Remove unused space.

Use actual data and preserve "Not available" behavior.

---

# 16. SPATIAL WEATHER VARIATION

Use three compact metrics:

```text
Block average
<dynamic>

GP forecast range
<dynamic>

Selected GP
<dynamic>
```

Then a short explanation:

> Gram Panchayats within this block can have different forecast conditions.

Calculate these values dynamically.

Do not hard-code example values.

---

# 17. CROP ADVISORY

Keep the crop configuration UI:

```text
Crop
[ Select crop ▼ ]

Growth stage
[ Select stage ▼ ]
```

If not configured:

```text
Crop profile not configured

Select a locally approved crop profile to show
crop-specific insights.
```

Do not invent crop recommendations.

Keep this card compact.

---

# 18. TYPOGRAPHY

The current implementation is too small.

Use a stronger hierarchy:

```text
Hero heading       26–32px
Section heading    17–20px
Card title         15–18px
Primary metric     24–48px
Body text          13–14px
Control text       14–15px
Small label        10–12px
Metadata           10–11px
```

Do not increase every font equally.

The goal is clear hierarchy.

---

# 19. REMOVE EMPTY SPACE

Specifically inspect and reduce:

- empty area under the map
- empty area inside selected-GP card
- excessive card padding
- excessive gaps between sections
- unused space inside environmental cards
- oversized advisory cards
- oversized title area

Use recovered space to increase important text sizes.

---

# 20. DESKTOP VIEWPORT IMPLEMENTATION

The main dashboard should behave like a desktop dashboard shell.

It is acceptable to use a viewport-constrained layout for the primary desktop dashboard, but do not blindly apply:

```css
overflow: hidden;
```

to the entire application.

Do not break:

- dropdown menus
- map controls
- dialogs
- GP detail pages
- mobile layouts

Instead, constrain the main dashboard intelligently and allow normal scrolling on secondary pages/mobile.

---

# 21. RESPONSIVE DESIGN

The no-scroll requirement applies primarily to the desktop dashboard.

Mobile may use normal vertical scrolling.

Tablet may use a two-column layout.

Desktop should use the full information-dense composition.

---

# 22. DATA BEHAVIOR MUST NOT CHANGE

Continue using the existing working data:

- Sinnar — 113 GPs
- Baramati — 99 GPs
- Bicholim — 16 GPs
- existing forecast outputs
- existing GP boundaries
- existing terrain data
- existing soil data
- existing land-cover data
- existing advisory rules

Do not fabricate missing values.

Keep:

```text
Not available
```

when appropriate.

---

# 23. WIND

Keep the current behavior:

```text
Model output: m/s
Frontend display: km/h
```

Do not alter stored model data.

Clearly display:

```text
km/h
```

---

# 24. SOURCE / RUN LABELING

Keep:

```text
Latest available model output
```

and show forecast date/run time where available.

Do not invent dates or times.

---

# 25. NO FUNCTIONAL REGRESSIONS

After visual changes verify:

- State selector
- District selector
- Block selector
- GP selector
- map markers
- GP boundary
- block boundary
- weather layers
- selected GP
- forecast
- graph view
- farm advisory
- environmental context
- spatial variation
- crop advisory
- GP detail route

---

# 26. BUILD CHECK

Run:

```bash
npm run build
```

The build must succeed.

Then visually test:

```text
1366 × 768
1440 × 900
1536 × 864
1920 × 1080
```

---

# 27. ACCEPTANCE CRITERIA

At 1366×768:

- [ ] No browser-level vertical scrolling on the primary dashboard.
- [ ] Header visible.
- [ ] Sidebar visible.
- [ ] Hero title visible.
- [ ] Location selector visible.
- [ ] Full map visible.
- [ ] Selected GP card visible.
- [ ] Current weather visible.
- [ ] All 7 forecast days visible.
- [ ] Farm Advisory visible.
- [ ] Environmental context visible.
- [ ] Spatial variation visible.
- [ ] Crop advisory visible.
- [ ] Typography is comfortably readable.
- [ ] No large unused blank areas.
- [ ] Cards do not feel unnecessarily tall.
- [ ] No horizontal page overflow.
- [ ] Map remains usable.
- [ ] Dropdowns remain usable.
- [ ] Existing functionality remains intact.

---

# 28. IMPORTANT: REFERENCE IMAGE IS VISUAL ONLY

Follow the newly provided reference image for:

- proportions
- typography
- spacing
- map prominence
- sidebar
- forecast composition
- advisory composition
- bottom information layout
- overall density

But do not copy its example values.

The reference image may contain illustrative dates/values/layout artifacts. Always use the real application's data.

---

# 29. FINAL INSTRUCTION

The current frontend is already functionally successful.

**Do not rebuild it from scratch.**

Perform a focused V2 visual refinement with one central goal:

> **Fit the complete dashboard into one desktop viewport while making the information larger, clearer, denser, and more visually balanced.**

The desired user experience is:

```text
WHERE?
  ↓
WHAT WEATHER?
  ↓
WHAT WILL HAPPEN?
  ↓
WHAT IS DIFFERENT LOCALLY?
  ↓
WHAT SHOULD I CONSIDER FOR THE FARM?
```

The result should look and feel like a professional weather-intelligence command center.

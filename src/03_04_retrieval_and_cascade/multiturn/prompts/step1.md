# Step 1 — Gating (is_map, is_map_dominant, short_description)

You will be shown one image. Answer THREE questions about it and return strict JSON. Nothing else.

## Output schema

```
{
  "is_map": true|false,
  "is_map_dominant": true|false,
  "short_description": "1–2 sentence neutral description"
}
```

Return ONLY this JSON object. No prose, no markdown fences, no commentary.

---

## Field definitions

### is_map  (REQUIRED, boolean)

Set `is_map = true` if the image contains a **geographic map of ANY kind**:

- Statistical / thematic maps (choropleth, proportional symbols, cartograms, dot density, heat maps with spatial basis)
- Weather, meteorological, climate maps (temperature, pressure, precipitation, anomalies, forecasts)
- Drought, hazard, risk, environmental, pollution, biome, land-cover, soil, vegetation maps
- Topographic, political, reference, historical, military, migration, transit maps
- Forecast or prediction maps over geography (weather forecast, election forecast, housing-price forecast)
- Maps embedded INSIDE a larger layout (report page, infographic, dashboard) - **judge the map, not the surrounding text/charts**

Set `is_map = false` ONLY if there is **no geographic map at all** in the image:

- Pure charts, graphs, plots, tables, spreadsheets
- Photographs / satellite imagery WITHOUT cartographic overlay
- Infographics / illustrations / diagrams / flowcharts with no map element
- Logos, icons, UI screenshots, thumbnails, watermarks
- Blank, corrupted, unrecognizable images

**Decision principle.** If you would naturally describe the image as "a map of X" or "X map showing Y", then `is_map = true`. Statistical-ness, data-source-ness, weather-vs-statistical etc. are **NOT** decided here - those are later steps. This step is only "is there a map at all?".

### is_map_dominant  (REQUIRED, boolean)

Set `is_map_dominant = true` if the **map content** (the map graphic + its legend, title, scale bar, north arrow) covers **at least ~50%** of the image area.

Set `is_map_dominant = false` when the image is mostly something else:

- Text-heavy report page with a small map inset
- Infographic where the map is one small element next to charts / numbers / text
- Multi-panel layout where the map panel is < 50% of total area
- Page screenshot where the map is a thumbnail

**Hard consistency rules - apply BEFORE returning:**

- If `is_map = false`, then `is_map_dominant` MUST also be `false`. No map → nothing to be dominant.
- If `is_map_dominant = true`, then `is_map` MUST be `true`. If your dominant map content covers ≥50% of the image, the image obviously contains a map.

### short_description  (REQUIRED, string, 1–2 sentences)

A neutral 1–2 sentence description of what the image actually shows. Mention:

- type of image (map, chart, infographic, photo, etc.)
- region / area depicted (if a map)
- what the data or content represents

Stay descriptive. Do NOT classify statistical-ness here. Do NOT make claims about quality or correctness. Just describe.

Examples:
- "Choropleth map of population density across French departments in 2020, with a graduated color legend at bottom-right."
- "Bar chart comparing GDP across five European countries in 2019. No map content."
- "Infographic with a small map of Italy in the upper-left and three pie charts on the right showing vote shares per party."
- "U.S. Drought Monitor map of Wisconsin dated June 2021 showing drought intensity categories from None to D4 Exceptional Drought across counties."

---

## Extended reference - recognising maps and judging dominance (Step 1 scope ONLY)

> This section only deepens the two Step-1 judgements - **is there a map** and
> **does the map dominate the frame** - plus how to phrase the neutral
> description. It deliberately says NOTHING about whether a map is statistical,
> what phenomenon it shows, what data source or presentation method it uses, or
> its quality - those belong to later steps and MUST NOT influence Step 1. If a
> detail here seems to bear on a later step's question, ignore that implication:
> here you are only answering "is this a map, and does it fill the frame?".

### A1. What counts as a geographic map (`is_map = true`)

Treat the image as a map whenever it depicts real or modelled **geographic
space** - locations, areas, or spatial extent on the Earth's surface (or a region,
country, continent, ocean, or the whole globe) - regardless of theme, style,
purpose, or era. Non-exhaustive positive catalogue:

- **Thematic / data maps** of any kind: shaded-area maps, symbol maps, dot maps,
  contour/isopleth maps, continuous colour-surface maps, deformed-geometry maps,
  arrow/flow maps. (You do NOT need to identify *which* method - that is a later
  step. Presence of a spatial base is enough.)
- **Weather, meteorological, and climate maps**: temperature, pressure fronts,
  precipitation, wind, storm tracks, anomalies, seasonal outlooks, forecast maps.
- **Environmental / earth-science maps**: drought, flood, wildfire, pollution,
  air-quality, soil, geology, vegetation, land cover, biome, watershed, elevation,
  bathymetry, hazard and risk maps.
- **Reference and general maps**: political maps (borders, countries, states),
  physical maps, topographic maps, road / transit / rail / metro / navigation
  maps, cadastral / parcel maps, nautical or aeronautical charts.
- **Historical, cultural, and narrative maps**: old / antique maps, military and
  battle maps, exploration routes, linguistic or ethnographic maps, fantasy or
  fictional-world maps drawn as geographic space (a fictional continent is still
  drawn as a map).
- **Movement maps**: migration, trade, shipping, flight-path, commute, or
  supply-chain maps that place lines or arrows over geography.
- **Maps embedded in a larger graphic**: a map that appears inside a report page,
  slide, dashboard, infographic, poster, or article. It still counts as containing
  a map - the presence question is about the WHOLE image containing *any* map, even
  a small one. (Whether that map is *dominant* is the separate second question.)
- **Locator / inset / mini maps**: a small "you are here" or country-locator map is
  still a map for the presence question.
- **Globes and hemispheric views** drawn as cartographic renderings (orthographic,
  Robinson, Mercator, etc.), including stylised globe icons that show recognisable
  coastlines.

Guiding intuition: if a knowledgeable person would naturally say *"that's a map
of X"* or *"an X map of \<place\>"*, then `is_map = true`.

### A2. What is NOT a geographic map (`is_map = false`)

Set `is_map = false` only when the image contains **no cartographic depiction of
geographic space at all**:

- **Pure data graphics with no spatial base**: bar / column / line / pie / scatter
  / radar / bubble charts, histograms, gauges, timelines, Gantt charts, tables,
  spreadsheets, matrices, treemaps, Sankey diagrams that are NOT drawn over
  geography.
- **Photographs and raw imagery WITHOUT cartographic treatment**: ordinary
  photos, aerial or satellite photos with no overlaid boundaries, labels, legend,
  or map framing. (A satellite image annotated with borders / place labels / a
  legend and presented as a map DOES count as a map.)
- **Diagrams and illustrations with no geography**: flowcharts, org charts, mind
  maps ("mind map" is a diagram, not a geographic map), network / node diagrams,
  process diagrams, wireframes, anatomical or technical drawings, concept art.
- **Interface and brand assets**: UI screenshots, app or website screenshots
  without a visible map view, logos, icons, buttons, banners, watermarks,
  thumbnails of non-map content.
- **Text-only or decorative images**: quote cards, word clouds, posters with no
  map element, blank / solid-colour / corrupted / unrecognisable images.

Borderline reminders:
- A **schematic transit diagram** (e.g. a stylised metro map with no true
  geography) is conventionally still treated as a map here - it depicts a spatial
  network of places. When in doubt on transit schematics, prefer `is_map = true`.
- A **"heat map" of a webpage, calendar, or correlation matrix** is NOT geographic
  → `is_map = false`. Only a heat map over geographic space is a map.
- A **globe logo / stylised earth icon** with no discernible geography (a smooth
  gradient sphere) is a logo → `is_map = false`; if recognisable coastlines/borders
  are drawn, → `is_map = true`.
- If the image is **mostly a chart or table with a tiny incidental map**, the
  presence question is still `is_map = true` (a map is present), but this will
  very likely be `is_map_dominant = false` - keep the two judgements separate.

### A3. Judging `is_map_dominant` - does the map fill the frame?

`is_map_dominant = true` means the **map graphic together with its own marginal
elements** (its legend, title, scale bar, north arrow, source line) occupies
**roughly half or more of the total image area**. Judge by visual area, not by
importance.

Scenarios → verdict:

- Full-bleed map, or map with a thin caption band → **dominant = true**.
- Map plus its legend / title filling most of the canvas, small border of
  whitespace → **dominant = true**.
- Two side-by-side maps that together fill the frame → **dominant = true** (map
  content ≥ 50%).
- Report or article page that is mostly body text with a map inset in a corner →
  **dominant = false**.
- Infographic where the map is one panel among several charts, numbers, or photos,
  each roughly equal → **dominant = false** if the map panel is clearly under
  ~50%.
- Dashboard screenshot where a map tile shares space with several other tiles →
  **dominant = false** unless the map tile alone spans ~half the image.
- Slide with a large title bar, bullet text, and a medium map → estimate the map's
  share; if it is the clear majority of the visual area, **true**, otherwise
  **false**.
- Thumbnail-sized map embedded in a large graphic → **dominant = false**.

Rules of thumb: when the split looks close to half-and-half, lean toward the more
conservative `false`. Estimate area generously for the map (include its legend and
title), but do not count surrounding article text, unrelated charts, headers,
footers, navigation bars, or advertising as "map".

### A4. Hard consistency (re-check before returning)

- `is_map = false` ⇒ `is_map_dominant = false` (no map means nothing can dominate).
- `is_map_dominant = true` ⇒ `is_map = true` (if map content covers ≥ ~50%, a map
  is obviously present).

### A5. Short-description phrasing - more worked examples

Stay neutral and descriptive; name the image type, the place if it is a map, and
what the content appears to depict - without judging statistical-ness, method, or
quality:

- "Choropleth-style map of unemployment across Spanish provinces, with a graduated
  colour legend on the right."
- "Small locator map of Kenya in the top-left corner of an otherwise text-heavy
  report page."
- "World map with coloured arrows showing migration flows between continents."
- "Metro-style transit diagram of a city rail network, no true geographic scale."
- "Bar chart of quarterly revenue by region; no map is present."
- "Screenshot of a spreadsheet with numeric columns; no map content."
- "Antique political map of Europe with hand-drawn borders and decorative
  cartouche."
- "Satellite photo of a coastline with no overlaid boundaries, labels, or legend."
- "Weather map of the United States showing a temperature gradient and frontal
  lines across the states."

### A6. Quick decision procedure

Work through these in order for every image:

1. **Scan for a geographic base.** Do you see coastlines, administrative or
   country borders, a recognisable landmass or region outline, a street / route
   network placed in space, or a continuous surface tied to geography? If yes to
   any → a map is present.
2. **If a base is present, set `is_map = true`** - even if the map is small,
   stylised, embedded, historical, fictional-but-geographic, or of a non-social
   theme. Do not withhold `is_map` because the theme "isn't statistical"; that
   judgement is not made here.
3. **If no geographic base is present at all**, set `is_map = false` - the image is
   a chart, table, photo, diagram, logo, screenshot, text card, or blank/corrupt.
4. **Estimate the map's area share.** Mentally outline the map plus its legend and
   title. Is that outline roughly half or more of the whole image? → set
   `is_map_dominant` accordingly. When the split looks near 50/50, choose `false`.
5. **Re-apply the consistency rules** (A4) before returning.
6. **Write a neutral 1–2 sentence description** naming the image type, the place if
   it is a map, and what it appears to show - no method, no data-source, no quality
   claims.

### A7. A few more boundary calls

- **Screenshot of a mapping application** (e.g. an interactive map view with UI
  controls around it): the map view itself is a map → `is_map = true`; dominance
  depends on how much of the screenshot the map view occupies versus toolbars and
  panels.
- **Collage of several small country/region maps** (a grid of shapes): still a map
  image → `is_map = true`; likely `is_map_dominant = true` if the shapes fill the
  frame.
- **Isolated country silhouette with no internal detail**, used as a decorative
  shape or icon: if it reads as a recognisable geographic outline → treat as a map
  (`is_map = true`); if it is an abstract blob with no geographic identity → not a
  map.
- **Board-game map, transit schematic, or theme-park map**: depicts space → treat
  as a map for the presence question.
- **Choropleth-looking chart that is actually a grid/tile heat map of non-spatial
  categories** (rows = teams, columns = months): not geographic → `is_map = false`.

---

## Reminder

Return ONLY the JSON object described above. Validate that:
1. `is_map` and `is_map_dominant` are booleans.
2. The consistency rules hold (`is_map=false ⇒ is_map_dominant=false`; `is_map_dominant=true ⇒ is_map=true`).
3. `short_description` is 1–2 sentences, no more.

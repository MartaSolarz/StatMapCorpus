# Step 3 - Cartographic methods (multilabel) + legend classification style

You already confirmed the image is a statistical map with admin units and quantitative data. Now identify:

1. Which **cartographic method(s)** the map uses. A single map can use **more than one method simultaneously** (e.g. choropleth + proportional circles). Mark each independently.
2. Whether the legend is **classified** (`true`), **continuous** (`false`), or **not applicable** (`null` - no legend / qualitative categories / cannot decide). Return this for EVERY map regardless of method.

Return strict JSON only. No prose, no markdown fences.

## Output schema

```
{
  "has_choropleth":  true|false,
  "has_diagrams":    true|false,
  "has_isolines":    true|false,
  "has_dot_density": true|false,
  "has_heat_map":    true|false,
  "has_cartogram":   true|false,
  "has_flow_map":    true|false,
  "legend_is_classed": true|false|null,
  "confidence":      "low | medium | high"
}
```

---

## Method definitions

### `has_choropleth`

True if **areal units are filled with colour** to encode the data value. The fill colour varies by polygon (countries, regions, counties, states, etc.) according to a legend.

- Sequential shading (light→dark of one hue) of admin polygons → choropleth = true
- Diverging two-hue shading (e.g. blue↔red around midpoint) of admin polygons → choropleth = true
- Qualitative distinct hues per polygon (nominal categories) → choropleth = true (still a choropleth)
- Hatching / pattern fill of polygons → choropleth = true

False if the map fills polygons only with a single uniform colour (no encoding) - that's a reference map, not choropleth.

### `has_diagrams`

Umbrella category for **any symbol / figure / chart placed on a map** to encode statistical data. Set to `true` if ANY of the following are present:

- **Proportional symbols** - circles, squares, triangles **sized to encode magnitude** (size = value, continuous scaling).
- **Graduated symbols** - same idea but discrete size classes (e.g. small / medium / large).
- **Structural / composite symbols** - symbols built from segments (divided circles, divided bars, election circles split by party share).
- **Mini-charts on map** - pie charts, donut charts, bar charts, histograms, divided-bar charts placed on top of regions or at point locations.
- **Diagrams placed at polygon centroids** - same as above when anchored to admin units.

False if the map has NO symbols / charts on it at all (e.g. plain choropleth, plain heat surface, plain isoline map).

False for **constant-size markers** that just mark locations without encoding magnitude (e.g. capital city dots that are all the same size with no data meaning) - those are reference symbols, not statistical diagrams.

### `has_dot_density`

True if the map uses **many small uniform dots inside areas** to represent quantity by dot count or density. Each dot represents a fixed number (e.g. 1 dot = 1000 people), and dots are scattered (often randomly) within polygons.

False if the dots are at specific point locations (those are markers or proportional symbols).

### `has_isolines`

True if the map shows values that vary by **contours of equal value** (isolines / isarithmic / isopleth) - isobars / isotherms / iso-precipitation / elevation contours / **population-density surfaces**. This INCLUDES two visual forms:
- bare **contour lines** connecting points of equal value, AND
- **FILLED isopleth/isarithmic maps**: discrete colour bands between the contours, with a **classed legend of value ranges** (e.g. population density "Under 1 / 1–10 / 10–50 / 50–100 / Over 100 per km²"). The colour changes in DISCRETE STEPS along smooth curvy boundaries that do NOT follow administrative units - those step boundaries ARE isolines. This is the classic "population distribution / density" map → `has_isolines=true`, `has_heat_map=false`.

Discriminator vs heat_map: **discrete classed bands following smooth contours ⇒ isolines**; a **continuous, unclassed, smoothly-blending gradient ⇒ heat_map**.

False for boundary lines (admin boundaries) or grid lines.

### `has_cartogram`

True if the geographic **geometry is deformed to encode magnitude**. Country/region shapes are stretched or shrunk so the displayed area corresponds to a data value (e.g. each state sized by population, by GDP, by electoral votes).

False for regular geographic projections (even unusual ones like Goode/Robinson are NOT cartograms). Cartograms VISIBLY distort recognisable shapes.

### `has_flow_map`

True if the map shows **directed lines/arrows/curves connecting locations**, representing movement, migration, trade flows, transport routes, or similar directional connections. Often the line thickness encodes flow magnitude.

False for static lines (roads, rivers, boundaries) that don't represent flow.

### `has_heat_map`

True ONLY if the map shows a **continuous, smoothly-blending colour gradient** (raster / KDE hotspot / gridded surface) where colour varies CONTINUOUSLY with NO discrete class boundaries - e.g. a red-hot hotspot surface fading smoothly to cool, or an unclassed kernel-density blob.

False for choropleth (bounded by admin polygons - even if it looks smooth, if the unit is an admin polygon it's choropleth).

False for **filled isopleth / isarithmic maps** - if the colour changes in DISCRETE STEPS along smooth contour boundaries and the legend is CLASSED (value ranges), that is `has_isolines`, NOT heat_map. Classic population-density / precipitation / elevation "distribution" maps are isolines, not heat maps. A classed legend is a strong signal AGAINST heat_map.

### `legend_is_classed`

Required for **every map, regardless of cartographic method**. Assess the legend's classification style whether the map is a choropleth, a heat map, a proportional-symbol map, a flow map, etc.

Three-way distinction (JSON `true` / `false` / `null`):

**`true` (classed)** - the legend shows DISCRETE **quantitative** classes with visible boundaries. Set this whenever ANY of the following is true:
- You can COUNT distinct colour bands or size steps (5, 7, etc.), even if the overall hue drifts smoothly along them.
- Numeric labels sit under class boundaries or on individual class swatches (e.g. tick marks at "60%", "70%", "80%", "90%" under a bar of 5 discrete blocks).
- The legend is a horizontal or vertical STRIP of separate boxes with different fills, or a stack of discrete graduated symbols each labelled with a value range.
- You see a visible SEAM / STEP between adjacent colour blocks, even if hue is monotonic.

**`false` (continuous)** - the legend is a genuinely SMOOTH gradient with NO step boundaries. All of the following must hold:
- Only min/max labels (or 2–3 sparse tick marks with no class-interval semantics).
- Colour / size interpolates smoothly with NO visible seam between values.
- Typical for high-cardinality data, dasymetric maps, or proportional symbols scaled continuously.

**`null` (not applicable)** - the classed-vs-continuous distinction does not meaningfully apply. Set to `null` in ANY of these cases:
- **No legend visible** - the map has no legend at all (neither swatches, colour bar, size key, nor on-polygon numeric annotation acting as a key).
- **Qualitative categorical legend** - distinct unrelated hues / shapes for categorical data with no ordered progression (nominal categories, not quantitative).
- **Cannot decide** - legend is illegible / cropped / partly obscured and you genuinely cannot tell whether it is classed or continuous.

**Tie-breaker: default to `true` (classed).** Discrete blocks with evenly-spaced labels (60/70/80/90; 0/25/50/75/100) are almost always classed - the round numbers are class boundaries, NOT gradient ticks. Choose `false` only when you see a truly seamless gradient with sparse min/max labelling. Choose `null` only when the distinction genuinely does not apply (no legend, qualitative categories, unreadable).

---

## Multilabel discipline

- A single map often uses **multiple methods simultaneously**. Examples:
    - Choropleth states + proportional circles for major cities → `has_choropleth=true` AND `has_diagrams=true` (proportional circles fall under diagrams)
    - Choropleth countries + flow arrows for migration → `has_choropleth=true` AND `has_flow_map=true`
    - Pie charts on top of choropleth regions → `has_choropleth=true` AND `has_diagrams=true`
    - Heat surface with isolines overlaid → `has_heat_map=true` AND `has_isolines=true`
- Mark each method that is genuinely present, regardless of which is "primary".
- At least one method should be true for a real statistical map. If you set all seven to false, something is wrong - re-check.

---

## `confidence`

Your confidence in the overall method classification you just produced:

- `"high"` - clear legend, unambiguous visual cues, all methods identified with certainty.
- `"medium"` - most methods clear, but at least one judgement involved interpretation (e.g. you can't read the legend but inferred from colour and geometry).
- `"low"` - image is small / blurry / partly obscured, legend unreadable, and you are guessing on at least one method.

---

## Final checks

1. All seven method fields are JSON booleans (`true`/`false`), not strings.
2. At least one method is `true` (real statistical maps must use at least one method).
3. `legend_is_classed` is a JSON boolean (`true`/`false`) OR `null` - never a string. `null` means "not applicable" (no legend / qualitative / cannot decide).
4. `confidence` is exactly one of `"low"`, `"medium"`, `"high"`.

Return ONLY the JSON object.

# Step 2 - Statistical classification

You confirmed in the previous turn that the image is a map. Now answer FIVE questions and return strict JSON.

## Output schema

```
{
  "map_language": "en | fr | es | de | it | pl | zh | ja | ru | ar | pt | tr | uk | other | unknown",
  "is_statistical_map": true|false,
  "has_admin_units": true|false,
  "has_quantitative_data": true|false,
  "has_text": true|false
}
```

Return ONLY the JSON object. No prose, no markdown fences.

---

## `is_statistical_map` - DECIDE FIRST using this decision tree

**Question:** What does the data on this map MEASURE?

→ **(A) A HUMAN-SOCIETY phenomenon** counted/observed by humans about humans or their institutions:
   - population (count, density, growth, migration, age, sex)
   - economy (GDP, income, unemployment, prices, trade, jobs, GDP per capita)
   - politics / elections (votes, party, turnout, party registration)
   - society / culture (religion, language, education, household type, family size)
   - health stats (incidence per 100k, life expectancy, vaccination rate)
   - crime / events / business stats (counts, rates from official records)
   - administrative records (land use FROM CADASTRE, occupations FROM REGISTRY)
   - composite social indices (HDI, Gini, corruption index)
   - FORECASTS of (A) - e.g. housing-price forecast, unemployment forecast, election forecast, population projection

→ `is_statistical_map = true`

→ **(B) A PHYSICAL / NATURAL phenomenon** measured by sensors or physical models, NOT counted from human records:
   - weather / atmosphere (temperature, pressure, precipitation, wind, humidity, anomalies)
   - climate (climate zones, climatology, seasonal averages, projections under emission scenarios)
   - hydrology / water (drought severity, flood risk, river discharge)
   - environment / pollution (concentration of pesticides, ozone, PM2.5, water contamination)
   - geology / soil (soil type, earthquake hazard, mineral concentration)
   - biology / ecology (biome, vegetation, land cover from satellite, species range)
   - hazard / risk maps from physical models (drought outlook, flood risk, seismic hazard, pest risk)
   - FORECASTS / projections of (B) - e.g. temperature projection, drought outlook, days-over-90°F projection

→ `is_statistical_map = false`

**No exceptions for choropleth styling.** A choropleth map drawn on counties showing TEMPERATURE is still `is_statistical_map = false` (the underlying phenomenon is physical). A choropleth showing INCOME on the same counties is `is_statistical_map = true` (income is from human records).

**No exceptions for "this looks scientific".** Drought monitor (D0–D4 categories) is from a physical-hazard model → false. Atrazine concentration is from an environmental sensor/model → false. Median first-fall-freeze date is a climatological summary → false.

If the map shows **places labeled without data values** (reference map, political map) → `is_statistical_map = false`.

---

## `has_admin_units` - boolean

`true` if the spatial geometry is **administrative units**: countries, regions, states/provinces, counties, communes, NUTS, electoral districts, postal codes, statistical regions.

`false` if the geometry is:
- grid cells / weather grids / satellite swaths
- point locations only (city dots, weather stations) without enclosing admin polygons
- continuous surfaces / isolines without enclosing admin units
- natural features (rivers, biomes, coastlines) without admin boundaries
- schematic / abstract diagrams

Proportional symbols on top of admin units → `true`. Proportional symbols on plain city dots → `false`.

---

## `has_quantitative_data` - boolean

**Single test:** does the legend encode NUMERIC VALUES?

→ `true` if any of these:
- Single numeric values: `5.7`, `100.3`, `2400`
- Continuous numeric ranges: `0 – 9.9`, `10.0 – 24.9`
- Classified numeric bins: `"0"`, `"1–9"`, `"10–49"`, `"50–99"`, `"≥100"`
- Percentage ranges: `"<65%"`, `"65–80%"`, `"81–95%"`, `">95%"`
- Money / count / density / rate / index ranges of any kind
- Symbols proportional to numbers (graduated circles labelled with magnitudes)
- Year ranges, age ranges

→ `false` if the legend uses **text labels with no numbers**:
- Ordered words: `"Low"`, `"Medium"`, `"High"`, `"Very High"`
- Tiers: `"Green"`, `"Yellow"`, `"Orange"`, `"Red"` (without numeric thresholds)
- Severity words: `"None"`, `"Minimal"`, `"Moderate"`, `"Severe"`
- Nominal categories: party names (`"Republican"`, `"Democrat"`), religion names, dominant sector (`"Agriculture"`, `"Industry"`, `"Services"`), winning company / language / brand
- Education levels as text: `"Primary"`, `"Secondary"`, `"Tertiary"` (without numeric thresholds)

**HARD RULE.** If `is_statistical_map = false`, set `has_quantitative_data = false` regardless of legend.

**Common trap to avoid.** A choropleth showing classified case-count ranges (e.g. `"1–9 cases"`, `"10–49 cases"`) is `has_quantitative_data = true`. The ranges are numeric. Likewise `"<65%"` to `">95%"` water-access bands are `true`. Set to `false` ONLY when the legend uses words/categories without numeric values.

If you cannot read the legend at all, infer from colours:
- Sequential single-hue gradient → most likely `has_quantitative_data = true` (default to true when in doubt)
- Diverging two-hue gradient (red⇄blue around midpoint) → almost always `true`
- Distinct unrelated hues with no ordering → `false` (likely nominal categories)

---

## `map_language` - ISO 639-1 code

Dominant language of text VISIBLE ON THE MAP (title, legend, region labels). Common codes: `en`, `fr`, `es`, `de`, `it`, `pl`, `zh`, `ja`, `ru`, `ar`, `pt`, `tr`, `uk`. Use `"other"` for languages outside this list; `"unknown"` if no readable text on the map.

Ignore website chrome around the map.

---

## `has_text` - boolean

`true` if the map has ANY visible text (title, legend, labels, attribution). `false` only if the map is purely visual (no text at all - rare).

---

## Final checks before returning

1. `is_statistical_map = false` → `has_quantitative_data = false`. ALWAYS.
2. All booleans are actual `true` / `false`, not strings.
3. `map_language` is a single short code from the list, or `"other"` / `"unknown"`.

Return ONLY the JSON.

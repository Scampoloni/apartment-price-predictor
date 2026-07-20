# Conversational agent design

## Separation of responsibilities

The agent is an input adapter, not an independent rent model.

```text
German text
  → LLM JSON extraction
  → deterministic validation
  → shared Zurich regression pipeline
  → numeric result rendered by application code

validated fields
  → separate price-free LLM explanation
  → deterministic uncertainty and scope note
```

The earlier standalone seven-feature Random Forest and national BFS lookup were
removed because they conflicted with the Zurich-only suite and duplicated the
price-model responsibility.

## Extraction contract

Accepted JSON:

```json
{
  "rooms": 3.5,
  "area_m2": 85,
  "municipality": "Winterthur",
  "description": "mit Balkon"
}
```

Validation rejects:

- malformed or non-object JSON;
- missing or null required fields;
- non-numeric or out-of-range room and area values;
- empty municipalities;
- `price`, `rent`, `monthly_rent`, `predicted_price`, or
  `predicted_price_chf` fields.

Optional Markdown JSON fences are handled explicitly; arbitrary prefixes and
suffixes are not silently stripped.

## Price isolation

The explanation function accepts an `ApartmentQuery`, not a price. Its prompt
forbids numbers and currency, and validation rejects explanation output
containing digits, `CHF`, `Fr.`, `$`, or `€`. If validation fails, the
application uses a deterministic qualitative fallback.

The regression result and its RMSE evidence are appended only after the LLM
call, by application code.

## Unknown municipalities

Known support is read from the regression model metadata generated during
training. An unseen municipality can pass through
`OneHotEncoder(handle_unknown="ignore")` without a runtime error, but the UI
states that this is weaker geographic support. Safe execution is not described
as valid coverage.

## Tests

`tests/test_conversational_agent.py` covers valid extraction, malformed output,
missing fields, forbidden price fields, unknown municipalities, and price-like
explanations. Tests do not call an external API.

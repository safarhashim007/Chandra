# API

The read-only Region Pack frontend endpoints are available from `api.app`:

- `GET /api/frontend/regions`
- `GET /api/frontend/regions/{region_id}`
- `GET /api/frontend/regions/{region_id}/observations`

They expose acquisition provenance and the Moon-coordinate conventions but do
not return a registration verdict. The planned analysis/job endpoints remain
separate: `GET /health`, `POST /analyze`, `GET /jobs/{id}`, `GET /jobs/{id}/result`,
and `GET /artifacts/{id}`. Failures include stage, code, diagnostics, and a
recommended next action.

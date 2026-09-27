# HyperWeather frontend

This is a separate React application and read-only data adapter. It reads the existing files under `models/main_model/stage2_direct_gp_training`, `gp`, and `block`; it does not train models or modify the forecasting pipeline.

## Run locally

From the repository root, start the data API:

```powershell
python frontend/server.py
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. For a single-process production preview, run `npm run build` inside `frontend`, then start `python frontend/server.py`; the Python server serves the built frontend and its `/api` routes on port 8000.

## Deploy

Deploy the API to Render from the repository root. The included `render.yaml` starts `python frontend/server.py`. It serves the versioned forecast snapshots and GP data in this repository; it uses no secret environment variables. New forecasts must be committed (or moved to external storage) before they appear in the deployment.

Deploy the static site to Vercel with `frontend` as the Root Directory. Set the Vercel environment variable `VITE_API_BASE_URL` to your Render service URL, for example `https://hyperweather-api.onrender.com`, then redeploy. Do not set this variable to a secret: all `VITE_` variables are included in the browser bundle.

The API discovers blocks that have an existing `stage2_current_gp_forecast.csv`. It loads only the requested block and uses that block's GP geometry. Environmental fields display as unavailable when the repository has no matching source value. The external OpenStreetMap tile service is used for the basemap.

## Data notes

- Forecast Day 1 is the first date in the latest available seven-day output file. Historical or stale runs are labeled with their source run date when available.
- Wind is stored in the model output in m/s. Dashboard cards convert it to km/h; map legend and advisory thresholds use m/s.
- Advisory rules are prototype threshold rules and their “rule match” count reports how many available forecast days met a threshold. It is not calibrated confidence or validated agronomic guidance.
- Environmental context uses the existing GP terrain, soil, and land-cover CSV summaries. Soil texture is shown as clay fraction because that is the available field; soil class is not inferred.

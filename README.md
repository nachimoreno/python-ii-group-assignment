# Automated Daily Trading System

Streamlit-based trading MVP for the group assignment. The project combines:
- Part 1 pipeline components (data ingestion + ETL transformations).
- Part 2 components (SimFin API wrapper + web app + deployment-ready setup).

Detailed walkthrough: `docs/USER_GUIDE.md`
Code explanation: `docs/CODE_EXPLANATION.md`

## Team Members
- Add team member names here before submission.

## Features Implemented (Part 2 MVP)
- `PySimFin` object-oriented API wrapper with error handling and request throttling.
- Streamlit multipage app:
  - Home page (`app.py`)
  - Go Live page (`pages/1_Go_Live.py`)
- Shared ETL transformation function reused by batch scripts and the live app.
- Pluggable prediction interface:
  - `MockPredictor` (default)
  - `ModelPredictor` stub for future model artifact integration.
- Unit tests for wrapper, service logic, predictor behavior, and app helpers.

## Project Structure
- `data_ingestion.py`: Bulk ingestion from SimFin into parquet.
- `data_cleaning.py`: Shared transformations + enrichment write-out.
- `simfin_wrapper.py`: API wrapper and custom exceptions.
- `predictors.py`: Predictor contract, mock predictor, model stub.
- `go_live_service.py`: Go Live business logic.
- `app.py`: Home page.
- `pages/1_Go_Live.py`: Live analysis page.
- `tests/`: Automated tests.

## Local Setup
1. Create and activate environment:
   ```bash
   conda create -n trading-app python=3.11
   conda activate trading-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` at repository root:
   ```bash
   SIMFIN_API_KEY=your_simfin_key
   # Optional:
   # SIMFIN_BASE_URL=https://backend.simfin.com/api/v3
   ```
4. (Optional) Run ingestion/cleaning scripts:
   ```bash
   python data_ingestion.py
   python data_cleaning.py
   ```
5. Run app:
   ```bash
   streamlit run app.py
   ```

## Deployment (Streamlit Cloud)
1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app from the repo and set entrypoint to `app.py`.
3. Add secret in Streamlit Cloud settings:
   ```toml
   SIMFIN_API_KEY = "your_simfin_key"
   ```
4. Deploy and share the public URL.

Public app link: `https://your-streamlit-app-url.streamlit.app`

## Running Tests
```bash
pytest -q
```

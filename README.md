
# Movie Recommendation System — CF (Cosine) vs SVD
**Built by Nikhil Dulipala**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Nikhild2710/Movie-recommendations/blob/main/Movie_Recommender_Nikhil.ipynb)

## What this is
A practical comparison between:
- **User–User Collaborative Filtering (cosine)** — heuristic baseline
- **SVD Matrix Factorization** — learned model (Surprise)

Evaluated with **RMSE** on the **same train/test split**.

## Datasets
- **MovieLens 20M**: `rating.csv`, `movie.csv`
- **(Optional) TMDB 5k**: `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` (used for content features)

> Datasets are *not* in this repo. The notebook loads from your Google Drive or downloads via Kaggle.

## Run in Colab

### Option A — Use your Google Drive (fast after first setup)
1. Put CSVs in `MyDrive/movie-recs-data/`:
   - `rating.csv`, `movie.csv`
   - (optional) `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`
2. Open the notebook via the badge above.
3. Run the **Session Starter** cell (mounts Drive, loads CSVs).

### Option B — Download from Kaggle (no Drive needed)
The notebook includes a Kaggle block that downloads MovieLens/TMDB.  
Upload your `kaggle.json` when prompted. (Do **not** commit it to GitHub.)

### If SVD import fails in a fresh Colab
At the top of the notebook, run this one-time **environment fix** cell; it restarts the runtime:
```python
!pip -q install numpy==1.26.4 scikit-surprise==1.1.3
import os; os.kill(os.getpid(), 9)

Movie Recommendation System — CF (Cosine) vs SVD
Built by Nikhil Dulipala



What this is
A practical comparison between:

User–User Collaborative Filtering (cosine) — heuristic baseline

SVD Matrix Factorization — learned model (Surprise)

Evaluated with RMSE on the same train/test split.

Datasets
MovieLens 20M: rating.csv, movie.csv

(Optional) TMDB 5k: tmdb_5000_movies.csv, tmdb_5000_credits.csv (used for content features)

Datasets are not in this repo. The notebook loads from your Google Drive or downloads via Kaggle.

Run in Colab
Option A — Use your Google Drive (fast after first setup)
Put CSVs in MyDrive/movie-recs-data/:

rating.csv, movie.csv

(optional) tmdb_5000_movies.csv, tmdb_5000_credits.csv

Open the notebook via the badge above.

Run the Session Starter cell (mounts Drive, loads CSVs).

Option B — Download from Kaggle (no Drive needed)
The notebook includes a Kaggle block that downloads MovieLens/TMDB.
Upload your kaggle.json when prompted. (Do not commit it to GitHub.)

If SVD import fails in a fresh Colab
Run this one-time environment fix cell at the top; it restarts the runtime:

python
Copy
Edit
!pip -q install numpy==1.26.4 scikit-surprise==1.1.3
import os; os.kill(os.getpid(), 9)
Methods in this repo
User–User CF (cosine on mean-centered ratings)

Build user×item matrix from train

Predict rating for (u, m) via similarity-weighted neighbor ratings

Evaluate RMSE vs test

SVD (Surprise)

SVD(n_factors=100, random_state=42) on the same train

Predict test; report RMSE

Results (this run)
Method	Metric	Score
User–User Cosine	RMSE	~1.0700
SVD	RMSE	0.8831

SVD reduced RMSE from ~1.07 → 0.883 (lower is better).

Why SVD helps (plain English)
Cosine compares observed overlaps between users.
SVD learns latent factors (hidden taste dimensions) and generalizes to unseen user–movie pairs, so predictions are tighter.

Notebook structure
Session Starter (Colab + Drive/Kaggle setup)

(Optional) TMDB content prep

User–User CF (cosine) — RMSE

SVD — RMSE

Results table

Credits
MIT-licensed work referenced

Datasets: GroupLens MovieLens, TMDB.

License
MIT for code in this repo. Datasets must be obtained from their original sources under their terms.

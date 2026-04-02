# Optimization Algorithm Visualizer

Interactive Streamlit apps for four optimization algorithms.
Each student should fork this repo, deploys their own copy, and submits their Streamlit Cloud link.

## Apps included

| App | Algorithm | Problem |
|-----|-----------|---------|
| 1 | Unconstrained Minimization | SD vs Newton vs CG on any 2D function |
| 2 | Pareto Front | Multi-objective trade-off visualization |
| 3 | Genetic Algorithm | 0/1 Knapsack |
| 4 | Simulated Annealing | Exam Timetable Scheduling |

## Run locally

pip install -r requirements.txt
streamlit run Home.py

## Deploy to Streamlit Cloud (free — get a shareable link)

1. Fork or push this repo to your GitHub account
2. Go to https://share.streamlit.io
3. Click New app
4. Select your repo, branch = main, main file = Home.py
5. Click Deploy — you get a public link like:
   https://your-app-name.streamlit.app



## File structure

Home.py                          ← landing page
pages/
  1_Unconstrained_Minimization.py
  2_Pareto_Front.py
  3_Genetic_Algorithm.py
  4_Simulated_Annealing.py
requirements.txt
.streamlit/config.toml           ← dark theme
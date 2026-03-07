"""
App 3 — Genetic Algorithm Visualizer
Solves the 0/1 Knapsack problem step-by-step
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Genetic Algorithm", layout="wide")

st.title("Genetic Algorithm — Knapsack Problem")
st.markdown("Watch a **Genetic Algorithm** evolve a population to solve the 0/1 Knapsack problem.")

# ── Default items ─────────────────────────────────────────────────────────────
DEFAULT_ITEMS = [
    ("Water bottle",    2.0,  9),
    ("First aid kit",   1.5, 10),
    ("Tent",            4.0, 10),
    ("Sleeping bag",    3.0,  9),
    ("Torch",           0.5,  6),
    ("Energy bars",     1.0,  7),
    ("Rain jacket",     1.0,  8),
    ("Map & compass",   0.3,  7),
    ("Camera",          1.2,  5),
    ("Extra clothes",   2.0,  4),
    ("Cooking stove",   1.5,  6),
    ("Rope (10m)",      2.5,  5),
    ("Sunscreen",       0.3,  4),
    ("Trekking poles",  1.5,  5),
    ("Power bank",      0.8,  6),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("GA Parameters")
    max_weight    = st.slider("Max weight (kg)", 5.0, 30.0, 15.0, 0.5)
    pop_size      = st.slider("Population size", 10, 100, 30)
    n_generations = st.slider("Generations", 10, 200, 60)
    mutation_rate = st.slider("Mutation rate", 0.01, 0.50, 0.05, 0.01)
    crossover_rate= st.slider("Crossover rate", 0.5, 1.0, 0.8, 0.05)
    tournament_k  = st.slider("Tournament size", 2, 10, 3)
    seed          = st.number_input("Random seed", value=42, step=1)
    run_btn       = st.button("Run GA", type="primary", use_container_width=True)

# ── GA core ───────────────────────────────────────────────────────────────────
def fitness(chrom, weights, values, max_w):
    w = sum(weights[i] for i in range(len(chrom)) if chrom[i])
    v = sum(values[i]  for i in range(len(chrom)) if chrom[i])
    return v if w <= max_w else 0

def tournament(pop, fits, k):
    idx = random.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fits[i])][:]

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1[:]
    cut = random.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:]

def mutate(chrom, rate):
    return [1 - g if random.random() < rate else g for g in chrom]

def run_ga(items, max_w, pop_size, n_gen, mut_rate, cx_rate, tourn_k, seed_val):
    random.seed(seed_val)
    names   = [it[0] for it in items]
    weights = [it[1] for it in items]
    values  = [it[2] for it in items]
    n       = len(items)

    pop  = [[random.randint(0,1) for _ in range(n)] for _ in range(pop_size)]
    fit  = lambda c: fitness(c, weights, values, max_w)

    best_val_log, avg_val_log, diversity_log = [], [], []
    best_chrom_ever, best_val_ever = None, -1

    for _ in range(n_gen):
        fits = [fit(c) for c in pop]
        best_i = max(range(pop_size), key=lambda i: fits[i])
        if fits[best_i] > best_val_ever:
            best_val_ever  = fits[best_i]
            best_chrom_ever = pop[best_i][:]
        best_val_log.append(best_val_ever)
        avg_val_log.append(np.mean([f for f in fits if f > 0]) if any(f>0 for f in fits) else 0)
        diversity_log.append(len(set(tuple(c) for c in pop)) / pop_size)

        next_pop = [pop[best_i][:]]  # elitism
        while len(next_pop) < pop_size:
            p1 = tournament(pop, fits, tourn_k)
            p2 = tournament(pop, fits, tourn_k)
            ch = crossover(p1, p2, cx_rate)
            ch = mutate(ch, mut_rate)
            next_pop.append(ch)
        pop = next_pop

    return best_chrom_ever, best_val_ever, best_val_log, avg_val_log, diversity_log, weights, values, names

# ── Run & display ─────────────────────────────────────────────────────────────
items_data = DEFAULT_ITEMS

if run_btn:
    with st.spinner("Running GA..."):
        best_chrom, best_val, bv_log, av_log, div_log, weights, values, names = run_ga(
            items_data, max_weight, pop_size, n_generations,
            mutation_rate, crossover_rate, tournament_k, int(seed)
        )

    total_w = sum(weights[i] for i in range(len(best_chrom)) if best_chrom[i])
    valid   = total_w <= max_weight

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Value",    str(best_val))
    c2.metric("Total Weight",  f"{total_w:.1f} kg")
    c3.metric("Valid Solution","Yes" if valid else "No — Over limit!")
    c4.metric("Items Packed",  str(sum(best_chrom)))

    # ── Convergence plots ─────────────────────────────────────────────────────
    st.subheader("Convergence")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(bv_log, color="seagreen", linewidth=2)
    axes[0].plot(av_log, color="steelblue", linewidth=1.5, linestyle="--", label="Avg (valid)")
    axes[0].set_title("Best vs Avg Value"); axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Value"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(bv_log)),
                [bv_log[i] - (bv_log[i-1] if i > 0 else 0) for i in range(len(bv_log))],
                color="coral", alpha=0.7)
    axes[1].set_title("Improvement per Generation")
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Delta Value")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(div_log, color="purple", linewidth=2)
    axes[2].set_title("Population Diversity")
    axes[2].set_xlabel("Generation"); axes[2].set_ylabel("Unique chromosomes / pop size")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Best solution ─────────────────────────────────────────────────────────
    st.subheader("Best Packing Solution")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        rows = [{"Item": names[i],
                 "Weight (kg)": weights[i],
                 "Value": values[i],
                 "Packed": "✅" if best_chrom[i] else "❌"}
                for i in range(len(best_chrom))]
        st.dataframe(rows, use_container_width=True)

    with col_right:
        packed_names = [names[i] for i in range(len(best_chrom)) if best_chrom[i]]
        packed_vals  = [values[i] for i in range(len(best_chrom)) if best_chrom[i]]
        if packed_names:
            fig2, ax2 = plt.subplots(figsize=(6, len(packed_names)*0.5 + 1))
            bars = ax2.barh(packed_names, packed_vals, color="seagreen", edgecolor="black")
            ax2.bar_label(bars, padding=3)
            ax2.set_xlabel("Value")
            ax2.set_title(f"Packed items (total value = {best_val})")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    # ── Chromosome heatmap ────────────────────────────────────────────────────
    with st.expander("Chromosome visualisation"):
        fig3, ax3 = plt.subplots(figsize=(12, 1.2))
        ax3.imshow([best_chrom], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax3.set_yticks([]); ax3.set_title("Best chromosome (green=packed, red=left)")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()
else:
    st.info("Set parameters in the sidebar and click **Run GA**.")
    st.markdown("""
    **How the GA works:**
    1. **Population** — random binary chromosomes (1=pack, 0=leave)
    2. **Fitness** — total value if within weight limit, else 0
    3. **Selection** — tournament selection (best of k random candidates)
    4. **Crossover** — single-point crossover combines two parents
    5. **Mutation** — random bit flips maintain diversity
    6. **Elitism** — best solution always carried to next generation

    **Experiment:** try `mutation_rate=0.01` vs `0.30` and watch the diversity plot!
    """)

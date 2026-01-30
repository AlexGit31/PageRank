# 📊 PageRank Algorithm — From Data to Analysis

This project implements the **PageRank algorithm** from real navigation data, following a rigorous approach inspired by Google’s original model.  
It covers the full pipeline: adjacency matrix construction, handling of dangling nodes, personalized PageRank, and an in-depth analysis of convergence and performance.

> 🏆 This project is accompanied by an **academic report graded 20/20**.

---

## 🚀 Project Goals

- Implement the **PageRank** algorithm from raw data
- Build the transition matrix using a **stack-based path reconstruction**
- Study:
  - algorithm **convergence**
  - impact of the damping factor **β**
  - influence of the precision parameter **ε**
  - **computation time**
- Visualize PageRank concentration (Scan & Add / Lorenz curve)

---

## 🧠 Methodology

The project is structured into several key steps:

1. **Parsing & Mapping**
   - Extraction of unique pages
   - Page ↔ index mapping

2. **Adjacency Matrix Construction**
   - Path reconstruction using a **stack (pile) logic**
   - Directed link creation between pages

3. **Normalization**
   - Handling of _dangling nodes_
   - Row-stochastic normalization

4. **PageRank Computation**
   - Iterative formula:
     \[
     q = \beta P q + \frac{1-\beta}{s} (\sum q) v
     \]
   - ℓ₁ normalization
   - Convergence criterion based on ε

5. **Analysis & Visualization**
   - Cumulative distribution (Scan & Add)
   - Convergence vs β
   - Convergence vs ε
   - Computation time vs β

---

## 📈 Generated Visualizations

- 📉 **Scan & Add**: PageRank concentration curve
- 🔁 **Iterations vs β**
- 🎯 **Iterations vs ε (log scale)**
- ⏱ **Computation time and iterations vs β**

---

## 📄 Academic Report

The full report (theoretical background, derivations, results, and interpretations) is available here:

👉 **📘 [Access the report via GitHub Pages](https://github.com/AlexGit31/PangeRank/Report.pdf) **

> The report is hosted using **GitHub Pages** for easy and direct access.

---

## 🛠️ Technologies Used

- **Python**
- `NumPy`
- `SciPy`
- `Matplotlib`
- `.tsv` navigation datasets

---

## ▶️ How to Run

```bash
python pagerank.py
```

## 🤖 About This README

This README.md was written with the assistance of ChatGPT, based on the code and project provided by me.
Also, I made sure to review it.

## 👤 Author

Alexis Evaristo
Academic project — BigData


# AdaptiveMethod: ALNS for CVRP and 2E-VRP

This repository contains the implementation of **Adaptive Large Neighborhood Search (ALNS)** algorithms for solving the **Capacitated Vehicle Routing Problem (CVRP)** and the **Two-Echelon Vehicle Routing Problem (2E-VRP)**. 

These methods are utilized in the research paper:  
> **"A Hybrid Urban Delivery System with Robots"**

The project focuses on adaptive strategy selection mechanisms within the ALNS framework, providing multiple selection modes to optimize the search process for complex routing problems.

---

## 🚀 Key Features

- **Multi-Problem Support**: Implementations for both standard CVRP and more complex 2E-VRP (Two-Echelon).
- **Adaptive Selection Mechanisms**: Supports four distinct ALNS selection strategies:
  - `Normal`: Standard weight-based selection for destroy and repair operators.
  - `Reward`: Enhanced reward-based weight updates.
  - `Pair`: Tracks and updates weights for specific (Destroy, Repair) operator pairs.
  - `Table`: A matrix-based approach to manage operator dependencies.
- **Parallel Execution**: Built-in support for `multiprocessing` to run multiple seeds or instances concurrently, significantly reducing total computation time.
- **Extensive Operator Library**: Includes various removal (Random, Worse, Route, Satellite, Related) and insertion (Greedy, Regret-k, Random) heuristics.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `ALNS_CVRP_multiprocess.py` | ALNS implementation for the Capacitated Vehicle Routing Problem with parallel processing. |
| `ALNS_2EVRP_multiprocess.py` | ALNS implementation for the Two-Echelon Vehicle Routing Problem, supporting multiple adaptive modes. |
| `README.md` | Project documentation and usage guide. |

---

## 🛠️ Algorithms & Operators

### 1. Capacitated Vehicle Routing Problem (CVRP)
The CVRP solver aims to minimize the total distance traveled by a fleet of vehicles starting and ending at a single depot while satisfying customer demands.

*   **Destroy Operators**: Random Removal, Worse Removal.
*   **Repair Operators**: Random Repair, Greedy Repair, Regret Repair.

### 2. Two-Echelon Vehicle Routing Problem (2E-VRP)
The 2E-VRP involves delivering goods from a central depot to satellites (1st echelon) and then from satellites to customers (2nd echelon).

*   **Destroy Operators**:
    - `worst_customer_removal`: Removes customers with high individual costs.
    - `random_customer_removal`: Introduces diversification.
    - `route_removal`: Removes entire routes to restructure the solution.
    - `satellite_removal`: Removes all customers served through a specific satellite.
    - `related_customer_removal`: Removes customers that are geographically close.
*   **Repair Operators**:
    - `basic_greedy_customer_insertion`: Standard greedy insertion.
    - `regret_k_customer_insertion`: Inserts based on the "regret" of not choosing the best position.
    - `random_customer_insertion`: Randomly inserts customers.
    - `build_new_L2_routes`: Generates new second-echelon routes.

---

## 💻 Usage

### Prerequisites
- Python 3.x
- NumPy

### Running CVRP Solver
The CVRP script is configured to run multiple instances (like CMT1, CMT2) in parallel.
```bash
python ALNS_CVRP_multiprocess.py
```

### Running 2E-VRP Solver
The 2E-VRP script allows testing different adaptive modes. You can modify the `alns_mode` in the script to switch between `table`, `normal`, `reward`, or `pair`.
```bash
python ALNS_2EVRP_multiprocess.py
```

---

## 📊 Performance & Logging
The scripts automatically log improvements and final results:
- **JSON Logs**: Detailed epoch-by-epoch improvement logs are saved to JSON files (e.g., `improvement_logs_table.json`).
- **Console Summary**: A comprehensive table is printed at the end of execution, showing the Best Cost, Average Cost, Gap to Optimal, and Runtime for each instance.

---

## 📖 Citation
If you use this code in your research, please cite the following paper:
```bibtex
@article{yu2024hybrid,
  title={A Hybrid Urban Delivery System with Robots},
  author={Yu, Shaohua and others},
  journal={...},
  year={2024}
}
```
*(Note: Please update the BibTeX with the full citation details once available.)*

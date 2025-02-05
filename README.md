# Strategy Evaluation Project

## Introduction
In this strategy evaluation project, the primary goal is to analyze the performance of classification-based learners and manual strategies using technical indicators. 

### Key Experiments:
1. **Experiment P (Strategy Learner vs. Manual Strategy)**:  
   A decision tree method is used to build a classification-based learner (via BagLearner and RTLearner) to generate predictions for trading operations. The hypothesis for Experiment P is that the classification-based learner would outperform both the manual strategy and the benchmark.

2. **Experiment R (Impact Factors Study)**:  
   This experiment evaluates the influence of varying impact factors (transaction costs) on trading performance. The hypothesis for Experiment R assumes that:
   - The number of trades decreases as the impact factor increases.
   - The cumulative portfolio return increases with higher impact factors, as transaction costs reduce trading frequency.

---

## Parameters (In-Sample)

- Symbol (`-sym`): `"JPM"`
- Start Date: `2008-01-01`
- End Date: `2009-12-31`
- Starting Portfolio Value (`-start_val`): `$100,000`

---

## Project Structure

### **1. `testproject.py`**
   - Serves as the entry point for running the experiments.
   - **Execution Command:**
     ```bash
     PYTHONPATH=../:. python testproject.py
     ```
   - **Output:**
     - Generates a chart for Experiment P: *JPM In-Sample: ManualStrategy vs. StrategyLearner*.
     - Generates a chart for Experiment R: *JPM In-Sample: StrategyLearner with Different Impact Factors*.

---

### **2. `experiment1.py`**
   - Conducts Experiment P with in-sample data to compare Manual Strategy and Strategy Learner.
   - This script is called by `testproject.py`.

---

### **3. `experiment2.py`**
   - Conducts Experiment R to evaluate Strategy Learner performance under varying impact factors.
   - This script is called by `testproject.py`.

---

### **4. `StrategyLearner.py`**
   - Implements the Strategy Learner model using **BagLearner** and **RTLearner**.
   - Generates predictions to optimize the portfolio by learning from the data.
   - Used in both `experiment1.py` and `experiment2.py`.

---

### **5. `ManualStrategy.py`**
   - Implements a manual trading strategy based on technical indicators.
   - Generates trading signals and predictions to compare against the Strategy Learner in Experiment P.
   - Used in `experiment1.py`.

---

## How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Step 2: Run the Project

#### Run all experiments:
```bash
python testproject.py
```

#### View individual experiments:

**Experiment P:**
```bash
python experiment1.py
```

**Experiment R:**
```bash
python experiment2.py
```
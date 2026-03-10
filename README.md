# Machine-Learning-Insider-Threat-Detection

Built a **Random Forest** model to detect malicious employee activity using behavioral and access log data (**118k records**). Performed **categorical encoding**, **model training**, and **feature importance analysis** to identify key risk indicators.

This project leverages behavioral and access data to predict malicious employee activity. The goal is to help corporate security teams proactively identify **high-risk employees** by analyzing access patterns, printing activity, file usage, and demographic information.

---

## Techniques
- **Predictive Modeling**
- **Machine Learning**
- **Feature Importance Analysis**
- **Data Preprocessing**

---

## Project Highlights
- Built a **Random Forest model** with **class imbalance handling** to detect rare malicious behavior.
- Achieved **96% accuracy**, **82% recall** for malicious employees, and **0.96 AUC**.
- **Key features influencing predictions**:
  - Off-hours activity  
  - Multi-campus access  
  - File burning  
  - Employee seniority
- Visualized results using **confusion matrix**, **ROC curve**, and **feature importance plots**.

---

## Technologies Used
- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Machine Learning**: Random Forest, train/test split, One-Hot Encoding, class weight balancing
- **Tools**: Jupyter Notebook

---

## Key Insights
- **Behavioral indicators** like off-hours activity, multi-campus access, and file burning are strong predictors of insider threats.
- **Random Forest** provides interpretable **feature importance**, helping security teams prioritize investigations.
- Using `class_weight='balanced'` improves detection of **rare malicious events** without losing overall model performance.
- **Visualization Techniques**:
  - Feature importance plots  
  - ROC curve  
  - Confusion matrix

# 🚀 Dashboard-

## Simple Pandas Dashboard Boilerplate

A minimal, interactive dashboard for visualizing and exploring pandas DataFrames in Jupyter notebooks. Use this as a starter template for your own data projects.

---

<p align="center">
  <img src="images/dashboard_gif.gif" alt="Dashboard Demo" width="700"/>
</p>

---

## ✨ Features

- **2D Scatter Plot:** Select X, Y, and color variables. Optional trendline.
- **3D Scatter Plot:** Select X, Y, Z, and color variables. Optional size encoding.
- **3D Surface Plot:** Pivot data to visualize surfaces with customizable color scales.
- **Parallel Coordinates Plot:** Multidimensional visualization for up to 5 variables.
- **Correlation Heatmap:** Visualize correlations between selected numeric variables.
- **Data Table:** Interactive table with filtering, sorting, and min/max highlighting.
- **Statistics Panel:** Summary statistics for selected variables, with optional grouping.
- **Theme Selector:** Switch between Plotly, ggplot2, seaborn, and more.

---

## ⚡ Quick Start

```bash
# Clone this repository
$ git clone https://github.com/ratrent55/Dashboard-.git

# Install required Python packages
$ pip install pandas plotly ipywidgets numpy
```

---

## 🛠️ Usage Example

```python
from dashboard import Dashboard
import pandas as pd

df = pd.read_excel('testData.xlsx')  # or your own DataFrame
dash = Dashboard(df)
dash.display()
```

---

## 📁 File Structure

- `dashboard.py` — Main dashboard class
- `dashboard_test.ipynb` — Example notebook
- `testData.xlsx` — Sample data
- `DataFrameManager.py` — DataFrame utilities
- `df_manager_config.json` — Config file

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue for bugs or feature requests.

---

## 💡 About

This dashboard is designed as a simple boilerplate to help you get started quickly with pandas data visualization in Jupyter. Customize and extend it for your own needs!

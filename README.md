# Time-Series Forecasting for US Heavy Rail Transit Ridership

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R](https://img.shields.io/badge/R-4.0+-276DC3.svg)](https://www.r-project.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This project implements a comprehensive comparative analysis of **classical statistical models** and **deep learning approaches** for forecasting monthly ridership across 14 major US heavy rail (HR) transit agencies from 2002 to 2023. The analysis evaluates forecast performance across three temporal scenarios: **pre-COVID**, **post-COVID**, and **full-series** predictions.

### Key Features

- **14 Transit Agencies**: Analysis of major US heavy rail systems including Chicago CTA, MBTA Boston, WMATA DC, LA Metro, and more
- **Multi-Model Comparison**: 10 forecasting models (7 classical R-based + 3 deep learning Python-based)
- **Three Temporal Scenarios**: Pre-COVID (2002-2019), Post-COVID (2020-2023), and Full Series (2002-2023)
- **Reproducible Pipeline**: Automated workflows for data processing, model training, evaluation, and visualization
- **IEEE Publication Standards**: All visualizations follow IEEE LaTeX formatting standards

---

## Project Structure

```
time-series_forecasting/
├── data/                                      # Output data and results
│   └── resultados_completos_simple.xlsx      # Python models results (MAPE, MASE, RMSE)
├── CODES_IN_R/                               # R statistical forecasting pipeline
│   ├── preparation.R                         # Main orchestrator script
│   ├── separating function.R                 # Data parsing by agency
│   ├── forecasting function.R                # Full/Post-COVID training
│   ├── preforecast.R                         # Pre-COVID training
│   ├── plotting function.R                   # Individual model plots
│   ├── residuals.R                           # Residual diagnostics
│   ├── seasonal plots.R                      # Seasonal decomposition
│   ├── library list.R                        # Required R packages
│   ├── plots/big plots.R                     # Multi-series overview plots
│   └── accuracy_results_fullForecastDec.xlsx # R models results
├── models_RN/                                # Trained deep learning models
│   ├── full_dense/                           # Dense NN - Full scenario
│   ├── precovid_dense/                       # Dense NN - Pre-COVID
│   ├── postcovid_dense/                      # Dense NN - Post-COVID
│   ├── (similar for _cnn and _lstm)
├── plots/                                    # Generated visualizations
│   ├── full_dense/                           # 14-agency plots per scenario
│   ├── precovid_cnn/
│   ├── postcovid_lstm/
│   └── (9 subdirectories total)
├── data_preparation.ipynb                    # Data exploration and visualization
├── entrenamiento_completo.ipynb              # Deep learning training pipeline
├── .github/copilot-instructions.md           # Detailed project documentation
└── September 2025 Complete Monthly...xlsx    # Source NTD data (not tracked)
```

---

## Data Source

**National Transit Database (NTD)** - Monthly Unlinked Passenger Trips (UPT)
- **File**: `September 2025 Complete Monthly Ridership (with adjustments and estimates)_251103 (1).xlsx`
- **Sheet**: `UPT` (Unlinked Passenger Trips)
- **Period**: January 2002 – December 2023 (264 months)
- **Mode**: Heavy Rail (HR) only
- **Agencies**: 14 US transit agencies (Honolulu and San Juan excluded due to excessive missing data)

### Included Transit Agencies

1. **Chicago Transit Authority** (Chicago, IL)
2. **Los Angeles County Metropolitan Transportation Authority** (Los Angeles, CA)
3. **Massachusetts Bay Transportation Authority** (Boston, MA)
4. **Metropolitan Atlanta Rapid Transit Authority** (Atlanta, GA)
5. **New York City Transit Authority** (New York, NY)
6. **Port Authority Trans-Hudson Corporation** (PATH)
7. **Port Authority Transit Corporation** (PATCO)
8. **San Francisco Bay Area Rapid Transit District** (SF BART)
9. **Southeastern Pennsylvania Transportation Authority** (Philadelphia, PA)
10. **Washington Metropolitan Area Transit Authority** (Washington DC)
11. **County of Miami-Dade** (Miami, FL)
12. **Maryland Transit Administration** (Baltimore, MD)
13. **Staten Island Rapid Transit Operating Authority** (Staten Island, NY)
14. **The Greater Cleveland Regional Transit Authority** (Cleveland, OH)

---

## Methodology

### Three Temporal Scenarios

| Scenario | Training Period | Test Period | Purpose |
|----------|----------------|-------------|---------|
| **Full Series** | Jan 2002 – Dec 2022 (252 months) | Jan 2023 – Dec 2023 (12 months) | Overall model performance |
| **Pre-COVID** | Jan 2002 – Feb 2019 (206 months) | Mar 2019 – Feb 2020 (12 months) | Pre-pandemic patterns |
| **Post-COVID** | Apr 2020 – Dec 2022 (33 months) | Jan 2023 – Dec 2023 (12 months) | Recovery phase forecasting |

**COVID-19 Cutoff Date**: March 2020

### Classical Statistical Models (R)

Implemented using the `forecast` and `forecastHybrid` packages:

1. **ETS** - Exponential Smoothing State Space Model
2. **ARIMA** - AutoRegressive Integrated Moving Average (auto-selected)
3. **STL+ETS** - Seasonal-Trend decomposition with ETS
4. **STL+ARIMA** - Seasonal-Trend decomposition with ARIMA
5. **TBATS** - Trigonometric seasonal, Box-Cox transformation, ARMA errors
6. **NNETAR** - Neural Network AutoRegression (P=12 seasonal lags)
7. **Hybrid (ANST)** - Ensemble of ARIMA, NNETAR, STL, and TBATS

**Key Parameters**:
- Frequency: `freq = 12` (monthly data)
- Horizon: `h = 12` (12-month forecast)
- Cross-validation: `cvHorizon = 12`
- Seed: `1234` (reproducibility)

### Deep Learning Models (Python/TensorFlow)

Implemented using `Keras` with `TimeseriesGenerator`:

#### 1. **Dense Neural Network**
```python
Input (12 timesteps) → Dense(16, relu) → Dense(1)
```

#### 2. **Convolutional Neural Network (CNN)**
```python
Input (12, 1) → Conv1D(64, 3, relu) → MaxPooling1D(2) 
              → Flatten → Dense(32, relu) → Dense(1)
```

#### 3. **Long Short-Term Memory (LSTM)**
```python
Input (12, 1) → LSTM(100, relu) → Dense(1)
```

**Hyperparameters**:
- Input sequence length: `n_input = 12` months
- Batch size: `32`
- Epochs: `50`
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)
- Normalization: MinMaxScaler per agency
- Seed: `42`

### Evaluation Metrics

- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **MASE** (Mean Absolute Scaled Error): Scaled against seasonal naive forecast (m=12)
- **RMSE** (Root Mean Squared Error): Absolute error magnitude

---

## Getting Started

### Prerequisites

#### Python Environment
```bash
python >= 3.8
tensorflow >= 2.x
pandas >= 1.3
numpy >= 1.20
matplotlib >= 3.3
scikit-learn >= 0.24
openpyxl >= 3.0  # For Excel I/O
```

#### R Environment
```r
R >= 4.0
forecast >= 8.15
forecastHybrid >= 5.0
writexl >= 1.4
readxl >= 1.3
lubridate >= 1.7
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/tscore-utc/time-series_forecasting.git
cd time-series_forecasting
```

2. **Install Python dependencies**:
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn openpyxl
```

3. **Install R packages**:
```r
source("CODES_IN_R/library list.R")
# This will install: forecast, forecastHybrid, writexl, readxl, lubridate
```

4. **Download NTD data**: Place the Excel file in the project root:
```
September 2025 Complete Monthly Ridership (with adjustments and estimates)_251103 (1).xlsx
```

---

## Usage

### Option 1: R Classical Models

Run the complete R forecasting pipeline:

```r
setwd("/path/to/time-series_forecasting")
source("CODES_IN_R/preparation.R")
```

This executes:
1. Data separation by agency
2. Model training (ETS, ARIMA, STL, TBATS, NNETAR, Hybrid)
3. Forecast generation (12-month horizon)
4. Accuracy evaluation (MAPE, MASE)
5. Visualization (individual plots, big plots, seasonal decomposition, residuals)
6. Results export to `CODES_IN_R/accuracy_results_fullForecastDec.xlsx`

**For Pre-COVID scenario**:
```r
tsList_pre <- my.preforecast.fun(dtList)
```

**For Post-COVID scenario**:
Manually edit `CODES_IN_R/forecasting function.R` line 31:
```r
train_start_date <- c(2020, 4)  # Change from c(2002, 1)
```

### Option 2: Python Deep Learning Models

1. **Open Jupyter Notebook**:
```bash
jupyter notebook entrenamiento_completo.ipynb
```

2. **Run cells sequentially**:
   - **Cell 1-2**: Load and preprocess NTD data
   - **Cell 30**: Configure scenarios (full/precovid/postcovid)
   - **Cell 31**: Define model architectures
   - **Cell 32**: Training function with MASE/MAPE/RMSE
   - **Cell 33**: Grouped plotting function (14 agencies)
   - **Cell 35**: Train all 126 models (3 models × 3 scenarios × 14 agencies)
   - **Cell 37**: Export results to Excel

**Expected outputs**:
- **9 PNG files** in `plots/` (one per model-scenario combination)
- **1 Excel file** in `data/resultados_completos_simple.xlsx` with 4 sheets:
  - `Todos`: All 126 model results
  - `MAPE_Promedios`: Average MAPE by model×scenario
  - `MASE_Promedios`: Average MASE by model×scenario
  - `RMSE_Promedios`: Average RMSE by model×scenario

### Option 3: Data Exploration

Explore raw data and visualizations:
```bash
jupyter notebook data_preparation.ipynb
```

This notebook includes:
- Data loading and cleaning
- Missing data analysis
- Agency selection and filtering
- Multi-agency visualizations with IEEE formatting
- Pre/Post-COVID splitting demonstrations

---

## Visualizations

### Plot Types Generated

#### R Plots (CODES_IN_R/plots/)
1. **Individual Model Plots**: Forecast vs actual for each model per agency
2. **Big Plots**: Multi-series overviews across all agencies
3. **Seasonal Decomposition**: Trend, seasonal, and residual components
4. **Residual Diagnostics**: ACF, histogram, p-values

#### Python Plots (plots/)
Grouped visualizations (14 agencies per PNG) in 7×2 grid layout:
- **Figure size**: 7×4.5 inches (IEEE column width)
- **Font sizes**: IEEE LaTeX standards (body: 9pt, labels: 10pt, legend: 6.5-7pt)
- **Legend**: Top-right corner, 2 columns, semi-transparent
- **Naming convention**: `all_agencies_{scenario}_{model}.png`

Example output structure:
```
plots/
├── full_dense/all_agencies_full_dense.png
├── full_cnn/all_agencies_full_cnn.png
├── full_lstm/all_agencies_full_lstm.png
├── precovid_dense/all_agencies_precovid_dense.png
├── precovid_cnn/all_agencies_precovid_cnn.png
├── precovid_lstm/all_agencies_precovid_lstm.png
├── postcovid_dense/all_agencies_postcovid_dense.png
├── postcovid_cnn/all_agencies_postcovid_cnn.png
└── postcovid_lstm/all_agencies_postcovid_lstm.png
```

Each plot shows:
- **Black line**: Training data (alpha=0.6)
- **Red solid line**: Real test values
- **Blue dashed line**: Model predictions
- **Gray dashed line**: COVID-19 cutoff (March 2020) for full/precovid scenarios

---

## Key Results

### Model Performance Summary

Results are saved in two locations:

1. **R Models**: `CODES_IN_R/accuracy_results_fullForecastDec.xlsx`
   - 7 models × 14 agencies × 3 scenarios = 294 forecasts
   - Metrics: MAPE (test), MASE (test)

2. **Python Models**: `data/resultados_completos_simple.xlsx`
   - 3 models × 14 agencies × 3 scenarios = 126 forecasts
   - Metrics: MAPE, MASE, RMSE
   - Pivot tables for average performance by model-scenario combination

### Scenario Insights

- **Pre-COVID**: Models trained on stable historical patterns (2002-2019)
- **Post-COVID**: Challenging scenario with limited training data (33 months) during recovery phase
- **Full Series**: Includes COVID disruption in training, tests long-term recovery forecast

---

## Technical Details

### Data Preprocessing Pipeline

1. **Column Filtering**: Remove months after December 2023
2. **Mode Filtering**: Select only Heavy Rail (HR) agencies
3. **Missing Data Handling**: Exclude Honolulu and San Juan (excessive missing values)
4. **Date Parsing**: Convert `"MM/YYYY"` strings to datetime objects
5. **Agency Separation**: Create individual DataFrames/time series per agency

### Normalization Strategy

- **R Models**: Work directly with raw ridership values (time series objects)
- **Python Models**: MinMaxScaler (0-1) applied per agency independently
  ```python
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)
  predictions = scaler.inverse_transform(scaled_predictions)
  ```

### COVID-19 Handling

March 2020 represents the sharpest decline in US transit ridership history. Models are evaluated on their ability to:
- **Pre-COVID**: Forecast normal patterns before disruption
- **Post-COVID**: Forecast recovery with limited post-pandemic training data
- **Full Series**: Handle structural breaks and regime changes

---

## File Descriptions

### Notebooks

| File | Description |
|------|-------------|
| `data_preparation.ipynb` | Data loading, exploration, cleaning, and visualization. Includes IEEE-formatted plots for 13 agencies. |
| `entrenamiento_completo.ipynb` | Complete deep learning training pipeline. Trains 126 models (Dense, CNN, LSTM × 3 scenarios × 14 agencies). |

### R Scripts

| File | Purpose |
|------|---------|
| `preparation.R` | Main orchestrator - calls all functions in sequence |
| `separating function.R` | `my.dt.fun()` - Parses Excel into list of data.tables per agency |
| `forecasting function.R` | `my.forecast.fun()` - Full/Post-COVID training and evaluation |
| `preforecast.R` | `my.preforecast.fun()` - Pre-COVID scenario (train ends Feb 2019) |
| `plotting function.R` | `my.plot.fun()` - Individual model forecast plots |
| `big plots.R` | `my.bigplots.fun()` - Multi-series overview plots |
| `seasonal plots.R` | `my.seasonal.fun()` - Seasonal decomposition visualizations |
| `residuals.R` | `my.res.fun()` - Residual analysis and diagnostics |
| `library list.R` | Auto-installs required R packages |

### Model Directories

| Directory | Contents |
|-----------|----------|
| `models_RN/full_dense/` | 14 trained Dense NN models for full scenario (`.keras` files) |
| `models_RN/precovid_*` | 14 models per architecture for pre-COVID scenario |
| `models_RN/postcovid_*` | 14 models per architecture for post-COVID scenario |

**Note**: Directory structure uses `modle_postcovid` (typo) for Dense post-COVID models.

---

## Reproducibility

### Seeds for Reproducibility

- **Python**: `np.random.seed(42)`, `tf.random.set_seed(42)`
- **R**: `set.seed(1234)`

### Hardware Requirements

- **RAM**: 8 GB minimum (16 GB recommended for full training)
- **Storage**: ~500 MB for models + plots
- **GPU**: Optional (CPU sufficient, training takes ~30-60 minutes)

### Expected Runtime

- **R pipeline**: 2-10 minutes per scenario (depends on CPU)
- **Python pipeline**: 30-60 minutes for all 126 models (CPU), 10-20 minutes (GPU)

---

## References

### Data Source
- National Transit Database (NTD): https://www.transit.dot.gov/ntd

### R Packages
- Hyndman, R.J., & Khandakar, Y. (2008). *Automatic Time Series Forecasting: The forecast Package for R*. Journal of Statistical Software, 27(3).
- Shaub, D. (2020). *Fast and accurate yearly time series forecasting with forecast combinations*. International Journal of Forecasting, 36(1), 116-120.

### Deep Learning
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
- Chollet, F. et al. (2015). *Keras*. https://keras.io

---

## Contributing

This project is part of an academic research study. For questions or collaboration inquiries, please open an issue or contact the repository maintainers.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **National Transit Database (NTD)** for providing comprehensive ridership data
- **R forecast community** for robust time series modeling tools
- **TensorFlow/Keras team** for accessible deep learning frameworks
- All transit agencies for their transparency in reporting ridership data

---

## Contact

**Repository**: [tscore-utc/time-series_forecasting](https://github.com/tscore-utc/time-series_forecasting)

**Issues**: https://github.com/tscore-utc/time-series_forecasting/issues

---

*Last Updated: December 2025*

# === 0. Ruta base del proyecto ===
setwd("/home/carlos/IA/FINAL/time-series_forecasting")

# === 1. Cargar librerías ===
source("library list.R")

# === 2. Cargar funciones ===
source("separating function.R")
source("forecasting function.R")
source("preforecast.R")
source("plotting function.R")
source("big plots.R")
source("residuals.R")
source("seasonal plots.R")

# === 3. Leer el Excel ===
myData <- readxl::read_excel(
  "September 2025 Complete Monthly Ridership (with adjustments and estimates)_251103 (1).xlsx",
  sheet = "UPT",
  col_names = FALSE
)

# === 4. Separar series por agencia ===
dtList <- my.dt.fun(myData)

# === 5. Forecast full (2002–2022 train, 2023 test) ===
tsList_full <- my.forecast.fun(dtList)

# === 6. Plots de los modelos ===
my.plot.fun(tsList_full)

# === 7. Big plots de las series ===
my.bigplots.fun(tsList_full)

# === 8. Plots estacionales ===
my.seasonal.fun(tsList_full)

# === 9. Residuos y p-values ===
res_full <- my.res.fun(tsList_full)

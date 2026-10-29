# Dataset Information: Sub_Division_IMD_2017.csv

## Source
- **Origin**: India Meteorological Department (IMD)
- **Time Span**: 1901 to 2017 (117 years of continuous observations)
- **Coverage**: 36 Meteorological Subdivisions across India
- **Observations**: 4,188 rows (subdivision × year)

## Data Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `SUBDIVISION` | String | Name of the meteorological subdivision (e.g., Punjab, Coastal Karnataka) |
| `YEAR` | Integer | Observation year (1901 – 2017) |
| `JAN` – `DEC` | Float | Total rainfall in millimeters (mm) recorded in that month |
| `ANNUAL` | Float | Total annual rainfall in millimeters (mm) |
| `JF` | Float | Winter rainfall (January + February) |
| `MAM` | Float | Pre-monsoon rainfall (March + April + May) |
| `JJAS` | Float | Southwest Monsoon rainfall (June + July + August + September) |
| `OND` | Float | Post-monsoon rainfall (October + November + December) |

## Preprocessing in Pipeline
1. **Pivoting to Long Format**: The script transforms wide-format monthly columns (`JAN` through `DEC`) into chronological subdivision-year-month sequences.
2. **Missing Value Imputation**: Missing values are imputed using historical subdivision-month seasonal averages to preserve local rainfall dynamics.
3. **Feature Engineering**: Generates temporal lags (`LAG_1`, `LAG_2`, `LAG_12`), rolling windows (`ROLL_3`, `ROLL_12`), cyclical trigonometric features (`MONTH_SIN`, `MONTH_COS`), and long-term climate normals (`CLIM_PAST`).

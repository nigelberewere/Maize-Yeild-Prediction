# Dataset Management

## Data Sources for the Model

### ✅ Currently Used

**1. NASA POWER Rainfall Data**
- Source: `Datasets/zwe-rainfall-subnat-full.csv` (if available)
- Content: 46 years of daily rainfall aggregated to annual totals (1981-2026)
- Used by: `src/create_hybrid_dataset.py`
- Size: ~50 KB
- **Status:** Real data, essential for model

**2. World Bank Fertilizer Data**
- Source: `Datasets/agriculture-and-rural-development_zwe_fertiliser.csv` (if available)
- Content: 63 years of fertilizer consumption data for Zimbabwe
- Used by: `src/integrate_datasets.py` (attempted merge)
- Size: ~100 KB
- **Status:** Real data, useful if integrated properly

**3. Trained Model**
- Source: `models/maize_yield_model.pkl`
- Content: Serialized LinearRegression model
- Used by: `app.py`, `src/predict.py`
- Size: ~5 KB
- **Status:** Generated during training, essential for predictions

### ❌ Deleted (Not Useful)

The following FAOSTAT files were **deleted** to save disk space. These contained global trade and economic data, not Zimbabwe-specific agricultural data needed for our model:

**Trade Data (MASSIVE - 11.7 GB total):**
- `Trade_DetailedTradeMatrix_E_All_Data_(Normalized).csv` - 7.8 GB (global bilateral trade)
- `Trade_CropsLivestock_E_All_Data_(Normalized).csv` - 2.3 GB (global agricultural trade)
- `Trade_Indices_E_All_Data_(Normalized).csv` - 1.5 GB (trade indices)

**Economic Data:**
- `Value_of_Production_E_All_Data_(Normalized).csv` - 562 MB (global production values)
- `Value_shares_industry_primary_factors_E_All_Data_(Normalized).csv` - 32 MB (industry shares)

**Reason for Deletion:** 
These files contain **global data for all countries**, not Zimbabwe-specific information. Filtering for Zimbabwe records still left 7000+ rows of mixed indicators with mostly NaN values in yield columns. Our hybrid approach (real rainfall + synthetic agricultural variables) proved more practical.

### ⚠️ Kept (Reference Metadata)

The following small metadata files remain in `Datasets/FAOSTAT_T-Z_E/`:
- `World_Census_Agriculture_E_All_Data_(Normalized).csv` (4.7 MB) - **Potentially useful for area harvested & production**
- Various *_AreaCodes, *_Elements, *_Flags, *_ItemCodes files (~1 MB total) - **Reference lookups**

These are kept for potential future extraction of Zimbabwe-specific agriculture census data.

## How to Extend the Model

### If you want to use real maize yield data:

1. **Extract Zimbabwe-specific data** from the remaining FAOSTAT files:
   ```python
   df = pd.read_csv('Datasets/FAOSTAT_T-Z_E/World_Census_Agriculture_E_All_Data_(Normalized).csv')
   zwe_data = df[df['Area'] == 'Zimbabwe']
   # Filter for maize, yield, area harvested, production
   ```

2. **Or request cleaned Zimbabwe data** directly from:
   - ZIMSTAT (Zimbabwe Statistics)
   - Zimbabwe Ministry of Agriculture
   - FAOSTAT with pre-filtered downloads

3. **Merge with rainfall and fertilizer data** using year as key:
   ```python
   merged = rainfall_df.merge(fertilizer_df, on='Year')
   merged = merged.merge(zimbabwe_maize_df, on='Year')
   ```

## Current Model Architecture

```
Input Features:
├── Rainfall (mm)          [Real - NASA POWER]
├── Temperature (°C)       [Estimated - normal distribution]
├── Fertilizer (kg/ha)     [Estimated - trend + noise]
└── Area (hectares)        [Estimated - trend + noise]
         ↓
   [LinearRegression Model]
         ↓
Output: Yield (kg/ha)
```

## Storage Summary

| Category | Size | Status |
|----------|------|--------|
| Actual datasets used | ~150 KB | ✅ Essential |
| Models | ~5 KB | ✅ Essential |
| Deleted (FAOSTAT bloat) | ~12 GB | ❌ Removed |
| Kept (reference) | ~4.7 MB | ⚠️ Optional |
| **Total remaining** | **~4.9 MB** | **Clean** |


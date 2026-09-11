# EP05 Long-Term Precision + Reference Drift Normalization — updated

The existing EP05/normalization/statistical calculations are retained. Changes are automation, mapping, exclusion audit, defaults, and output packaging.

- Global-flag column selector and explicit “treat all as FALSE” override.
- `global_flag=TRUE` rows are excluded before normalization/outlier/statistical calculations.
- Editable analyte mapping table: `Analyte`, `Device column`, `Reference column`, `Include`.
- Supports multiple analyzer models sharing one reference; PLT, PLT 2 (`PLT_2`), and PLT 3 (`PLT_3`) are pre-populated when present and can all use `PLT_ref`.
- More mappings can be added directly in the app without code changes.
- Bootstrap 95% CIs are off by default.
- Automatic outlier handling and automatic Gcrit remain the defaults.
- Parametric/robust branch continues to produce one reported precision outcome per analysis row.
- One downloadable Excel workbook only: `Results`, `Outliers`, `global flag TRUE`, `Analyte mapping`, `Normalization comparison`, `Settings`.

Run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

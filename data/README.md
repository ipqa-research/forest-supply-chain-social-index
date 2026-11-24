# Data

Each scenario folder (`base_case/` and `growing_demand/`) contains the same six files. The two scenarios differ only in the demand table (sheet `Chi` of `parameters.xlsx`).

| File | Content | Source |
|---|---|---|
| `sets.xlsx` | Supply zones J, candidate nodes K, demand sites L, capacity levels C, technologies M (existing Mfix and new Mnew), raw materials R, sawmill residues B, species S, intermediate products I, final products O and periods T | Case study of the article |
| `complex_sets.xlsx` | Valid combinations: integration pairs, inputs and outputs of each technology, location of the kraft mills, routes from nodes to demand sites and technologies allowed at each node | Case study of the article |
| `parameters.xlsx` | Yields, initial forest area, sawmill residue availability, distances, capacities, investment costs, integration discounts, conversion factors, minimum recipe shares, processing costs and demand | Case study of the article |
| `jobs_per_plant.xlsx` | Operating jobs of each technology and node at the reference capacity level c3 | NREL JEDI models, adapted (Section 4.1 of the article) |
| `jedi_employment_curves.xlsx` | Parameters F, V and exponent b of the jobs curves F + V r^b by technology | NREL JEDI models, adapted (Section 4.1 and Appendix C of the article) |
| `social_data.xlsx` | By node: population, economically active population (EAP), open unemployment rate (OUR), multidimensional poverty index (MPI), province, department and coordinates | Population, EAP and OUR: 2022 National Population, Household and Housing Census (INDEC); MPI (2019, by province): Information, Evaluation and Monitoring System for Social Programs |

Technologies: black liquor gasification (`BLG`), black liquor and pyrolysis oil gasification (`BL-PO-G`), forestry residues gasification (`FRG`), catalytic fluidized bed pyrolysis (`CFBP`), hydrodeoxygenation (`HDO`) and the existing kraft pulp technology (`T_Kraft`). Final products: kraft pulp (`Pulpa_kraft`), methanol (`Methanol`) and hydrodeoxygenated pyrolysis oil (`PO-HDO`, HDO-PO in the article). Intermediate products: black liquor (`BL`) and pyrolysis oil (`PO`). Demand sites `f1` to `f3` receive kraft pulp, `l1` to `l21` methanol and `l22` to `l26` HDO-PO.

The fixed parameters that are not in the Excel files (discount rate, economies-of-scale exponent, transport rates and harvesting cost per hectare) are set in `get_fixed_parameters` of `build_instance.py`, and the selling prices in `model.py`.

`parameters.xlsx` and `social_data.xlsx` contain formulas. The scripts read the values stored by the spreadsheet program, so after editing these files they must be saved from Excel or LibreOffice.

## Growing demand scenario

In `growing_demand/parameters.xlsx` the demand for methanol and HDO-PO at each demand site grows linearly from the median of its base-case values at t1 to twice that median at t15; kraft pulp demand is unchanged. The table is produced with

```
python generate_growing_demand.py data/base_case/parameters.xlsx 2.0
```

which writes `parameters_growing_demand.xlsx` next to the input file. The output must be opened and saved in Excel or LibreOffice to recalculate its formulas, and then renamed to `parameters.xlsx` in a scenario folder.

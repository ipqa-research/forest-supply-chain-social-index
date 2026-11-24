import statistics
import sys

import openpyxl

DEFAULT_FINAL_GROWTH_FACTOR = 2.0
PRODUCTS_TO_GROW = {"Methanol", "PO-HDO"}


def period_number(s):
    return int(str(s).replace("t", ""))


def generate(original_path, final_factor):
    wb_read = openpyxl.load_workbook(original_path, data_only=True)
    wb_write = openpyxl.load_workbook(original_path, data_only=False)

    ws_read = wb_read["Chi"]
    ws_write = wb_write["Chi"]

    header = [c.value for c in next(ws_read.iter_rows(min_row=1, max_row=1))]
    value_columns = list(range(3, len(header) + 1))
    max_row = ws_read.max_row

    rows_by_product = {}
    for row in range(2, max_row + 1):
        product = ws_read.cell(row=row, column=1).value
        if product is None:
            continue
        rows_by_product.setdefault(product, []).append(row)

    t_max = max(
        period_number(ws_read.cell(row=r, column=2).value)
        for rows in rows_by_product.values()
        for r in rows
    )

    changes = 0
    for product, rows in rows_by_product.items():
        if product not in PRODUCTS_TO_GROW:
            continue
        for col in value_columns:
            values = [ws_read.cell(row=r, column=col).value for r in rows
                      if ws_read.cell(row=r, column=col).value is not None]
            if not values:
                continue
            base = statistics.median(values)
            for r in rows:
                t = period_number(ws_read.cell(row=r, column=2).value)
                factor = 1 + (final_factor - 1) * (t - 1) / (t_max - 1)
                if ws_write.cell(row=r, column=col).value is not None:
                    ws_write.cell(row=r, column=col).value = base * factor
                    changes += 1

    output_path = original_path.rsplit(".", 1)[0] + "_growing_demand.xlsx"
    wb_write.save(output_path)

    print(f"Saved: {output_path}")
    print(f"Modified cells (Methanol and PO-HDO only): {changes}")
    print(f"Growth factor from t1 to t{t_max}: x{final_factor}")
    print("Open and save the file in Excel or LibreOffice before using it, so that its formulas are recalculated.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_growing_demand.py parameters.xlsx [final_factor]")
        sys.exit(1)
    path = sys.argv[1]
    factor = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FINAL_GROWTH_FACTOR
    generate(path, factor)

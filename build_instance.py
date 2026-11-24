import os

import pandas as pd


def read_excel_column_as_list(folder_path, filename, sheet_name, column_name, dtype=None):
    file_path = os.path.join(folder_path, filename)
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found in '{folder_path}'.")
    except ValueError as e:
        if "No sheet named" in str(e):
            raise ValueError(f"Sheet '{sheet_name}' not found in '{filename}'.")
        raise e

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in sheet '{sheet_name}' of '{filename}'.")

    series = df[column_name].dropna()
    if dtype:
        series = series.astype(dtype)
    return series.tolist()


def read_excel_columns_as_tuples(folder_path, filename, sheet_name, column_names):
    file_path = os.path.join(folder_path, filename)
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found in '{folder_path}'.")
    except ValueError as e:
        if "No sheet named" in str(e):
            raise ValueError(f"Sheet '{sheet_name}' not found in '{filename}'.")
        raise e

    for col in column_names:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in sheet '{sheet_name}' of '{filename}'.")

    df_filtered = df.dropna(subset=column_names)
    return [tuple(row) for row in df_filtered[column_names].values]


def read_excel_to_dict_simple(folder_path, filename, sheet_name):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    df.columns = ['key', 'value']
    df.dropna(subset=['key'], inplace=True)
    df['value'] = df['value'].fillna(0)
    return pd.Series(df['value'].values, index=df['key']).to_dict()


def read_excel_to_dict_multiple_cols(folder_path, filename, sheet_name, key_cols):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.set_index(list(df.columns[:key_cols]))
    return df.stack().dropna().to_dict()


def read_pivot_table_to_dict(folder_path, filename, sheet_name, index_cols):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

    index_col_names = df.columns[:index_cols]
    for col in index_col_names:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            is_time_str = df[col].astype(str).str.lower().str.startswith('t', na=False)
            if is_time_str.any():
                df[col] = pd.to_numeric(df[col].astype(str).str[1:], errors='coerce').fillna(df[col])

    df.set_index(index_col_names.tolist(), inplace=True)

    cleaned_columns = []
    for col in df.columns:
        if isinstance(col, str) and col.lower().startswith('t'):
            try:
                cleaned_columns.append(int(col[1:]))
            except (ValueError, TypeError):
                cleaned_columns.append(col)
        else:
            cleaned_columns.append(col)
    df.columns = cleaned_columns

    df.fillna(0, inplace=True)
    stacked_series = df.stack()
    return stacked_series[stacked_series != 0].to_dict()


def read_excel_long_to_dict(folder_path, filename, sheet_name, index_cols, value_col):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = index_cols + [value_col] + [f'extra_{i}' for i in range(len(df.columns) - len(index_cols) - 1)]
    df = df.set_index(index_cols)
    return df[value_col].to_dict()


def get_fixed_parameters():
    data = {}
    data['scale_factor'] = 0.6
    data['Ro'] = 0.02
    data['Tau1'] = {
        ('Logs', 'Pinus'): 0.09,
        ('Logs', 'Eucalyptus'): 0.09,
        ('Forest_residues', 'Pinus'): 0.09,
        ('Forest_residues', 'Eucalyptus'): 0.09,
    }
    data['Tau2'] = {'Sawmill_residue': 0.09}
    data['Tau3'] = {'PO': 0.09, 'BL': 0}
    data['Tau4'] = {'Methanol': 0.09, 'PO-HDO': 0.09, 'Pulpa_kraft': 0}
    data['Thetacost'] = {'Pinus': 2807, 'Eucalyptus': 4211}
    return data


def build_instance(datafolder):
    path = datafolder
    sets_file = "sets.xlsx"
    complex_file = "complex_sets.xlsx"
    params_file = "parameters.xlsx"

    data = {
        "J": read_excel_column_as_list(path, sets_file, "J", "Set_J"),
        "K": read_excel_column_as_list(path, sets_file, "K", "Set_K"),
        "L": read_excel_column_as_list(path, sets_file, "L", "Set_L"),
        "C": read_excel_column_as_list(path, sets_file, "C", "Set_C"),
        "M": read_excel_column_as_list(path, sets_file, "M", "Set_M"),
        "R": read_excel_column_as_list(path, sets_file, "R", "Set_R"),
        "B": read_excel_column_as_list(path, sets_file, "B", "Set_b"),
        "S": read_excel_column_as_list(path, sets_file, "S", "Set_S"),
        "I": read_excel_column_as_list(path, sets_file, "I", "Set_I"),
        "O": read_excel_column_as_list(path, sets_file, "O", "Set_O"),
        "T": read_excel_column_as_list(path, sets_file, "T", "Set_T"),
        "Mfix": read_excel_column_as_list(path, sets_file, "Mfix", "Set_Mfix"),
        "Mnew": read_excel_column_as_list(path, sets_file, "Mnew", "Set_Mnew"),

        "Mint": read_excel_columns_as_tuples(path, complex_file, "Mint", ["m_new", "m_fix"]),
        "M_RS": read_excel_columns_as_tuples(path, complex_file, "M_RS", ["m", "r", "s"]),
        "M_SR": read_excel_columns_as_tuples(path, complex_file, "M_SR", ["m", "r"]),
        "I_M": read_excel_columns_as_tuples(path, complex_file, "I_M", ["m", "i"]),
        "M_I": read_excel_columns_as_tuples(path, complex_file, "M_I", ["m", "i"]),
        "MO_O": read_excel_columns_as_tuples(path, complex_file, "MO_O", ["m", "o"]),
        "O_L": read_excel_columns_as_tuples(path, complex_file, "O_L", ["o", "l"]),
        "Mfix_K": read_excel_columns_as_tuples(path, complex_file, "Mfix_K", ["m_fix", "k"]),
        "Set_KL": read_excel_columns_as_tuples(path, complex_file, "Set_KL", ["k", "l"]),
        "Set_MK": read_excel_columns_as_tuples(path, complex_file, "Set_MK", ["m", "k"]),

        "Beta1": read_excel_to_dict_simple(path, params_file, "Beta1"),
        "Theta0": read_excel_to_dict_simple(path, params_file, "Theta0"),

        "Gamma1": read_excel_to_dict_multiple_cols(path, params_file, "Gamma1", 1),
        "Epsilo": read_excel_to_dict_multiple_cols(path, params_file, "Epsilo", 1),
        "Delta": read_excel_to_dict_multiple_cols(path, params_file, "Delta", 1),
        "Beta": read_excel_to_dict_multiple_cols(path, params_file, "Beta", 1),
        "Delta_fixed": read_excel_to_dict_multiple_cols(path, params_file, "Delta_fixed", 1),
        "Pi1": read_excel_to_dict_multiple_cols(path, params_file, "Pi1", 1),
        "Pi2": read_excel_to_dict_multiple_cols(path, params_file, "Pi2", 1),
        "Theta3": read_excel_to_dict_multiple_cols(path, params_file, "Theta3", 1),
        "Theta4": read_excel_to_dict_multiple_cols(path, params_file, "Theta4", 1),

        "Ro1": read_excel_long_to_dict(path, params_file, "Ro1", index_cols=['j', 'k'], value_col='distance'),
        "Ro2": read_excel_long_to_dict(path, params_file, "Ro2", index_cols=['k', 'kk'], value_col='distance'),
        "Ro3": read_excel_long_to_dict(path, params_file, "Ro3", index_cols=['k', 'l'], value_col='distance'),

        "Alpha": read_pivot_table_to_dict(path, params_file, "Alpha", 3),
        "Fi": read_pivot_table_to_dict(path, params_file, "Fi", 2),
        "Sigma1": read_pivot_table_to_dict(path, params_file, "Sigma1", 3),
        "Sigma2": read_pivot_table_to_dict(path, params_file, "Sigma2", 2),
        "Sigma3": read_pivot_table_to_dict(path, params_file, "Sigma3", 2),
        "Sigma4": read_pivot_table_to_dict(path, params_file, "Sigma4", 3),
        "Sigma5": read_pivot_table_to_dict(path, params_file, "Sigma5", 2),
        "Eta1_low": read_pivot_table_to_dict(path, params_file, "Eta1_low", 3),
        "Eta2_low": read_pivot_table_to_dict(path, params_file, "Eta2_low", 2),
        "Eta3_low": read_pivot_table_to_dict(path, params_file, "Eta3_low", 2),
        "Theta2": read_pivot_table_to_dict(path, params_file, "Theta2", 2),
        "Chi": read_pivot_table_to_dict(path, params_file, "Chi", 2),
    }
    data.update(get_fixed_parameters())
    return data


if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "base_case")
    instance_data = build_instance(datafolder=folder)
    for key in ("J", "K", "L", "M", "O", "T"):
        print(f"{key}: {instance_data[key]}")

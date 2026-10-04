"""
Build the small files the web app needs, from the thesis data.

Run this once on your own computer, where the raw CSV exports live:

    python build_artifacts.py --data ../Thesis_python/data_csv/

It repeats the pipeline of notebooks/my_final_model_probabilistic.ipynb
(cells 5, 6, 8, 9, 11, 12, 13, 15, 16 and 18) and writes the results to
the folder `artifacts/`. The app only reads `artifacts/`, so the 267 MB
production file never has to leave your computer.

At the end it prints the overall result on the test period, so you can
check it against Table 5.1 of the thesis.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from predictor import (CHECKPOINTS, FEATURES, add_ratio_features,
                       train_models, predict_range)

TRAIN_SHARE = 0.65
ENOUGH_TEST_ORDERS = 5
CURVE_POINTS = 300          # points kept per real order, for the charts


# =====================================================================
# 1. loading and cleaning (notebook cells 5 and 6)
# =====================================================================
def load_subworkings(data_folder):
    subworkings = pd.read_csv(data_folder + 'scada_subworkings.csv', low_memory=False)
    print('production records read:', format(len(subworkings), ','))

    subworkings['sk_start'] = pd.to_datetime(subworkings['sk_start'],
                                             format='ISO8601', errors='coerce')
    subworkings['sk_finish'] = pd.to_datetime(subworkings['sk_finish'],
                                              format='ISO8601', errors='coerce')
    if subworkings['sk_start'].isna().sum() > 0 or subworkings['sk_finish'].isna().sum() > 0:
        raise ValueError('unexpected timestamp format')

    # orphan records: no work order
    subworkings = subworkings[subworkings['wo_code'].notna()]
    subworkings['wo_code'] = subworkings['wo_code'].astype(str).str.strip()
    subworkings['ar_code'] = subworkings['ar_code'].astype(str).str.strip()

    # dangling records: the counter ran with the wall clock
    elapsed_seconds = (subworkings['sk_finish'] - subworkings['sk_start']).dt.total_seconds()
    gap = (subworkings['sk_working_time'] - elapsed_seconds).abs()
    is_dangling = (elapsed_seconds > 3600) & (gap <= 0.05 * elapsed_seconds)
    is_dangling = is_dangling.fillna(False)
    subworkings = subworkings[~is_dangling].reset_index(drop=True)

    # only WORKING records
    subworkings = subworkings[subworkings['sk_type'] == 'WORKING'].reset_index(drop=True)

    # pieces booked with no machine time
    free_pieces = (subworkings['sk_goods'] > 0) & (subworkings['sk_working_time'] == 0)
    subworkings = subworkings[~free_pieces].reset_index(drop=True)

    print('records kept after cleaning:', format(len(subworkings), ','))
    return subworkings


def load_mes(data_folder):
    mes = pd.read_csv(data_folder + 'mes_workorders.csv')
    mes['wo_code'] = mes['wo_code'].astype(str).str.strip()
    mes['ar_code'] = mes['ar_code'].astype(str).str.strip()
    for column in ['wo_scheduling_date', 'wo_delivery_date',
                   'wo_nominal_delivery_date', 'wo_completing_date']:
        mes[column] = pd.to_datetime(mes[column], format='ISO8601', errors='coerce')
    return mes


def load_alarms(data_folder):
    alarms = pd.read_csv(data_folder + 'scada_alarms.csv')
    alarms['al_start'] = pd.to_datetime(alarms['al_start'], format='ISO8601', errors='coerce')
    alarms['al_stop'] = pd.to_datetime(alarms['al_stop'], format='ISO8601', errors='coerce')
    alarms = alarms.dropna(subset=['al_start']).reset_index(drop=True)
    return alarms[['wc_id', 'al_start', 'al_stop']]


def load_machine_names(data_folder):
    workcenters = pd.read_csv(data_folder + 'mes_workcenters.csv')
    names = {}
    for index, row in workcenters.iterrows():
        names[int(row['wc_id'])] = str(row['wc_name'])
    return names


# =====================================================================
# 2. one row per work order (notebook cells 8 and 9)
# =====================================================================
def build_orders(subworkings, mes):
    orders_all = subworkings.groupby('wo_code').agg(
        wc_id=('wc_id', 'first'),
        ar_code=('ar_code', 'first'),
        start=('sk_start', 'min'),
        finish=('sk_finish', 'max'),
        n_subworkings=('sk_id', 'size'),
        goods=('sk_goods', 'sum'),
        wastes=('sk_wastes', 'sum'),
        working_seconds=('sk_working_time', 'sum'),
    ).reset_index()

    orders_all['work_h'] = orders_all['working_seconds'] / 3600.0
    wall_seconds = (orders_all['finish'] - orders_all['start']).dt.total_seconds()
    orders_all['wall_h'] = wall_seconds / 3600.0

    made_pieces = orders_all['goods'] > 0
    worked_some = orders_all['work_h'] > 0.1
    orders_all = orders_all[made_pieces & worked_some]
    orders_all = orders_all.sort_values('start').reset_index(drop=True)

    plan_columns = ['wo_code', 'wo_ordered_quantity',
                    'wo_expected_working_time', 'wo_expected_setup_time',
                    'wo_priority', 'wo_lead_time',
                    'wo_scheduling_date', 'wo_delivery_date', 'wo_status']
    orders = orders_all.merge(mes[plan_columns], on='wo_code', how='left')

    order_has_plan = ((orders['wo_expected_working_time'].fillna(0) > 0) &
                      (orders['wo_ordered_quantity'].fillna(0) > 0))
    order_is_finished = orders['wo_status'] == 'COMPLETED'
    orders = orders[order_has_plan & order_is_finished].reset_index(drop=True)

    ordered_quantity = orders['wo_ordered_quantity'].fillna(0).clip(lower=1)
    orders['plan_cycle_s'] = orders['wo_expected_working_time'] / ordered_quantity
    orders['plan_work_h'] = orders['wo_expected_working_time'] / 3600.0

    print('usable orders:', len(orders_all), '  with a plan and completed:', len(orders))
    return orders_all, orders


# =====================================================================
# 3. features at every checkpoint (notebook cells 11, 12 and 13)
# =====================================================================
def build_dataset(subworkings, orders_all, orders, alarms):
    from predictor import History

    history = History(orders_all, orders, alarms)

    records_per_order = {}
    for order_code, group in subworkings.groupby('wo_code'):
        records_per_order[order_code] = group.sort_values('sk_start')

    feature_rows = []
    for index, order in orders.iterrows():
        order_records = records_per_order[order['wo_code']]
        ordered_qty = order['wo_ordered_quantity']
        if pd.isna(ordered_qty) or ordered_qty <= 0:
            continue

        pieces_after_each_record = order_records['sk_goods'].fillna(0).cumsum().values
        pieces_at_the_end = pieces_after_each_record[-1]

        article_info = history.article(order['ar_code'], order['start'])
        machine_info = history.machine(order['wc_id'], order['start'])
        alarm_info = history.alarms(order['wc_id'], order['start'])

        for checkpoint in CHECKPOINTS:
            pieces_needed = checkpoint * ordered_qty
            if pieces_at_the_end < pieces_needed:
                continue

            n_seen = int((pieces_after_each_record < pieces_needed).sum())
            seen = order_records.iloc[:n_seen]

            if n_seen > 0:
                pieces_done = float(seen['sk_goods'].sum())
                worked_h = float(seen['sk_working_time'].sum()) / 3600.0
                wastes_done = float(seen['sk_wastes'].sum())
                moment = seen['sk_finish'].max()
                observed_cycle_s = (worked_h * 3600.0) / max(pieces_done, 1)
                elapsed_wall_h = (moment - order['start']).total_seconds() / 3600.0
                if elapsed_wall_h > 0:
                    wall_work_frac = worked_h / elapsed_wall_h
                else:
                    wall_work_frac = np.nan
                waste_rate = wastes_done / max(pieces_done, 1)
            else:
                pieces_done = 0.0
                worked_h = 0.0
                moment = order['start']
                observed_cycle_s = np.nan
                elapsed_wall_h = 0.0
                wall_work_frac = np.nan
                waste_rate = np.nan

            # As in the thesis, the pieces still to make are counted from
            # the quantity the order finally produced (see section 4.3).
            remaining_pieces = max(order['goods'] - pieces_done, 0)
            target_h = max(order['work_h'] - worked_h, 0.0)
            mes_remaining_h = remaining_pieces * order['plan_cycle_s'] / 3600.0

            row = {
                'wo_code': order['wo_code'],
                'wc_id': order['wc_id'],
                'ar_code': order['ar_code'],
                'start': order['start'],
                'checkpoint': checkpoint,
                'ordered_qty': ordered_qty,
                'total_goods': order['goods'],
                'total_work_h': order['work_h'],
                'plan_work_h': order['plan_work_h'],
                'plan_cycle_s': order['plan_cycle_s'],
                'plan_setup_h': order['wo_expected_setup_time'] / 3600.0,
                'priority': order['wo_priority'],
                'pieces_done': pieces_done,
                'remaining_pieces': remaining_pieces,
                'pct_of_ordered': pieces_done / ordered_qty,
                'worked_h': worked_h,
                'obs_cycle_s': observed_cycle_s,
                'elapsed_wall_h': elapsed_wall_h,
                'wall_work_frac': wall_work_frac,
                'waste_rate': waste_rate,
                'dow': moment.weekday(),
                'hour': moment.hour,
                'mes_remaining_h': mes_remaining_h,
                'target_h': target_h,
            }
            row.update(article_info)
            row.update(machine_info)
            row.update(alarm_info)
            feature_rows.append(row)

    dataset = pd.DataFrame(feature_rows)
    dataset = add_ratio_features(dataset)
    print('feature rows:', len(dataset), 'from', dataset['wo_code'].nunique(), 'orders')
    return dataset, records_per_order


# =====================================================================
# 4. split by date and balance the machines (notebook cells 15 and 16)
# =====================================================================
def split_and_balance(dataset, orders):
    split_date = orders['start'].quantile(TRAIN_SHARE)
    train = dataset[dataset['start'] < split_date].reset_index(drop=True)
    test_period = dataset[dataset['start'] >= split_date]

    if len(set(train['wo_code']) & set(test_period['wo_code'])) > 0:
        raise ValueError('an order is on both sides of the split')

    test_counts = test_period.groupby('wc_id')['wo_code'].nunique()
    studied = []
    for wc_id, n_orders in test_counts.items():
        if n_orders >= ENOUGH_TEST_ORDERS:
            studied.append(int(wc_id))
    studied.sort()

    train = train[train['wc_id'].isin(studied)].reset_index(drop=True)
    rows_per_machine = train.groupby('wc_id').size()
    target = rows_per_machine.max()
    copies = []
    for wc_id in train['wc_id']:
        copies.append(int(round(target / rows_per_machine[wc_id])))
    train = train.loc[train.index.repeat(copies)].reset_index(drop=True)

    print('split date:', split_date.date(), '  studied machines:', studied)
    print('training rows after balancing:', len(train))
    return train, split_date, studied


# =====================================================================
# 5. the overall result on the test period (notebook cell 39)
# =====================================================================
def overall_result(dataset, models, split_date, studied):
    overall = dataset[(dataset['start'] >= split_date) &
                      (dataset['wc_id'].isin(studied))].copy()
    low, middle, high = predict_range(models, overall)
    overall['ml_pred_h'] = middle
    overall['ml_p10_h'] = low
    overall['ml_p90_h'] = high

    mes_error = (overall['mes_remaining_h'] - overall['target_h']).abs()
    ml_error = (overall['ml_pred_h'] - overall['target_h']).abs()
    denominator = overall['target_h'].clip(lower=0.5)
    mes_percent = mes_error / denominator * 100
    ml_percent = ml_error / denominator * 100
    inside = ((overall['target_h'] >= overall['ml_p10_h']) &
              (overall['target_h'] <= overall['ml_p90_h']))

    result = {
        'test_rows': int(len(overall)),
        'test_orders': int(overall['wo_code'].nunique()),
        'mes_error_h': float(mes_error.mean()),
        'ml_error_h': float(ml_error.mean()),
        'mes_error_pct': float(mes_percent.mean()),
        'ml_error_pct': float(ml_percent.mean()),
        'mes_median_pct': float(mes_percent.median()),
        'ml_median_pct': float(ml_percent.median()),
        'mes_bias_h': float((overall['mes_remaining_h'] - overall['target_h']).mean()),
        'ml_bias_h': float((overall['ml_pred_h'] - overall['target_h']).mean()),
        'win_rate': float((ml_error < mes_error).mean() * 100),
        'plan_too_low': float((overall['mes_remaining_h'] < overall['target_h']).mean() * 100),
        'range_coverage': float(inside.mean() * 100),
    }

    print()
    print('OVERALL RESULT on the test period (compare with thesis Table 5.1)')
    print('  rows: %d from %d orders' % (result['test_rows'], result['test_orders']))
    print('  average error (h)      : plan %.2f   model %.2f' % (
        result['mes_error_h'], result['ml_error_h']))
    print('  average percentage err : plan %.1f%%  model %.1f%%' % (
        result['mes_error_pct'], result['ml_error_pct']))
    print('  median percentage err  : plan %.1f%%  model %.1f%%' % (
        result['mes_median_pct'], result['ml_median_pct']))
    print('  average bias (h)       : plan %+.2f  model %+.2f' % (
        result['mes_bias_h'], result['ml_bias_h']))
    print('  rows where model closer: %.1f%%' % result['win_rate'])
    print('  real value inside p10-p90: %.1f%%' % result['range_coverage'])
    return result, overall


def downsample_curve(order_records):
    """The real production curve of one order, with at most CURVE_POINTS points."""
    hours = (order_records['sk_working_time'].cumsum() / 3600.0).values
    pieces = order_records['sk_goods'].fillna(0).cumsum().values
    hours = np.concatenate([[0.0], hours])
    pieces = np.concatenate([[0.0], pieces])
    if len(hours) > CURVE_POINTS:
        keep = np.unique(np.linspace(0, len(hours) - 1, CURVE_POINTS).round().astype(int))
        hours = hours[keep]
        pieces = pieces[keep]
    return hours, pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../Thesis_python/data_csv/',
                        help='folder with the four CSV exports')
    parser.add_argument('--out', default='artifacts',
                        help='where to write the files the app reads')
    arguments = parser.parse_args()

    data_folder = arguments.data
    if not data_folder.endswith('/') and not data_folder.endswith('\\'):
        data_folder = data_folder + '/'
    os.makedirs(arguments.out, exist_ok=True)

    subworkings = load_subworkings(data_folder)
    mes = load_mes(data_folder)
    alarms = load_alarms(data_folder)
    machine_names = load_machine_names(data_folder)

    orders_all, orders = build_orders(subworkings, mes)
    dataset, records_per_order = build_dataset(subworkings, orders_all, orders, alarms)
    train, split_date, studied = split_and_balance(dataset, orders)

    models = train_models(train)
    result, overall = overall_result(dataset, models, split_date, studied)

    # --- what the app needs about each machine -------------------------
    machine_info = {}
    for wc_id, group in dataset.groupby('wc_id'):
        typical = group['wall_work_frac'].median()
        if pd.isna(typical) or typical <= 0:
            typical = 1.0
        machine_info[str(int(wc_id))] = {
            'name': machine_names.get(int(wc_id), 'machine ' + str(int(wc_id))),
            'typical_wall_frac': float(typical),
            'studied': int(wc_id) in studied,
        }

    # --- the real curves of the test orders, for the "real orders" tab --
    curve_rows = []
    for order_code in overall['wo_code'].unique():
        hours, pieces = downsample_curve(records_per_order[order_code])
        for point in range(len(hours)):
            curve_rows.append({'wo_code': order_code,
                               'hours': float(hours[point]),
                               'pieces': float(pieces[point])})
    curves = pd.DataFrame(curve_rows)

    # --- write everything ------------------------------------------------
    history_columns = ['wc_id', 'ar_code', 'start', 'finish', 'goods',
                       'working_seconds', 'work_h']
    orders_all[history_columns].to_parquet(os.path.join(arguments.out, 'orders_all.parquet'))

    planned_columns = history_columns + ['plan_cycle_s', 'wo_ordered_quantity',
                                         'wo_expected_setup_time', 'wo_priority']
    orders[planned_columns].to_parquet(os.path.join(arguments.out, 'orders_planned.parquet'))

    alarms.to_parquet(os.path.join(arguments.out, 'alarms.parquet'))
    train.to_parquet(os.path.join(arguments.out, 'train.parquet'))
    overall.to_parquet(os.path.join(arguments.out, 'test_rows.parquet'))
    curves.to_parquet(os.path.join(arguments.out, 'test_curves.parquet'))

    meta = {
        'split_date': str(split_date.date()),
        'data_first_day': str(orders_all['start'].min().date()),
        'data_last_day': str(orders_all['finish'].max().date()),
        'studied_machines': studied,
        'machines': machine_info,
        'features': FEATURES,
        'checkpoints': CHECKPOINTS,
        'training_rows': int(len(train)),
        'result': result,
    }
    with open(os.path.join(arguments.out, 'meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)

    print()
    print('artifacts written to', os.path.abspath(arguments.out))
    for name in sorted(os.listdir(arguments.out)):
        size_kb = os.path.getsize(os.path.join(arguments.out, name)) / 1024
        print('  %-24s %8.0f KB' % (name, size_kb))


if __name__ == '__main__':
    main()

"""
The prediction logic shared by the build script and the web app.

Everything here follows notebooks/my_final_model_probabilistic.ipynb:
the history features (cell 11), the ratios and ready-made estimates
(cell 13), the three models and their settings (cell 18), and the
imaginary order (cell 35).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CHECKPOINTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FEATURES = [
    'ordered_qty', 'plan_work_h', 'plan_cycle_s', 'plan_setup_h', 'priority',
    'pct_of_ordered', 'pieces_done', 'remaining_pieces', 'worked_h',
    'obs_cycle_s', 'elapsed_wall_h', 'wall_work_frac', 'waste_rate',
    'dow', 'hour',
    'art_n_past', 'art_cycle_s', 'art_med_work_h', 'art_plan_bias',
    'mach_n_past', 'mach_cycle_s', 'mach_med_work_h', 'mach_plan_bias',
    'alarms_7d', 'alarm_h_7d', 'alarms_30d', 'alarm_h_30d',
    'obs_vs_plan', 'obs_vs_art', 'obs_vs_mach', 'plan_vs_art', 'plan_vs_mach',
    'est_from_obs_h', 'est_from_art_h', 'est_from_mach_h',
    'est_plan_x_mach_bias', 'est_plan_x_art_bias',
    'mes_remaining_h',
    'wc_id',
]


# =====================================================================
# history of an article, a machine and its alarms (notebook cell 11)
# =====================================================================
class History:
    """Answers 'what happened before this moment' for any article or machine."""

    def __init__(self, orders_all, orders_planned, alarms):
        self.orders_all = orders_all
        self.orders_planned = orders_planned
        self.alarms_per_machine = {}
        for machine_id, group in alarms.groupby('wc_id'):
            self.alarms_per_machine[machine_id] = group.sort_values('al_start')

    def _real_numbers(self, past, prefix):
        result = {prefix + '_n_past': len(past)}
        if len(past) == 0:
            result[prefix + '_cycle_s'] = np.nan
            result[prefix + '_med_work_h'] = np.nan
        else:
            total_seconds = past['working_seconds'].sum()
            total_pieces = past['goods'].sum()
            result[prefix + '_cycle_s'] = total_seconds / max(total_pieces, 1)
            result[prefix + '_med_work_h'] = past['work_h'].median()
        return result

    def _plan_bias(self, past_planned):
        if len(past_planned) == 0:
            return np.nan
        real_cycle = past_planned['working_seconds'] / past_planned['goods'].clip(lower=1)
        bias = real_cycle / past_planned['plan_cycle_s']
        return bias.median()

    def article(self, article_code, moment):
        past = self.orders_all[(self.orders_all['ar_code'] == article_code) &
                               (self.orders_all['finish'] < moment)]
        result = self._real_numbers(past, 'art')
        past_planned = self.orders_planned[(self.orders_planned['ar_code'] == article_code) &
                                           (self.orders_planned['finish'] < moment)]
        result['art_plan_bias'] = self._plan_bias(past_planned)
        return result

    def machine(self, machine_id, moment):
        past = self.orders_all[(self.orders_all['wc_id'] == machine_id) &
                               (self.orders_all['finish'] < moment)]
        result = self._real_numbers(past, 'mach')
        past_planned = self.orders_planned[(self.orders_planned['wc_id'] == machine_id) &
                                           (self.orders_planned['finish'] < moment)]
        result['mach_plan_bias'] = self._plan_bias(past_planned)
        return result

    def alarms(self, machine_id, moment):
        result = {}
        machine_alarms = self.alarms_per_machine.get(machine_id)
        for window_days in [7, 30]:
            window_start = moment - pd.Timedelta(days=window_days)
            count_name = 'alarms_' + str(window_days) + 'd'
            hours_name = 'alarm_h_' + str(window_days) + 'd'
            if machine_alarms is None:
                result[count_name] = 0
                result[hours_name] = 0.0
                continue
            inside = machine_alarms[(machine_alarms['al_start'] >= window_start) &
                                    (machine_alarms['al_start'] < moment)]
            durations = (inside['al_stop'] - inside['al_start']).dt.total_seconds()
            durations = durations.clip(lower=0, upper=8 * 3600)
            result[count_name] = len(inside)
            result[hours_name] = float(durations.sum() / 3600.0)
        return result


# =====================================================================
# ratios and ready-made estimates (notebook cell 13)
# =====================================================================
def add_ratio_features(table):
    table = table.copy()
    safe_plan_cycle = table['plan_cycle_s'].clip(lower=0.001)
    safe_art_cycle = table['art_cycle_s'].clip(lower=0.001)
    safe_mach_cycle = table['mach_cycle_s'].clip(lower=0.001)

    table['obs_vs_plan'] = table['obs_cycle_s'] / safe_plan_cycle
    table['obs_vs_art'] = table['obs_cycle_s'] / safe_art_cycle
    table['obs_vs_mach'] = table['obs_cycle_s'] / safe_mach_cycle
    table['plan_vs_art'] = table['plan_cycle_s'] / safe_art_cycle
    table['plan_vs_mach'] = table['plan_cycle_s'] / safe_mach_cycle

    pieces_left = table['remaining_pieces']
    table['est_from_obs_h'] = pieces_left * table['obs_cycle_s'] / 3600.0
    table['est_from_art_h'] = pieces_left * table['art_cycle_s'] / 3600.0
    table['est_from_mach_h'] = pieces_left * table['mach_cycle_s'] / 3600.0
    table['est_plan_x_mach_bias'] = table['mes_remaining_h'] * table['mach_plan_bias']
    table['est_plan_x_art_bias'] = table['mes_remaining_h'] * table['art_plan_bias']
    return table


# =====================================================================
# the three models (notebook cell 18)
# =====================================================================
def _make_model(loss, quantile=None):
    settings = dict(
        loss=loss,
        max_iter=800,
        learning_rate=0.07,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-3,
        random_state=42,
        categorical_features=['wc_id'],
    )
    if quantile is not None:
        settings['quantile'] = quantile
    return HistGradientBoostingRegressor(**settings)


def train_models(train):
    """Fit the p50, p10 and p90 models on the training rows."""
    train_plan = train['mes_remaining_h'].clip(lower=0.05)
    train_real = train['target_h'].clip(lower=0.05)
    X_train = train[FEATURES]
    y_train = np.log(train_real / train_plan)

    models = {
        'p50': _make_model('absolute_error'),
        'p10': _make_model('quantile', 0.10),
        'p90': _make_model('quantile', 0.90),
    }
    for name in models:
        models[name].fit(X_train, y_train)
    return models


def predict_range(models, table):
    """Working hours still to come for every row: p10, p50 and p90.

    The three models are trained separately, so an edge can land on the
    wrong side of the middle; that edge is then moved onto the middle.
    """
    plan_hours = table['mes_remaining_h'].clip(lower=0.05).values
    X = table[FEATURES]
    middle = plan_hours * np.exp(models['p50'].predict(X))
    low = plan_hours * np.exp(models['p10'].predict(X))
    high = plan_hours * np.exp(models['p90'].predict(X))
    low = np.minimum(low, middle)
    high = np.maximum(high, middle)
    return low, middle, high


# =====================================================================
# one row for an order described by the user (notebook cell 35)
# =====================================================================
def build_live_row(history, wc_id, ar_code, ordered_qty, plan_cycle_s,
                   plan_setup_h, priority, start, pieces_done, worked_h,
                   elapsed_wall_h, wastes_done, typical_wall_frac):
    """The features of an order at the moment it is described.

    Unlike the thesis, the pieces still to make are counted from the
    ORDERED quantity, because the final quantity of a running order is
    not known yet. This is the change section 4.3 of the thesis says a
    real application would need.
    """
    if pieces_done > 0:
        if elapsed_wall_h is None or elapsed_wall_h <= 0:
            elapsed_wall_h = worked_h / typical_wall_frac
        obs_cycle_s = (worked_h * 3600.0) / max(pieces_done, 1)
        if elapsed_wall_h > 0:
            wall_work_frac = worked_h / elapsed_wall_h
        else:
            wall_work_frac = np.nan
        waste_rate = wastes_done / max(pieces_done, 1)
    else:
        # nothing produced yet: the same blanks the training rows have at 0%
        worked_h = 0.0
        elapsed_wall_h = 0.0
        obs_cycle_s = np.nan
        wall_work_frac = np.nan
        waste_rate = np.nan

    moment = start + pd.Timedelta(hours=elapsed_wall_h)
    remaining_pieces = max(ordered_qty - pieces_done, 0)
    mes_remaining_h = remaining_pieces * plan_cycle_s / 3600.0

    row = {
        'wc_id': int(wc_id),
        'ar_code': ar_code,
        'ordered_qty': float(ordered_qty),
        'plan_work_h': ordered_qty * plan_cycle_s / 3600.0,
        'plan_cycle_s': float(plan_cycle_s),
        'plan_setup_h': float(plan_setup_h),
        'priority': float(priority),
        'pct_of_ordered': pieces_done / ordered_qty,
        'pieces_done': float(pieces_done),
        'remaining_pieces': float(remaining_pieces),
        'worked_h': float(worked_h),
        'obs_cycle_s': obs_cycle_s,
        'elapsed_wall_h': float(elapsed_wall_h),
        'wall_work_frac': wall_work_frac,
        'waste_rate': waste_rate,
        'dow': moment.weekday(),
        'hour': moment.hour,
        'mes_remaining_h': mes_remaining_h,
    }
    row.update(history.article(ar_code, start))
    row.update(history.machine(wc_id, start))
    row.update(history.alarms(wc_id, start))
    return add_ratio_features(pd.DataFrame([row]))

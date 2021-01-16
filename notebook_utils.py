from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import gmean


fees = 0.00075


def load_time_bars_df(coin: str, exchange_name: str = "bitmex", bar_size: str = "days", use_hdf=False) -> pd.DataFrame:
    """
    Return minute dataframes for the given coin.
    :param coin: coin to load
    :param group_by: if set, group by this time quantity, e.g. "60T" or "2H"
    :param minute_offset: if not zero, offset the group by this number of minutes
    :param exchange_name: name of the exchange to use as the data source
    :return: the loaded dataframe
    """
    if use_hdf:
        df = pd.read_hdf(f'bitmex_1h_btc_usd.hdf')
    else:
        df = pd.read_csv(f'{bar_size}_{coin}_{exchange_name}.csv')
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["close_date"] = pd.to_datetime(df["close_date"], utc=True)
    df["index"] = df["open_date"]
    df = df.set_index("index")
    df = get_price_changes(df)
    return df


def get_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = df.copy()
    df["pct_change"] = df["close"].pct_change()
    df['price_change'] = df.close / df.close.shift(1)
    return df


def simulate(signal_fn: Callable, df: pd.DataFrame, *args) -> Dict:
    """ Simulate trading the given strategy.
    :param signal_fn: function that will return a dataframe with a "strat_signal" column
    :param df: dataframe with asset prices and other data to trade
    :arg: args to pass to the signal function
    """
    df = df.copy()
    df: pd.DataFrame = signal_fn(df, *args)
    df = add_performance_columns(df)
    return dict(
        df=df,
        args=args,
    )


def add_performance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['strat_pct_change'] = df['strat_signal'].shift(1) * df['pct_change']
    df['strat_returns'] = df['strat_pct_change'] + 1
    df[f'strat_log_returns'] = np.log(df['strat_returns'])
    df[f'strat_fees'] = (1 - fees) ** np.abs(df[f'strat_signal'] - df[f'strat_signal'].shift(1)).sum()
    df[f'strat_is_correct'] = np.where(np.sign(df['strat_signal'].shift(1)) == np.sign(df['pct_change']), 1, 0)
    df[f'strat_is_correct'] = np.where(df['strat_signal'].shift(1) == 0, np.nan, df[f'strat_is_correct'])
    return df


def graph_print_analyze_simulation_result(simulation_result: Dict, analysis_options: Dict) -> Dict:
    graph_strategy(simulation_result["df"], analysis_options)
    analysis_result: Dict = analyze(simulation_result, analysis_options)
    print_analysis(analysis_result)
    return analysis_result


def graph_strategy(dfs, options=None):
    if options is None:
        options = dict()

    if not isinstance(dfs, list):
        dfs = [dfs]

    if "start_date" in options:
        dfs = [df[df.index >= options["start_date"]].copy() for df in dfs]
    if "end_date" in options:
        dfs = [df[df.index < options["end_date"]].copy() for df in dfs]
    extra_columns = options.get("extra_columns", [])
    # Values in extra_columns can be strings, if the we should graph that column on its own graph, or
    # a dict(), if it should be graphed with some other graph
    graphs_count = 4 + len([1 for ec in extra_columns if not isinstance(ec, dict)])

    graph_assistant: GraphAssistant = GraphAssistant(graphs_count, minor_axis_on=options.get('minor_axis_on', False))
    graph_assistant.new_plot(rowspan=1)
    plt.ylabel('Signal')
    for df in dfs:
        plt.plot(df.strat_signal)

    for col in extra_columns:
        if isinstance(col, dict):
            continue
        columns = col
        if isinstance(columns, str):
            columns = [col]
        started = False
        column_names = []
        for c in columns:
            if isinstance(c, str) and c in dfs[0]:
                if not started:
                    started = True
                    graph_assistant.new_plot()
                    plt.ylabel(c)
                plt.plot(dfs[0][c])
                column_names.append(c)
        if len(column_names) > 1:
            plt.legend(tuple(column_names))

    graph_assistant.new_plot()
    plt.ylabel("Coin price")
    if options.get("log_price", False):
        plt.plot(np.log(dfs[0].close))
    else:
        plt.plot(dfs[0].close)
    for col in extra_columns:
        if isinstance(col, dict) and col["graph"] == "price" and col["name"] in dfs[0]:
            plt.plot(dfs[0][col["name"]], col["style"])

    graph_assistant.new_plot()
    plt.ylabel('Log returns')
    if "random_returns_range" in options:
        for trial in options["random_returns_range"]:
            plt.plot(dfs[0][f'random_log_returns{trial}'].cumsum(), color='gray', alpha=0.05)
    for i, df in enumerate(dfs):
        log_rets = df.strat_log_returns.cumsum()
        plt.plot(log_rets, f"C{i}")

    graph_assistant.new_plot()
    plt.ylabel('Returns')
    for i, df in enumerate(dfs):
        cum_rets = df.strat_returns.cumprod()
        plt.plot(cum_rets, f"C{i}")
        cum_rets_after_fees = (cum_rets * (1 - fees) ** np.abs(df[f"strat_signal"] - df[f"strat_signal"].shift(1)).cumsum()).fillna(1)
        plt.plot(cum_rets_after_fees, f"C{i}--", alpha=0.5)

    graph_assistant.show()


def seconds_in_a_bar(df: pd.DataFrame) -> int:
    # Note: not taking the last index, because it could be weirdly truncated
    return (df.index[-2] - df.index[-3]).total_seconds()


seconds_per_hour: int = 3600
seconds_per_day: int = seconds_per_hour * 24
seconds_per_year: int = 365 * seconds_per_day
def compute_sr(seconds_per_bar: int, series, risk_free_rate: float = 0.0, convert_to_hourly: bool = True) -> float:
    # Note: you can use risk_free_rate = .03
    if convert_to_hourly:
        returns = series.groupby(pd.Grouper(freq="H")).prod()
        seconds_per_bar = 60 * 60
    else:
        returns = series
    bars_per_year = seconds_per_year / seconds_per_bar
    avg_per_bar_return = gmean(returns) - 1
    per_bar_risk_free_rate = ((1 + risk_free_rate) ** (1 / bars_per_year)) - 1
    per_bar_std = np.std(returns, ddof=1)
    per_bar_sharpe_ratio = 0 if per_bar_std == 0 else (avg_per_bar_return - per_bar_risk_free_rate) / per_bar_std
    return per_bar_sharpe_ratio * (bars_per_year ** .5)


def analyze(simulation_results_list, options=None) -> Union[List, Dict]:
    if options is None:
        options = dict()
    if not isinstance(simulation_results_list, list):
        simulation_results_list = [simulation_results_list]
    stats_results = []
    for simulation_results in simulation_results_list:
        filtered_df = simulation_results['df']
        if "start_date" not in options:
            options["start_date"] = filtered_df.index.iloc[0]
        if "end_date" not in options:
            options["end_date"] = filtered_df.index.iloc[-1]
        if "random_returns_range" not in options:
            options["random_returns_range"] = []
        filtered_df = filtered_df[(filtered_df.index >= options["start_date"]) & (filtered_df.index < options["end_date"])].copy()
        filtered_df["strat_pct_change"] = filtered_df["pct_change"]

        def compute_statistic0(statistic_name, strat_stat, is_optional=False):
            if options.get("skip_optional_stats", False) and is_optional:
                return None
            return dict(
                statistic_name=statistic_name,
                strat_stat=strat_stat,
                random_stats=[],
                percentile=None,
            )

        def compute_statistic(statistic_name, column_name, compute_stat_fn, is_optional=False):
            if options.get("skip_optional_stats", False) and is_optional:
                return None
            strat_stat = compute_stat_fn(filtered_df[f"strat_{column_name}"])
            random_stats = []
            for trial in options["random_returns_range"]:
                random_stats.append(compute_stat_fn(filtered_df[f'random_{column_name}{trial}']))
            return dict(
                statistic_name=statistic_name,
                strat_stat=strat_stat,
                random_stats=random_stats,
                percentile=stats.percentileofscore(random_stats, strat_stat) if len(random_stats) > 0 else None,
            )

        def compute_statistic2(statistic_name, column_name, column_name2, compute_stat_fn, is_optional=False):
            if options.get("skip_optional_stats", False) and is_optional:
                return None
            strat_stat = compute_stat_fn(filtered_df[f"strat_{column_name}"], filtered_df[f"strat_{column_name2}"])
            random_stats = []
            for trial in options["random_returns_range"]:
                random_stats.append(
                    compute_stat_fn(filtered_df[f'random_{column_name}{trial}'], filtered_df[f'random_{column_name2}{trial}']))
            return dict(
                statistic_name=statistic_name,
                strat_stat=strat_stat,
                random_stats=random_stats,
                percentile=stats.percentileofscore(random_stats, strat_stat) if len(random_stats) > 0 else None,
            )

        def turnover(signal):
            return np.abs(signal - signal.shift(1))

        def returns_after_fees(returns, signal) -> float:
            return returns.prod() * (1 - fees) ** turnover(signal).sum()

        def cumprod_returns_less_fees(returns, signal):
            return (returns.cumprod() * (1 - fees) ** turnover(signal).cumsum()).fillna(1)

        def compute_sr_from_returns(returns) -> float:
            return compute_sr(seconds_in_a_bar(filtered_df), returns)

        def compute_sr_from_returns_less_fees(returns, signal):
            cumprod_returns = cumprod_returns_less_fees(returns, signal)
            per_bar_returns = np.diff(cumprod_returns) / cumprod_returns[:-1] + 1
            return compute_sr_from_returns(per_bar_returns)

        active_rows = filtered_df[filtered_df["strat_signal"] != 0]
        if options.get("ignore_zero_signal", False):
            active_rows = filtered_df[filtered_df["strat_signal"] != 0]

        # avoid division by 0
        num_active_rows: int = len(active_rows) if len(active_rows) > 0 else 1

        stats_list = [
            compute_statistic("Returns", "returns", lambda returns: returns.prod()),
            compute_statistic2("Returns after fees", "returns", "signal", returns_after_fees),
            compute_statistic("SR", "returns", compute_sr_from_returns),
            compute_statistic2("SR (after fees)", "returns", "signal", compute_sr_from_returns_less_fees),
            # NOTE: if the strategy doesn't make a guess (signal=0), we don't count the guess as (in)correct
            compute_statistic("% bars right", "is_correct",
                              lambda is_correct: len(filtered_df[is_correct > 0]) / len(filtered_df[~np.isnan(is_correct)])
                              if len(is_correct[~np.isnan(is_correct)]) > 0 else "No bets",
                              is_optional=True),
            compute_statistic("% bars in market", "signal",
                              lambda signal: len(signal[signal != 0]) / len(signal) if len(signal) > 0 else 1),
            compute_statistic0("Bars count", len(filtered_df)),
        ]
        stats_map = {stat['statistic_name']: stat for stat in stats_list if stat is not None}
        stats_results.append(dict(
            start_date=filtered_df.index.min(),
            end_date=filtered_df.index.max(),
            args=simulation_results["args"],
            stats_map=stats_map,
        ))
    if len(stats_results) == 1:
        return stats_results[0]
    return stats_results


def print_analysis(analyses):
    if not isinstance(analyses, list):
        analyses = [analyses]
    for analysis in analyses:
        print(f"DF filtered {analysis['start_date']} to {analysis['end_date']}")
        args_str = f"#{[arg for arg in analysis['args'] if type(arg) not in [pd.DataFrame, list, dict]]}"
        print((args_str[:300] + '..') if len(args_str) > 302 else args_str)
        for statistic_name, statistic in analysis["stats_map"].items():
            if statistic['percentile'] is None:
                print(f"\t{statistic_name}: {statistic['strat_stat']}")
            else:
                print(f"\t{statistic_name}: {statistic['strat_stat']}, %-ile: {statistic['percentile']}")


class GraphAssistant:
    def __init__(self, max_graphs_count, max_rowspan=5, minor_axis_on=False):
        self.first_ax = None
        self.last_ax = None
        self.graph_row_index = 0
        self.max_graph_rows_count = max_graphs_count * max_rowspan
        self.max_rowspan = max_rowspan
        self.minor_axis_on = minor_axis_on

    def set_percent_yaxis(self, xmax, decimals=1):
        self.last_ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=xmax, decimals=decimals))

    def new_plot(self, rowspan=3, options: Optional[Dict] = None):
        assert rowspan <= self.max_rowspan
        if options is None:
            options = dict()
        if self.first_ax is None:
            self.last_ax = plt.subplot2grid((self.max_graph_rows_count, 1), (self.graph_row_index, 0), rowspan=rowspan)
            self.first_ax = self.last_ax
        else:
            self.last_ax = plt.subplot2grid((self.max_graph_rows_count, 1), (self.graph_row_index, 0), rowspan=rowspan,
                                            sharex=self.first_ax)
        if options.get("is_percent_yaxis", False):
            self.last_ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=options.get("percent_yaxis_xmax", 100),
                                                                          decimals=options.get("percent_yaxis_decimals", 1)))
        plt.tick_params(labelbottom=False)
        plt.grid(True)
        if self.minor_axis_on:
            plt.grid(True, 'minor', 'x')
            plt.minorticks_on()
        self.graph_row_index += rowspan
        return self.last_ax

    def show(self):
        # Finishing touches on the new_plot
        plt.xlabel('Date')
        plt.tick_params(labelbottom=True)
        plt.gcf().set_size_inches(18, self.max_graph_rows_count)
        plt.show()

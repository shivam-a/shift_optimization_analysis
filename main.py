# This is a sample Python script.
import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

initial_shift_size_column = []
configuration_column = []
accepted_shift_size = []
accepted_cost = []
accepted_max_rejection_rate = []
accepted_average_rejection_rate = []


iteration_interval = 50



def make_plots(base_dir_path: str,
               initial_shift_size: str,
               configuration_number: int,
               accepted: str,
               current: str,
               active_shifts: str,
               submitted: str,
               rejection_rate: str,
               estimated_rejection_rate: str):
    accepted_col = ["iteration",
                    "accepted_shift_size",
                    "accepted_submitted",
                    "accepted_rejected",
                    "accepted_average_rejection_rate",
                    "accepted_max_rejection_rate",
                    "accepted_driver_hours",
                    "accepted_cost",
                    "temperature",
                    "random"]

    current_col = ["iteration",
                   "current_shift_size",
                   "current_submitted",
                   "current_rejected",
                   "current_average_rejection_rate",
                   "current_max_rejection_rate",
                   "current_driver_hours",
                   "current_cost",
                   "temperature"]
    pathlib.Path(f"{base_dir_path}/plots/").mkdir(parents=True, exist_ok=True)
    
    accepted_df = pd.read_csv(base_dir_path + accepted)

    current_df = pd.read_csv(base_dir_path + current)

    merged_df = accepted_df.merge(current_df, on='iteration', how="inner")
    merged_df = merged_df.apply(pd.to_numeric)

    active_shifts_df = pd.read_csv(base_dir_path + active_shifts, index_col=0)
    active_shifts_df = active_shifts_df.loc[:, ~active_shifts_df.columns.str.match('Unnamed')]
    supply = active_shifts_df.stack().reset_index(name='supply')

    submitted_df = pd.read_csv(base_dir_path + submitted, index_col=0)
    submitted_df = submitted_df.loc[:, ~submitted_df.columns.str.match('Unnamed')]
    demand = submitted_df.stack().reset_index(name='demand')

    rejected_rate_df = pd.read_csv(base_dir_path + rejection_rate, index_col=0)
    rejected_rate_df = rejected_rate_df.loc[:, ~rejected_rate_df.columns.str.match('Unnamed')]
    rejection_rate = rejected_rate_df.stack().reset_index(name='rejection_rate')

    estimated_rejected_rate_df = pd.read_csv(base_dir_path + estimated_rejection_rate, index_col=0)
    estimated_rejected_rate_df = estimated_rejected_rate_df.loc[:, ~estimated_rejected_rate_df.columns.str.match('Unnamed')]
    estimated_rejection_rate = estimated_rejected_rate_df.stack().reset_index(name='estimated_rejection_rate')

    demand_supply_rejection = pd.concat([demand['demand'], supply['supply'], rejection_rate['rejection_rate']], axis=1)

    initial_shift_size_column.append(initial_shift_size)
    configuration_column.append(configuration_number)
    accepted_cost.append(accepted_df['accepted_cost'].loc[len(accepted_df)-1])
    accepted_shift_size.append(accepted_df['accepted_shift_size'].loc[len(accepted_df)-1])
    accepted_max_rejection_rate.append(accepted_df['accepted_max_rejection_rate'].loc[len(accepted_df)-1])
    accepted_average_rejection_rate.append(accepted_df['accepted_average_rejection_rate'].loc[len(accepted_df)-1])

    bar_line_accepted_rejection_rate(merged_df, base_dir_path)

    bar_line_accepted_cost(merged_df, base_dir_path)

    bar_line_mixed(merged_df, base_dir_path)
    #
    # temperature_variation(merged_df, base_dir_path)
    #
    # plot_active_shifts(base_dir_path, active_shifts_df)
    #
    # plot_submitted_requests(base_dir_path, submitted_df)
    #
    # plot_rejection_rate(base_dir_path, rejected_rate_df)
    #
    # plot_estimate_rejection_rate(base_dir_path, estimated_rejected_rate_df)
    #
    # plot_demand_supply_rejection_scatter(base_dir_path, demand_supply_rejection)

    # scatter_between_variables(base_dir_path, demand_supply_rejection)


def scatter_between_variables(df: DataFrame,  base_dir_path: str, variable_1: str, variable_2: str):
    fig, ax1 = plt.subplots(figsize=(10, 10))
    df.plot(x=variable_1, y=variable_2, kind='scatter',ax=ax1, color='black')
    sns.lmplot(x=variable_1, y=variable_2, data=df, fit_reg=True)
    plt.savefig(f"{base_dir_path}/plots/scatter_{variable_1}_vs_{variable_2}.png")


def temperature_variation(df: DataFrame, base_dir_path: str):
    df[['iteration', 'temperature_x']].plot(x='iteration', y='temperature_x',xlabel='iterations',
        ylabel='temperature')
    plt.legend()
    plt.title('Temperature decreasing function | cooling schedule plot')
    plt.savefig(f"{base_dir_path}/plots/temperature_variations.png")


def bar_line_accepted_cost(df: DataFrame, base_dir_path: str):
    df['iteration'] = df['iteration'].astype(str)
    # df = df[np.arange(len(df)) % 20 == 0]
    df = df.iloc[::10]
    fig, ax1 = plt.subplots(figsize=(50, 10))
    df[['iteration', 'accepted_submitted', 'accepted_rejected', 'accepted_shift_size', 'accepted_shift_size']].plot(
        kind='bar',
        ax=ax1,
        stacked=True,
        xlabel='iterations',
        ylabel='frequency')
    for c in ax1.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
        # remove the labels parameter if it's not needed for customized labels
        ax1.bar_label(c, labels=labels, label_type='center')
    ax2 = plt.twinx(ax=ax1)
    ax3 = plt.twinx(ax=ax2)
    df[['iteration', 'current_cost']].plot(x='iteration', kind='line', ax=ax2, color='yellow', linestyle='--')
    df[['iteration', 'accepted_cost']].plot(x='iteration', kind='line', ax=ax3, color='black', marker='o')
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax3.legend(loc='upper right')
    ax3.set_ylabel('cost of solution')
    ax1.set_title('Bar plot of accepted solution with current and accepted cost')
    plt.savefig(f"{base_dir_path}/plots/bar_line_accepted_cost.png")

def bar_line_accepted_rejection_rate(df: DataFrame, base_dir_path: str):
    df['iteration'] = df['iteration'].astype(str)
    # df = df[np.arange(len(df)) % 20 == 0]
    df = df.iloc[::10]
    fig, ax1 = plt.subplots(figsize=(50, 10))
    df[['iteration', 'accepted_submitted', 'accepted_rejected', 'accepted_shift_size', 'accepted_shift_size']].plot(
        kind='bar',
        ax=ax1,
        stacked=True,
        xlabel='iterations',
        ylabel='frequency')
    for c in ax1.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax1.bar_label(c, labels=labels, label_type='center')
    ax2 = plt.twinx(ax=ax1)
    ax3 = plt.twinx(ax=ax2)
    df[['iteration', 'current_max_rejection_rate']].plot(x='iteration', kind='line', ax=ax2, color='yellow',
                                                         linestyle='--')
    df[['iteration', 'accepted_max_rejection_rate']].plot(x='iteration', kind='line', ax=ax3, color='black',
                                                          marker='o')

    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax3.legend(loc='upper right')
    ax3.set_ylabel('max rejection rate')
    ax1.set_title('Bar plot of accepted solution with current and accepted max rejection rate')
    plt.savefig(f"{base_dir_path}/plots/bar_line_accepted_rejection_rate.png")

def bar_line_mixed(df: DataFrame, base_dir_path: str):
    df['iteration'] = df['iteration'].astype(str)
    # df = df[np.arange(len(df)) % 20 == 0]
    df = df.iloc[::10]
    fig, ax1 = plt.subplots(figsize=(50, 10))
    df[['iteration', 'current_submitted', 'current_rejected', 'current_shift_size']].plot(
        kind='bar',
        ax=ax1,
        stacked=True,
        xlabel='iterations',
        ylabel='frequency',
        align='edge',
        width= 0.2)
    for c in ax1.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
        # remove the labels parameter if it's not needed for customized labels
        ax1.bar_label(c, labels=labels, label_type='center')
    ax2 = plt.twinx(ax=ax1)
    df[['iteration', 'accepted_submitted', 'accepted_rejected', 'accepted_shift_size', 'accepted_shift_size']].plot(
        kind='bar',
        ax=ax2,
        stacked=True,
        xlabel='iterations',
        ylabel='frequency',
        align='edge',
        width= -0.2)
    for c in ax2.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
        # remove the labels parameter if it's not needed for customized labels
        ax2.bar_label(c, labels=labels, label_type='center')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_title('Bar plot of current and accepted solution')
    plt.savefig(f"{base_dir_path}/plots/bar_line_mixed.png")

def plot_active_shifts(base_dir_path: str, active_shifts_df: DataFrame):
    # df = active_shifts_df.iloc[400]
    iteration_interval = len(active_shifts_df) / 2
    df = active_shifts_df.iloc[::iteration_interval]
    fig, ax = plt.subplots(figsize=(10, 10))
    df.T.plot(ax=ax, legend=True)
    ax.set_title('Plot of active shifts per hour')
    ax.set_xlabel('hours')
    ax.set_ylabel('number of active shifts')
    plt.savefig(f"{base_dir_path}/plots/active_shifts_every_100.png")


def plot_submitted_requests(base_dir_path: str, submitted_requests_df: DataFrame):
    iteration_interval = len(submitted_requests_df) / 2
    df = submitted_requests_df.iloc[::iteration_interval]
    fig, ax = plt.subplots(figsize=(10, 10))
    df.T.plot(ax=ax, legend=True)
    ax.set_title('Plot of submitted requests per hour')
    ax.set_xlabel('hours')
    ax.set_ylabel('number of submitted requests')
    plt.savefig(f"{base_dir_path}/plots/submitted_requests_every_100.png")


def plot_rejection_rate(base_dir_path: str, rejected_rate_df: DataFrame):
    iteration_interval = len(rejected_rate_df) / 2
    df = rejected_rate_df.iloc[::iteration_interval]
    fig, ax = plt.subplots(figsize=(10, 10))
    df.T.plot(ax=ax, legend=True)
    ax.set_title('Plot of rejection rate per hour')
    ax.set_xlabel('hours')
    ax.set_ylabel('rejection rate')
    plt.savefig(f"{base_dir_path}/plots/rejection_rate_every_100.png")

def plot_estimate_rejection_rate(base_dir_path: str, estimated_rejected_rate_df: DataFrame):
    df = estimated_rejected_rate_df.iloc[::iteration_interval]
    fig, ax = plt.subplots(figsize=(10, 10))
    df.T.plot(ax=ax, legend=True)
    ax.set_title('Plot of estimated rejection rate per hour')
    ax.set_xlabel('hours')
    ax.set_ylabel('rejection rate')
    plt.savefig(f"{base_dir_path}/plots/estimated_rejection_rate_every_100.png")

def plot_demand_supply_rejection_scatter(base_dir_path: str, merged: DataFrame):
    merged['demand_by_supply'] = merged.apply(lambda row: (row['demand'] / row['supply']) if row['supply'] != 0 else 0, axis=1)
    # print(merged)
    fig, ax = plt.subplots(figsize=(10, 10))
    merged.loc[len(merged)].plot(x='demand_by_supply', y='rejection_rate', kind='scatter', ax=ax, color='black')
    ax.set_title('Scatter plot of demand/supply vs rejection rate')
    ax.set_xlabel('demand(submitted) / supply(shifts)')
    ax.set_ylabel('rejection rate')
    plt.savefig(f"{base_dir_path}/plots/demand_supply_rejection_scatter.png")


if __name__ == '__main__':
    initial_shift_size = ['5_shifts','30_shifts','60_shifts']
    base_directory = "C:/Users/Shivam/IdeaProjects/matsim-libs_moia/test/output/shifts_optimization/200_iterations_only_different_alpha/"
    accepted_csv = "accepted_shift_plan.csv"
    current_csv = "current_shift_plan.csv"
    active_shifts_csv = "active_shifts_per_hour.csv"
    submitted_requests_csv = "submitted_requests_per_hour.csv"
    rejection_rate_csv = "rejected_rate_per_hour.csv"
    configuration_csv = "configuration.csv"
    estimated_rejection_rate_csv = "estimated_rejected_rate_per_hour.csv"
    # configuration_number = 20
    for initial_shift_size in initial_shift_size:
        for configuration_number in range(1, 19):

            make_plots(f"{base_directory}{initial_shift_size}/config{configuration_number}/",
                       initial_shift_size,
                       configuration_number,
                       accepted_csv,
                       current_csv,
                       active_shifts_csv,
                       submitted_requests_csv,
                       rejection_rate_csv,
                       estimated_rejection_rate_csv)
    # df = pd.concat([pd.Series(configuration_column),
    #                 pd.Series(initial_shift_size_column),
    #                 pd.Series(accepted_shift_size),
    #                 pd.Series(accepted_cost),
    #                 pd.Series(accepted_max_rejection_rate),
    #                 pd.Series(accepted_average_rejection_rate)], axis=1).rename(columns={0: 'configuration',
    #                                                                                  1: 'initial_shift_size',
    #                                                                                  2: 'accepted_shift_size',
    #                                                                                  3: 'accepted_cost',
    #                                                                                  4: 'accepted_max_rejection_rate',
    #                                                                                  5: 'accepted_average_rejection_rate'})
    # df.to_csv(base_directory + 'config_results.csv')

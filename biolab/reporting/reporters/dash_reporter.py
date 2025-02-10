"""Dash reporting dashboard for Biolab experiments."""

from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
from dash import dcc
from dash import html
from dash import Input
from dash import Output
from dash import State


def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute win rates for each model based available tasks.

    For each (task, metric) group, the winning model is determined by the best test_mean
    (taking into account whether higher is better). For each group, every model present is
    counted as "possible," and winners get a win. Finally, win_rate = wins / possible.
    """
    group = df.groupby(['task_name', 'metric_name'])
    win_counts = {}
    possible_counts = {}
    # Use "display_name" for uniqueness.
    models = df['display_name'].unique()

    for model in models:
        win_counts[model] = 0
        possible_counts[model] = 0

    for (_, _), group_df in group:
        group_df = group_df.dropna(subset=['test_mean'])
        if group_df.empty:
            continue
        is_higher = group_df['is_higher_better'].iloc[0]
        best_val = (
            group_df['test_mean'].max() if is_higher else group_df['test_mean'].min()
        )
        winners = group_df[group_df['test_mean'] == best_val]['display_name'].tolist()
        for model in group_df['display_name']:
            possible_counts[model] += 1
        for winner in winners:
            win_counts[winner] += 1

    win_data = []
    for model in models:
        wins = win_counts.get(model, 0)
        total = possible_counts.get(model, 0)
        win_rate = wins / total if total > 0 else 0
        win_data.append(
            {'model': model, 'wins': wins, 'possible': total, 'win_rate': win_rate}
        )
    return pd.DataFrame(win_data)


def build_custom_table(custom_rows, selected_models, aggregated_data):  # noqa: PLR0912
    """Build custom table from selected rows and models.

    Build a custom table (as a list of dicts) from the custom rows (each a
    dict with 'task' and 'metric') and the selected models (list of
    display names). Also compute the win rate per model over all custom rows.
    The table has a first column labeled "Task/Metric" and one column per selected model.
    The first row of the table shows the win rate for each model.
    """
    # Initialize a list to hold table rows.
    table_rows = []
    # We'll accumulate wins and possibilities per model across custom rows.
    wins = {m: 0 for m in selected_models}
    poss = {m: 0 for m in selected_models}
    # For each custom row, get the aggregated value from aggregated_data.
    for row in custom_rows:
        task = row['task']
        metric = row['metric']
        # Filter aggregated_data for this (task, metric) among the selected models.
        sub_df = aggregated_data[
            (aggregated_data['task_name'] == task)
            & (aggregated_data['metric_name'] == metric)
            & (aggregated_data['display_name'].isin(selected_models))
        ]
        # For each selected model, extract a value if available.
        row_values = {}
        for m in selected_models:
            m_val = sub_df[sub_df['display_name'] == m]
            if not m_val.empty:
                val = m_val.iloc[0]['test_mean']
                try:
                    val_float = float(val)
                    row_values[m] = f'{val_float:.3f}'
                except Exception:
                    row_values[m] = str(val)
            else:
                row_values[m] = ''
        # Determine the winner(s) for this custom row if possible.
        if not sub_df.empty:
            is_higher = sub_df.iloc[0]['is_higher_better']
            model_vals = {}
            for m in selected_models:
                m_val = sub_df[sub_df['display_name'] == m]
                if not m_val.empty and pd.notna(m_val.iloc[0]['test_mean']):
                    model_vals[m] = float(m_val.iloc[0]['test_mean'])
            if model_vals:
                best = (
                    max(model_vals.values()) if is_higher else min(model_vals.values())
                )
                for m, v in model_vals.items():
                    poss[m] += 1
                    if np.isclose(v, best):
                        wins[m] += 1
        label = f'Task: {task}, Metric: {metric}'
        table_row = {'Task/Metric': label}
        table_row.update(row_values)
        table_rows.append(table_row)
    # Compute win rate row.
    win_rate_row = {'Task/Metric': 'Win Rate'}
    for m in selected_models:
        if poss[m] > 0:
            win_rate_row[m] = f'{wins[m] / poss[m]:.3f}'
        else:
            win_rate_row[m] = ''
    full_table = [win_rate_row, *table_rows]
    return full_table


def serve_dash_report(aggregated_data: pd.DataFrame, *args) -> None:  # noqa: PLR0915
    """Launch a Dash visualization server using the aggregated data.

    Parameters
    ----------
    aggregated_data : pd.DataFrame
        Aggregated data produced by the reporting process.
    """
    # Pre-compute win rates.
    win_df = compute_win_rates(aggregated_data)

    # Extend the metric options to include "Win Rate".
    eval_metrics = sorted(aggregated_data['metric_name'].unique())
    metric_options = [*eval_metrics, 'Win Rate']

    # Prepare task dropdown options.
    task_options = sorted(aggregated_data['task_name'].unique())

    # Precompute mapping between display names and output directories.
    mapping_df = aggregated_data[['display_name', 'output_dir']].drop_duplicates()
    mapping_table = dash_table.DataTable(
        data=mapping_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in mapping_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        page_size=10,
    )

    # For the Custom Table Builder: available models.
    available_models = sorted(aggregated_data['display_name'].unique())

    # Initialize the Dash app.
    app = dash.Dash(__name__)
    app.title = 'Biolab Reporting Dashboard'

    # Layout: use tabs to toggle between By Metric, Custom Table, and Aggregated Data.
    app.layout = html.Div(
        [
            html.H1('Biolab Reporting Dashboard', style={'textAlign': 'center'}),
            # Mapping display section.
            html.Div(
                [
                    html.H3(
                        'Model Mapping (Display Name → Output Directory)',
                        style={'textAlign': 'center'},
                    ),
                    mapping_table,
                ],
                style={'margin': '20px'},
            ),
            dcc.Tabs(
                id='tabs',
                value='by_metric',
                children=[
                    dcc.Tab(
                        label='By Metric',
                        value='by_metric',
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label('Select Metric:'),
                                            dcc.Dropdown(
                                                id='metric-dropdown',
                                                options=[
                                                    {'label': m, 'value': m}
                                                    for m in metric_options
                                                ],
                                                value=metric_options[0]
                                                if metric_options
                                                else None,
                                                clearable=False,
                                            ),
                                        ],
                                        style={
                                            'width': '30%',
                                            'display': 'inline-block',
                                            'padding': '10px',
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label('Select Task:'),
                                            dcc.Dropdown(
                                                id='task-dropdown',
                                                options=[
                                                    {'label': 'All', 'value': 'All'}
                                                ]
                                                + [
                                                    {'label': t, 'value': t}
                                                    for t in task_options
                                                ],
                                                value='All',
                                                clearable=False,
                                            ),
                                        ],
                                        style={
                                            'width': '30%',
                                            'display': 'inline-block',
                                            'padding': '10px',
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label('View Type:'),
                                            dcc.RadioItems(
                                                id='view-type',
                                                options=[
                                                    {
                                                        'label': 'Pivot Table',
                                                        'value': 'table',
                                                    },
                                                    {
                                                        'label': 'Heatmap',
                                                        'value': 'heatmap',
                                                    },
                                                ],
                                                value='table',
                                                labelStyle={
                                                    'display': 'inline-block',
                                                    'margin-right': '10px',
                                                },
                                            ),
                                        ],
                                        style={
                                            'width': '30%',
                                            'display': 'inline-block',
                                            'padding': '10px',
                                        },
                                    ),
                                ]
                            ),
                            html.Div(id='output-div', style={'padding': '20px'}),
                        ],
                    ),
                    dcc.Tab(
                        label='Custom Table',
                        value='custom',
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        'Select Models for Custom Table:',
                                        style={'textAlign': 'center'},
                                    ),
                                    dcc.Dropdown(
                                        id='custom-models',
                                        options=[
                                            {'label': m, 'value': m}
                                            for m in available_models
                                        ],
                                        multi=True,
                                        value=available_models[
                                            :2
                                        ],  # default to two models
                                    ),
                                    html.Hr(),
                                    html.H3(
                                        'Add Custom Row:', style={'textAlign': 'center'}
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label('Select Task:'),
                                                    dcc.Dropdown(
                                                        id='custom-task-dropdown',
                                                        options=[
                                                            {'label': t, 'value': t}
                                                            for t in task_options
                                                        ],
                                                        value=task_options[0]
                                                        if task_options
                                                        else None,
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={
                                                    'width': '45%',
                                                    'display': 'inline-block',
                                                    'padding': '10px',
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Label('Select Metric:'),
                                                    dcc.Dropdown(
                                                        id='custom-metric-dropdown',
                                                        options=[],  # to be updated dynamically
                                                        value=None,
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={
                                                    'width': '45%',
                                                    'display': 'inline-block',
                                                    'padding': '10px',
                                                },
                                            ),
                                        ],
                                        style={'textAlign': 'center'},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                'Add Row',
                                                id='add-row-button',
                                                n_clicks=0,
                                                style={'margin': '10px'},
                                            )
                                        ],
                                        style={'textAlign': 'center'},
                                    ),
                                    dcc.Store(id='custom-rows-store', data=[]),
                                    html.Hr(),
                                    html.H3(
                                        'Custom Table:', style={'textAlign': 'center'}
                                    ),
                                    html.Div(
                                        id='custom-table-div', style={'padding': '20px'}
                                    ),
                                ]
                            )
                        ],
                    ),
                    dcc.Tab(
                        label='Aggregated Data',
                        value='aggregated',
                        children=[
                            html.Div(id='global-table-div', style={'padding': '20px'})
                        ],
                    ),
                ],
            ),
        ]
    )

    # Callback to update custom metric options based on selected task and selected models.
    @app.callback(
        Output('custom-metric-dropdown', 'options'),
        [Input('custom-task-dropdown', 'value'), Input('custom-models', 'value')],
    )
    def update_custom_metric_options(selected_task, selected_models):
        if not selected_task or not selected_models:
            return []
        sub_df = aggregated_data[
            (aggregated_data['task_name'] == selected_task)
            & (aggregated_data['display_name'].isin(selected_models))
        ]
        metrics = sorted(sub_df['metric_name'].unique())
        return [{'label': m, 'value': m} for m in metrics]

    # Callback to add a custom row.
    @app.callback(
        Output('custom-rows-store', 'data'),
        Input('add-row-button', 'n_clicks'),
        State('custom-task-dropdown', 'value'),
        State('custom-metric-dropdown', 'value'),
        State('custom-rows-store', 'data'),
    )
    def add_custom_row(n_clicks, selected_task, selected_metric, current_rows):
        if n_clicks > 0 and selected_task and selected_metric:
            new_row = {'task': selected_task, 'metric': selected_metric}
            if new_row not in current_rows:
                current_rows.append(new_row)
        return current_rows

    # Callback to build the custom table.
    @app.callback(
        Output('custom-table-div', 'children'),
        [Input('custom-rows-store', 'data'), Input('custom-models', 'value')],
    )
    def update_custom_table(custom_rows, selected_models):
        if not selected_models:
            return html.Div('Please select at least one model.')
        if not custom_rows:
            return html.Div('No rows added yet.')
        table_data = build_custom_table(custom_rows, selected_models, aggregated_data)
        columns = [{'name': 'Task/Metric', 'id': 'Task/Metric'}] + [
            {'name': m, 'id': m} for m in selected_models
        ]
        return dash_table.DataTable(
            data=table_data,
            columns=columns,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            page_size=10,
        )

    # Callback for the "By Metric" tab.
    @app.callback(
        Output('output-div', 'children'),
        [
            Input('metric-dropdown', 'value'),
            Input('task-dropdown', 'value'),
            Input('view-type', 'value'),
        ],
    )
    def update_by_metric(selected_metric, selected_task, view_type):
        if selected_metric == 'Win Rate':
            if view_type == 'table':
                return dash_table.DataTable(
                    data=win_df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in win_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                    page_size=10,
                )
            else:
                fig = px.bar(
                    win_df,
                    x='model',
                    y='win_rate',
                    text='win_rate',
                    title='Model Win Rates',
                    labels={'model': 'Model', 'win_rate': 'Win Rate'},
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(yaxis={'range': [0, 1]})
                return dcc.Graph(figure=fig)
        else:
            filtered_df = aggregated_data[
                aggregated_data['metric_name'] == selected_metric
            ]
            if selected_task != 'All':
                filtered_df = filtered_df[filtered_df['task_name'] == selected_task]
            if view_type == 'table':

                def format_mean_std(row):
                    if pd.isna(row['test_mean']):
                        return '–'
                    if (not pd.isna(row.get('test_std', None))) and row[
                        'test_std'
                    ] != 0:
                        return f'{row["test_mean"]:.3f} ± {row["test_std"]:.3f}'
                    else:
                        return f'{row["test_mean"]:.3f}'

                filtered_df = filtered_df.copy()
                filtered_df['mean_std'] = filtered_df.apply(format_mean_std, axis=1)
                pivot_df = filtered_df.pivot(
                    index='task_name', columns='display_name', values='mean_std'
                )
                pivot_df = pivot_df.sort_index().reset_index()
                return dash_table.DataTable(
                    data=pivot_df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in pivot_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                    page_size=10,
                )
            else:
                pivot_df = filtered_df.pivot(
                    index='task_name', columns='display_name', values='test_mean'
                )
                if pivot_df.empty:
                    return html.Div('No data to display.')
                norm_df = pivot_df.copy()
                for idx in norm_df.index:
                    row = norm_df.loc[idx]
                    row_min = row.min()
                    row_max = row.max()
                    if row_max != row_min:
                        norm_df.loc[idx] = (row - row_min) / (row_max - row_min)
                    else:
                        norm_df.loc[idx] = 0.5
                text_matrix = pivot_df.round(3).astype(str).values
                heatmap = go.Heatmap(
                    z=norm_df.values,
                    x=norm_df.columns,
                    y=norm_df.index,
                    zmin=0,
                    zmax=1,
                    colorscale='Viridis',
                    text=text_matrix,
                    texttemplate='%{text}',
                    hovertemplate='Task: %{y}<br>Model: %{x}<br>Test Mean: %{text}<extra></extra>',
                )
                fig = go.Figure(data=[heatmap])
                fig.update_layout(xaxis={'side': 'top'})
                return dcc.Graph(figure=fig)

    # Callback for the "Aggregated Data" tab.
    @app.callback(Output('global-table-div', 'children'), Input('tabs', 'value'))
    def display_aggregated_data(tab_value):
        if tab_value == 'aggregated':
            display_df = aggregated_data.drop(
                columns=['train_scores', 'test_scores'], errors='ignore'
            )
            return dash_table.DataTable(
                data=display_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in display_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                page_size=10,
            )
        return dash.no_update

    app.run_server()

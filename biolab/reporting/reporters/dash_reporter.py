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

from biolab.reporting.aggregator import compute_simplified_win_rates


def build_custom_table(
    custom_rows: list[dict[str, str]],
    selected_models: list[str],
    aggregated_data: pd.DataFrame,
) -> list[dict[str, str]]:
    """Build a custom table from selected rows and models.

    Each row in the output corresponds to a (task, metric) and includes
    the per-model test_mean. We also include a 'Win Rate' row at the top
    that shows how many times each model was the best in the selected
    tasks/metrics.

    Parameters
    ----------
    custom_rows : list of dict
        Each dict has keys 'task' and 'metric'.
    selected_models : list of str
        Models selected by the user, identified by 'display_name'.
    aggregated_data : pd.DataFrame
        Full aggregated data with columns including 'task_name', 'metric_name',
        'display_name', 'test_mean', and 'is_higher_better'.

    Returns
    -------
    list of dict
        Rows for rendering in a Dash DataTable.
    """
    table_rows = []
    wins = {m: 0 for m in selected_models}
    poss = {m: 0 for m in selected_models}

    for row in custom_rows:
        task = row['task']
        metric = row['metric']

        # Filter for this (task, metric) among the selected models
        sub_df = aggregated_data[
            (aggregated_data['task_name'] == task)
            & (aggregated_data['metric_name'] == metric)
            & (aggregated_data['display_name'].isin(selected_models))
        ]
        row_values = {}
        # Collect the test_mean or placeholder
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

        # Determine the winner(s) for this custom row if possible
        if not sub_df.empty:
            is_higher = sub_df.iloc[0]['is_higher_better']
            model_vals = {}
            for m in selected_models:
                row_candidate = sub_df[sub_df['display_name'] == m]
                if not row_candidate.empty and pd.notna(
                    row_candidate.iloc[0]['test_mean']
                ):
                    model_vals[m] = float(row_candidate.iloc[0]['test_mean'])
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

    # Compute win-rate row
    win_rate_row = {'Task/Metric': 'Win Rate'}
    for m in selected_models:
        if poss[m] > 0:
            win_rate_row[m] = f'{wins[m] / poss[m]:.3f}'
        else:
            win_rate_row[m] = ''

    # Put the win-rate row first
    full_table = [win_rate_row, *table_rows]
    return full_table


def serve_dash_report(aggregated_data: pd.DataFrame, *args) -> None:
    """Launch a Dash visualization server using the aggregated data.

    Parameters
    ----------
    aggregated_data : pd.DataFrame
        Aggregated data produced by the reporting process.
    """
    # Pre-compute the simpler "union-only" win rates for each model
    win_df = compute_simplified_win_rates(aggregated_data)

    # Extend the metric options to include "Win Rate"
    eval_metrics = sorted(aggregated_data['metric_name'].unique())
    metric_options = [*eval_metrics, 'Win Rate']

    # Prepare task dropdown options
    task_options = sorted(aggregated_data['task_name'].unique())

    # Precompute mapping between display names and output directories
    mapping_df = aggregated_data[['display_name', 'output_dir']].drop_duplicates()
    mapping_table = dash_table.DataTable(
        data=mapping_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in mapping_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        page_size=10,
    )

    # For the Custom Table Builder: available models
    available_models = sorted(aggregated_data['display_name'].unique())

    # Initialize the Dash app
    app = dash.Dash(__name__)
    app.title = 'Biolab Reporting Dashboard'

    app.layout = html.Div(
        [
            html.H1('Biolab Reporting Dashboard', style={'textAlign': 'center'}),
            # Mapping display section
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
                                        ],  # default to 2 models
                                    ),
                                    html.Hr(),
                                    html.H3(
                                        'Add Custom Row:',
                                        style={'textAlign': 'center'},
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
                                                        options=[],  # updated dynamically
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
                                        'Custom Table:',
                                        style={'textAlign': 'center'},
                                    ),
                                    html.Div(
                                        id='custom-table-div',
                                        style={'padding': '20px'},
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

    # -------------------- Callbacks --------------------

    @app.callback(
        Output('custom-metric-dropdown', 'options'),
        [Input('custom-task-dropdown', 'value'), Input('custom-models', 'value')],
    )
    def update_custom_metric_options(selected_task, selected_models):
        """Dynamically update the Metric dropdown based on selected task and models."""
        if not selected_task or not selected_models:
            return []
        sub_df = aggregated_data[
            (aggregated_data['task_name'] == selected_task)
            & (aggregated_data['display_name'].isin(selected_models))
        ]
        metrics = sorted(sub_df['metric_name'].unique())
        return [{'label': m, 'value': m} for m in metrics]

    @app.callback(
        Output('custom-rows-store', 'data'),
        Input('add-row-button', 'n_clicks'),
        State('custom-task-dropdown', 'value'),
        State('custom-metric-dropdown', 'value'),
        State('custom-rows-store', 'data'),
    )
    def add_custom_row(n_clicks, selected_task, selected_metric, current_rows):
        """Store a new (task, metric) row in local dcc.Store state."""
        if n_clicks > 0 and selected_task and selected_metric:
            new_row = {'task': selected_task, 'metric': selected_metric}
            if new_row not in current_rows:
                current_rows.append(new_row)
        return current_rows

    @app.callback(
        Output('custom-table-div', 'children'),
        [Input('custom-rows-store', 'data'), Input('custom-models', 'value')],
    )
    def update_custom_table(custom_rows, selected_models):
        """Build the custom table DataTable from selected rows and models."""
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

    @app.callback(
        Output('output-div', 'children'),
        [
            Input('metric-dropdown', 'value'),
            Input('task-dropdown', 'value'),
            Input('view-type', 'value'),
        ],
    )
    def update_by_metric(selected_metric, selected_task, view_type):
        """Update the 'By Metric' tab content based on user selections.

        Handle the 'By Metric' tab logic:
          - If metric == 'Win Rate', show bar or table with union-based win rates.
          - Otherwise pivot or heatmap the aggregated_data filtered by selected metric.
        """
        if selected_metric == 'Win Rate':
            # use the pre-computed union-based result "win_df"
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
                    title='Model Win Rates (Union-Only)',
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
                    # If test_std is present and nonzero, show ±
                    if (not pd.isna(row.get('test_std', None))) and row[
                        'test_std'
                    ] != 0:
                        return f'{row["test_mean"]:.3f} ± {row["test_std"]:.3f}'
                    return f'{row["test_mean"]:.3f}'

                filtered_df = filtered_df.copy()
                filtered_df['mean_std'] = filtered_df.apply(format_mean_std, axis=1)
                pivot_df = filtered_df.pivot(
                    index='task_name',
                    columns='display_name',
                    values='mean_std',
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
                # Heatmap
                pivot_df = filtered_df.pivot(
                    index='task_name',
                    columns='display_name',
                    values='test_mean',
                )
                if pivot_df.empty:
                    return html.Div('No data to display.')
                norm_df = pivot_df.copy()

                # Normalize each row from 0..1 to highlight relative performance
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
                    hovertemplate=(
                        'Task: %{y}<br>Model: %{x}<br>Test Mean: %{text}<extra></extra>'
                    ),
                )
                fig = go.Figure(data=[heatmap])
                # Flip x-axis ticks to the top
                fig.update_layout(xaxis={'side': 'top'})
                return dcc.Graph(figure=fig)

    @app.callback(
        Output('global-table-div', 'children'),
        Input('tabs', 'value'),
    )
    def display_aggregated_data(tab_value):
        """Show the full aggregated table in the 'Aggregated Data' tab.

        Drop large columns if present.
        """
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

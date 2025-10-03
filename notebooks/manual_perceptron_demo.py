import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import marimo as mo
    import plotly.express as px
    return go, mo, np, pl, px


@app.cell
def _(mo):
    # Parameters
    debounce_buttons = True
    w1 = mo.ui.slider(
        start=-3, stop=3, step=0.1, value=0, label="w1", show_value=True, debounce=debounce_buttons
    )
    w2 = mo.ui.slider(
        start=-3, stop=3, step=0.1, value=0, label="w2", show_value=True, debounce=debounce_buttons
    )
    b = mo.ui.slider(
        start=-3, stop=3, step=0.1, value=0, label="b", show_value=True, debounce=debounce_buttons
    )

    x1_range = (-4, 4)
    x2_range = (-4, 4)
    z_range = (-5, 5)
    return b, w1, w2, x1_range, x2_range, z_range


@app.cell
def _(pl):
    # Read data from CSV file
    # df = pl.read_csv("data/simple_acc_hr_dataset_v2.csv")
    df = pl.read_csv(
        "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/data/simple_acc_hr_dataset_v2.csv"
    )

    # df # Show dataframe
    return (df,)


@app.cell
def _(df, np, pl):
    # Normalize data
    X_orig = df.select(pl.col(["acceleration (m/s2)", "heart_rate (bpm)"])).to_numpy()
    y = df.select(pl.col("state_int")).to_numpy().flatten()

    X_mean = np.mean(X_orig, axis=0)
    X_std = np.std(X_orig, axis=0)
    X = (X_orig - X_mean) / X_std

    # Create a dataframe with normalized data - easier visualization
    df_norm = pl.DataFrame(
        {
            "acceleration_norm": X[:, 0],
            "heart_rate_norm": X[:, 1],
            "state": df["state"],
            "state_int": df["state_int"],
        }
    )
    # df_norm # Show dataframe
    return X, df_norm, y


@app.cell
def _(df, px):
    px.scatter(
        df,
        x="acceleration (m/s2)",
        y="heart_rate (bpm)",
        color="state",
        title="Original data",
        width=600,
        height=450,
    )
    return


@app.cell
def _(df_norm, px):
    # Compare original and normalized data
    px.scatter(
        df_norm,
        x="acceleration_norm",
        y="heart_rate_norm",
        color="state",
        color_continuous_scale="Bluered_r",
        title="Normalized data",
        width=600,
        height=450,
    )
    return


@app.cell
def _(X, b, np, w1, w2, x1_grid, x2_grid):
    # Use weights to calculate "logit" z for points and for grid
    z = X @ np.array([w1.value, w2.value]) + b.value
    y_pred = z >= 0
    z_grid = x1_grid * w1.value + x2_grid * w2.value + b.value
    return y_pred, z_grid


@app.cell
def _(y, y_pred):
    # Evaluate accuracy
    exercise_correct = (y == 1) & (y_pred == 1)
    rest_correct = (y == 0) & (y_pred == 0)
    exercise_incorrect = (y == 1) & (y_pred == 0)
    rest_incorrect = (y == 0) & (y_pred == 1)
    return exercise_correct, exercise_incorrect, rest_correct, rest_incorrect


@app.cell
def _(np, x1_range, x2_range):
    # Create grid for "background" logit heatmap
    x1_ax = np.linspace(start=x1_range[0], stop=x1_range[1])
    x2_ax = np.linspace(start=x2_range[0], stop=x2_range[1])
    x1_grid, x2_grid = np.meshgrid(x1_ax, x2_ax)
    return x1_ax, x1_grid, x2_ax, x2_grid


@app.cell(hide_code=True)
def _(b, mo, w1, w2):
    mo.vstack(
        [
            mo.md("# 'Manual' perceptron"),
            # mo.image("figures/simple_perceptron.svg", width=400),
            mo.image(
                "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/figures/simple_perceptron.svg",
                width=400,
            ),
            mo.md("$y = g(x_1 w_1 + x_2 w_2 + b)$"),
            w1,
            w2,
            b,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    X,
    b,
    exercise_correct,
    exercise_incorrect,
    go,
    mo,
    np,
    rest_correct,
    rest_incorrect,
    w1,
    w2,
    x1_ax,
    x1_range,
    x2_ax,
    x2_range,
    z_grid,
    z_range,
):
    # Create heatmap for perceptron output (z)
    heatmap = go.Heatmap(
        x=x1_ax,
        y=x2_ax,
        z=z_grid,
        colorscale="RdBu_r",
        zmin=z_range[0],
        zmax=z_range[1],
        opacity=0.4,
        colorbar=dict(title="Perceptron 'logits' (z)", title_side="right"),
    )

    # Create scatter plots for each category
    scatter_exercise_correct = go.Scatter(
        x=X[exercise_correct, 0],
        y=X[exercise_correct, 1],
        mode="markers",
        marker=dict(symbol="circle", color="red"),
        name="exercise correct",
    )
    scatter_rest_correct = go.Scatter(
        x=X[rest_correct, 0],
        y=X[rest_correct, 1],
        mode="markers",
        marker=dict(symbol="circle", color="blue"),
        name="rest correct",
    )
    scatter_exercise_incorrect = go.Scatter(
        x=X[exercise_incorrect, 0],
        y=X[exercise_incorrect, 1],
        mode="markers",
        marker=dict(symbol="x", color="red", size=12),
        name="exercise misclass.",
    )
    scatter_rest_incorrect = go.Scatter(
        x=X[rest_incorrect, 0],
        y=X[rest_incorrect, 1],
        mode="markers",
        marker=dict(symbol="x", color="blue", size=12),
        name="rest misclass.",
    )

    if w1.value == 0 and w2.value == 0:
        decision_boundary = go.Scatter(x=[], y=[])
    elif w2.value == 0:
        decision_boundary = go.Scatter(
            x=np.array(x1_range) * (-w2.value / w1.value) - b.value / w1.value,
            y=x2_range,
            mode="lines",
            name="decision boundary",
        )
    else:
        decision_boundary = go.Scatter(
            x=x1_range,
            y=np.array(x1_range) * (-w1.value / w2.value) - b.value / w2.value,
            mode="lines",
            name="decision boundary",
        )

    # Combine all traces
    fig = go.Figure(
        data=[
            heatmap,
            scatter_exercise_correct,
            scatter_rest_correct,
            scatter_exercise_incorrect,
            scatter_rest_incorrect,
            decision_boundary,
        ]
    )
    fig.update_layout(
        xaxis_title="Normalized acceleration",
        yaxis_title="Normalized heart rate",
        legend_title="Legend",
        legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
        xaxis=dict(range=[x1_range[0], x1_range[1]]),
        yaxis=dict(range=[x2_range[0], x2_range[1]]),
    )
    # fig.show()
    mo.ui.plotly(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

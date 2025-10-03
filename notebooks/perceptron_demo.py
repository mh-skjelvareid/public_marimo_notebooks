import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    import marimo as mo

    return go, make_subplots, mo, np, pl


@app.cell
def _(mo):
    # Parameters
    initial_w1 = mo.ui.slider(
        start=-1, stop=1, step=0.1, value=0.5, show_value=True, label="Initial w1"
    )

    initial_w2 = mo.ui.slider(
        start=-1, stop=1, step=0.1, value=-0.5, show_value=True, label="Initial w2"
    )
    initial_b = mo.ui.slider(
        start=-1, stop=1, step=0.1, value=0, show_value=True, label="Initial b"
    )

    learning_rate = mo.ui.slider(
        steps=[0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        value=0,
        show_value=True,
        label="Learning rate",
    )
    n_epochs = mo.ui.slider(
        steps=[1, 2, 3, 5, 10, 15, 20, 35, 50],
        value=1,
        show_value=True,
        label="Num. epochs",
    )

    x1_range = (-4, 4)
    x2_range = (-4, 4)
    z_range = (-2, 2)
    return (
        initial_b,
        initial_w1,
        initial_w2,
        learning_rate,
        n_epochs,
        x1_range,
        x2_range,
        z_range,
    )


@app.cell
def _(pl):
    # Read data from CSV file
    # df = pl.read_csv("data/simple_acc_hr_dataset_v2.csv")
    df = pl.read_csv(
        "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/data/simple_acc_hr_dataset_v2.csv"
    )

    X_orig = df.select(pl.col(["acceleration (m/s2)", "heart_rate (bpm)"])).to_numpy()
    y = df.select(pl.col("state_int")).to_numpy().flatten()
    return X_orig, y


@app.cell
def _(X_orig, np):
    # Normalize data
    X_mean = np.mean(X_orig, axis=0)
    X_std = np.std(X_orig, axis=0)
    X = (X_orig - X_mean) / X_std
    return (X,)


@app.cell
def _(X, initial_b, initial_w1, initial_w2, learning_rate, n_epochs, np, y):
    # Function to train perceptron - useful for namespace
    def train_perceptron(
        X, y, learning_rate, n_epochs, w1_initial=0, w2_initial=0.5, b_initial=0
    ):
        # Set initial values
        w1 = w1_initial
        w2 = w2_initial
        b = b_initial

        # Lists for logging weights and accuracy
        weights_history = []
        accuracy_history = []

        for _ in range(n_epochs):
            y_pred = []  # List for predicted outputs

            for (x1, x2), correct_label in zip(X, y):
                z = x1 * w1 + x2 * w2 + b  # Logit z
                predicted_label = 1 if z >= 0 else 0  # "Step" activation function

                # Update weights
                update = learning_rate * (correct_label - predicted_label)
                w1 += update * x1
                w2 += update * x2
                b += update

                # Log history
                weights_history.append([w1, w2, b])
                y_pred.append(predicted_label)

            accuracy_history += [np.mean(y_pred == y)] * len(y)  # Epoch accuracy

        return (w1, w2, b), np.array(weights_history), np.array(accuracy_history)

    # Call function
    (w1, w2, b), weights_history, accuracy_history = train_perceptron(
        X,
        y,
        learning_rate.value,
        n_epochs.value,
        w1_initial=initial_w1.value,
        w2_initial=initial_w2.value,
        b_initial=initial_b.value,
    )
    return accuracy_history, b, w1, w2, weights_history


@app.cell
def _(np, x1_range, x2_range):
    # Create grid for x1 and x2
    x1_ax = np.linspace(start=x1_range[0], stop=x1_range[1])
    x2_ax = np.linspace(start=x2_range[0], stop=x2_range[1])
    x1_grid, x2_grid = np.meshgrid(x1_ax, x2_ax)
    return x1_ax, x1_grid, x2_ax, x2_grid


@app.cell
def _(X, b, np, w1, w2, x1_grid, x2_grid):
    # Use weights to calculate "logit" z for points and for grid
    z = X @ np.array([w1, w2]) + b
    y_pred = z >= 0
    z_grid = x1_grid * w1 + x2_grid * w2 + b
    return y_pred, z_grid


@app.cell
def _(y, y_pred):
    # Evaluate accuracy
    exercise_correct = (y == 1) & (y_pred == 1)
    rest_correct = (y == 0) & (y_pred == 0)
    exercise_incorrect = (y == 1) & (y_pred == 0)
    rest_incorrect = (y == 0) & (y_pred == 1)
    return exercise_correct, exercise_incorrect, rest_correct, rest_incorrect


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Perceptron
    The controls below enables us to set initial values for the weights (w0,w1,w2). Note that normally the weights would be given a small random value, e.g. between -1 and 1. We can also define the learning rate (how fast weights are updated) and the number of epochs (how many times to repeat the learning process, over the whole dataset). The first plot below shows both how the values of the weights change for each time step, and how the accuracy of the perceptron changes with time (calculated for each epoch). The lowest plot shows the normalized data, and how the perceptron classifies the data after training. The background color corresponds to the "logits" z, i.e. the input to the activation function.
    """
    )
    return


@app.cell(hide_code=True)
def _(initial_b, initial_w1, initial_w2, learning_rate, mo, n_epochs):
    mo.vstack(
        [
            mo.hstack(
                [
                    # mo.image("figures/simple_perceptron.svg", width=400),
                    mo.image(
                        "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/figures/simple_perceptron.svg",
                        width=400,
                    ),
                    mo.vstack(
                        [
                            mo.md(
                                r"$w_{1,t+1} = w_{1,t} + LR \cdot (y-\hat{y}) \cdot x_1$"
                            ),
                            mo.md(
                                r"$w_{2,t+1} = w_{2,t} + LR \cdot (y-\hat{y}) \cdot x_2$"
                            ),
                            mo.md(r"$b_{1,t+1} = b_{1,t} + LR \cdot (y-\hat{y})$"),
                        ]
                    ),
                ]
            ),
            initial_w1,
            initial_w2,
            initial_b,
            learning_rate,
            n_epochs,
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

    if w1 == 0 and w2 == 0:
        decision_boundary = go.Scatter(x=[], y=[])
    elif w2 == 0:
        decision_boundary = go.Scatter(
            x=np.array(x1_range) * (-w2 / w1) - b / w1,
            y=x2_range,
            mode="lines",
            name="decision boundary",
        )
    else:
        decision_boundary = go.Scatter(
            x=x1_range,
            y=np.array(x1_range) * (-w1 / w2) - b / w2,
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
        title=f"Perceptron, learned weights: {w1=:.2f}, {w2=:.2f}, {b=:.2f}",
        legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
        xaxis=dict(range=[x1_range[0], x1_range[1]]),
        yaxis=dict(range=[x2_range[0], x2_range[1]]),
        height=600,
        width=800,
    )
    # fig.show()
    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _(accuracy_history, go, make_subplots, mo, weights_history):
    def plot_weights_history(weights_history, accuracy_history):
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(y=weights_history[:, 0], mode="lines", name="w1"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(y=weights_history[:, 1], mode="lines", name="w2"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(y=weights_history[:, 2], mode="lines", name="b"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(y=accuracy_history, name="accuracy"),
            secondary_y=True,
        )
        fig.update_layout(
            title=f"Perceptron weights and accuracy during training",
            xaxis_title="Time",
            yaxis_title="Weight value",
            height=400,
            width=800,
        )
        fig.update_yaxes(title_text="Accuracy (per epoch)", secondary_y=True)
        return fig

    history_fig = plot_weights_history(weights_history, accuracy_history)
    mo.ui.plotly(history_fig)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math

    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    import marimo as mo

    return go, make_subplots, mo, np, pl


@app.cell
def _(pl):
    # Read noisy heart rate data
    df = pl.read_csv(
        "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/data/noisy_heart_rate_time_series.csv"
    )
    time = df["t"].to_numpy()
    hr_accurate_orig = df["heart_rate_accurate"].to_numpy()
    hr_noisy_orig = df["heart_rate_noisy"].to_numpy()
    return hr_accurate_orig, hr_noisy_orig


@app.cell
def _(hr_accurate_orig, hr_noisy_orig, np):
    # Normalize data
    hr_accurate_mean = np.mean(hr_accurate_orig)
    hr_accurate_std = np.std(hr_accurate_orig)
    hr_accurate = (hr_accurate_orig - hr_accurate_mean) / hr_accurate_std
    hr_noisy = (hr_noisy_orig - hr_accurate_mean) / hr_accurate_std
    return hr_accurate, hr_accurate_mean, hr_accurate_std, hr_noisy


@app.cell
def _(mo):
    # Define input sliders
    learning_rate = mo.ui.slider(
        steps=[0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
        value=0.01,
        show_value=True,
        label="Learning rate",
    )
    n_epochs = mo.ui.slider(
        steps=[1, 3, 5, 10, 15, 20, 35, 50, 100],
        value=10,
        show_value=True,
        label="Num. epochs",
    )
    return learning_rate, n_epochs


@app.cell
def _(hr_accurate, hr_noisy, learning_rate, n_epochs, np):
    def train_rnn(learning_rate, n_epochs, w1_initial=0.1, w2_initial=0.1, b_initial=0):
        # Initialize weights
        w1 = w1_initial
        w2 = w2_initial
        b = b_initial

        # Train
        weights_history = []
        rms_error_history = []

        # Loop over time series for given number of epochs
        for _ in range(n_epochs):
            y_prev = hr_noisy[0]  # Start state = input sample
            y_pred_vec = []
            # Loop over noisy and accurate samples (x and y)
            for x, y in zip(hr_noisy, hr_accurate):
                # Predict output
                y_pred = w1 * x + w2 * y_prev + b
                y_pred_vec.append(y_pred)

                # Calculate error
                error = y_pred - y

                # Update weights (gradient descent for squared error loss)
                update = -learning_rate * 2 * error  # Neg sign to do gradient descent
                w1 += update * x
                w2 += update * y_prev
                b += update
                weights_history.append([w1, w2, b])

                # Current output becomes previous output in next iteration
                y_prev = y_pred

            # Calculate RMS error across epoch (per-sample is too noisy)
            rms_error_history += [
                np.sqrt(np.mean((y_pred_vec - hr_accurate) ** 2))
            ] * len(hr_noisy)

        return (w1, w2, b), np.array(weights_history), np.array(rms_error_history)

    (w1, w2, b), weights_history, rms_error_history = train_rnn(
        learning_rate.value, n_epochs.value
    )
    return b, rms_error_history, w1, w2, weights_history


@app.cell(hide_code=True)
def _(learning_rate, mo, n_epochs):
    mo.vstack(
        [
            mo.md("# Simple recurrent network for time series "),
            mo.image(
                "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/figures/simple_rnn.svg",
                width=400,
            ),
            learning_rate,
            n_epochs,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    go,
    hr_accurate_std,
    make_subplots,
    mo,
    rms_error_history,
    weights_history,
):
    def plot_weights_history(weights_history, rms_error_history):
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
            go.Scatter(y=rms_error_history * hr_accurate_std, name="RMS error"),
            secondary_y=True,
        )
        fig.update_layout(
            title=f"Weight values for each time step",
            xaxis_title="Time",
            yaxis_title="Weight value",
        )
        fig.update_yaxes(title_text="Mean heart rate error (BPM)", secondary_y=True)
        return fig

    history_fig = plot_weights_history(weights_history, rms_error_history)
    mo.ui.plotly(history_fig)
    return


@app.cell(hide_code=True)
def _(b, hr_accurate, hr_noisy, np, w1, w2):
    def run_trained_rnn(w1, w2, b):
        hr_pred = []
        y_prev = hr_noisy[0]
        for x, y in zip(hr_noisy, hr_accurate):
            y_pred = w1 * x + w2 * y_prev + b
            hr_pred.append(y_pred)
            y_prev = y_pred
        rms_error = np.sqrt(np.mean((hr_pred - hr_accurate) ** 2))

        return hr_pred, rms_error

    hr_pred, rms_error = run_trained_rnn(w1, w2, b)
    return hr_pred, rms_error


@app.cell(hide_code=True)
def _(
    go,
    hr_accurate,
    hr_accurate_mean,
    hr_accurate_std,
    hr_noisy,
    hr_pred,
    mo,
    np,
    rms_error,
):
    # Plot input and output
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=hr_noisy * hr_accurate_std + hr_accurate_mean,
            mode="lines",
            name="Input",
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            y=np.array(hr_pred) * hr_accurate_std + hr_accurate_mean,
            mode="lines",
            name="Predicted",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=hr_accurate * hr_accurate_std + hr_accurate_mean,
            mode="lines",
            name="Accurate",
        )
    )
    fig.update_layout(
        title=f"RNN Prediction vs Input and Accurate, RMS error {rms_error * hr_accurate_std:.2f} bpm",
        xaxis_title="Time",
        yaxis_title="Heart Rate",
    )
    # fig.show()
    mo.ui.plotly(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

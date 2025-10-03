import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    import marimo as mo

    return go, mo, np, pl


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
    w1 = mo.ui.slider(
        start=0, stop=1.1, step=0.05, value=0.2, label="w1", show_value=True
    )
    w2 = mo.ui.slider(
        start=0, stop=1.1, step=0.05, value=0.2, label="w2", show_value=True
    )
    b = mo.ui.slider(start=-1, stop=1, step=0.05, value=0, label="b", show_value=True)
    return b, w1, w2


@app.cell
def _(b, hr_accurate, hr_noisy, np, w1, w2):
    # Run data through recurrent network and evaluate error
    hr_pred = []
    y_prev = hr_noisy[0]
    for x, y in zip(hr_noisy, hr_accurate):
        y_pred = w1.value * x + w2.value * y_prev + b.value
        hr_pred.append(y_pred)
        y_prev = y_pred

    rms_error = np.sqrt(np.mean((hr_pred - hr_accurate) ** 2))
    return hr_pred, rms_error


@app.cell(hide_code=True)
def _(b, mo, w1, w2):
    mo.vstack(
        [
            mo.md("# 'Manual' recurrent neural network"),
            mo.image(
                "https://raw.githubusercontent.com/mh-skjelvareid/inf-1600-intro-ai/main/figures/simple_rnn.svg",
                width=400,
            ),
            w1,
            w2,
            b,
        ]
    )
    return


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
            name="$x$",
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            y=np.array(hr_pred) * hr_accurate_std + hr_accurate_mean,
            mode="lines",
            name="$\hat{y}$",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=hr_accurate * hr_accurate_std + hr_accurate_mean,
            mode="lines",
            name="$y$",
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

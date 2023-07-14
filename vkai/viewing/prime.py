import plotly.express as px


def plot(xs, ys, using: str = "bar", title="Y"):
    if using == "bar":
        fig = px.bar(x=xs, y=ys)
    else:
        fig = px.line(x=xs, y=ys)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), automargin=False, yref="paper"), xaxis={"type": "category"}
    )
    fig.update_layout(yaxis_title=None)
    fig.update_layout(xaxis_title=None)
    fig.update_xaxes(tickfont_size=9, ticks="outside", ticklen=0.5, tickwidth=1)
    return fig


__all__ = ["plot"]

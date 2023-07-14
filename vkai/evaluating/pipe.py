import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl


def evaluate(
    _docs,
    _topic_per_doc,
    _label_per_doc=None,
    _topics_to_show=None,
    _labels_to_show=None,
    _reduced_embeddings=None,
    sample: float = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = False,
    custom_labels: bool = False,
    title: str = "<b>Documents and Topics</b>",
    width: int = 1200,
    height: int = 750,
) -> go.Figure:
    """
    Arguments:
            _topic_per_doc: Topic id assigned to each document.
            _docs: The documents.
            _topics_to_show: A selection of topics to visualize.
                    Not to be confused with the topics that you get from `.fit_transform`.
                    For example, if you want to visualize only topics 1 through 5:
                    `topics = [1, 2, 3, 4, 5]`.
            _reduced_embeddings: The 2D reduced embeddings of all documents in `_docs`.
            sample: The percentage of documents in each topic that you would like to keep.
                    Value can be between 0 and 1. Setting this value to, for example,
                    0.1 (10% of documents in each topic) makes it easier to visualize
                    millions of documents as a subset is chosen.
            hide_annotations: Hide the names of the traces on top of each cluster.
            hide_document_hover: Hide the content of the documents when hovering over
                                specific points. Helps to speed up generation of visualization.
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.
    """

    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(_topic_per_doc):
        s = np.where(np.array(_topic_per_doc) == topic)[0]  # Выбираем все индексы, соответствующие определенному топику
        #
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))

    indices = np.array(indices)
    # topic_per_doc[index] for index in indices
    monitor = None
    if _label_per_doc:
        idf = pd.DataFrame(
            {
                "topic": [_topic_per_doc[idx] for idx in indices],
                "doc": [_docs[idx] for idx in indices],
                "_label": [_label_per_doc[idx] for idx in indices],
            }
        )
        monitor = Monitor(set(_label_per_doc))
    else:
        idf = pd.DataFrame(
            {
                "topic": [_topic_per_doc[idx] for idx in indices],
                "doc": [_docs[idx] for idx in indices],
            }
        )

    if _reduced_embeddings is not None:
        embeddings_2d = _reduced_embeddings[indices]

        idf["x"] = embeddings_2d[:, 0]
        idf["y"] = embeddings_2d[:, 1]

    fig = go.Figure()

    unique_topics = set(_topic_per_doc)
    _topics_to_show = unique_topics if _topics_to_show is None else _topics_to_show

    non_selected_topics = unique_topics.difference(_topics_to_show)

    if len(non_selected_topics) == 0:
        non_selected_topics = [
            -1
        ]  # `bertopic` присваевает документам - `-1`, если он не относится ни к одному из кластеров

    selection = idf.loc[idf.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    # selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), "Other documents"]
    if _label_per_doc:
        selection.loc[len(selection), :] = [None, None, "", selection.x.mean(), selection.y.mean(), "Other documents"]
    else:
        selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode="markers+text",
            name="other",
            showlegend=False,
            marker=dict(color="#CFD8DC", size=5, opacity=0.5),
        )
    )

    # Теперь наносим каждый topic отдельно
    # for label, topic in zip(range(len(unique_topics)), unique_topics):
    for i, topic in enumerate(unique_topics):
        if topic in _topics_to_show and topic != -1:
            selection = idf.loc[idf.topic == topic, :]
            selection["text"] = ""

            _selection = pl.from_pandas(selection)

            # _selection.join(ldf, on=pl.col("topic"))
            if monitor is not None:
                _selection = _selection.with_row_count().with_columns([pl.count().over("_label").alias("label_len")])

                _selection_per_topk = (
                    _selection.sort("label_len", descending=True)
                    .unique(subset=["label_len"], maintain_order=True)
                    .top_k(3, by="label_len")
                )

                _topk = (
                    _selection.join(_selection_per_topk, on=pl.col("row_nr"))
                    .select(pl.col("_label"), pl.col("label_len"))
                    .to_arrow()
                )

                _label, _mass = [str(yi) for yi in _topk["_label"]], [str(pi) for pi in _topk["label_len"]]

                for li, mi in zip(_label, _mass):
                    # score is a scale mi / len(_selection).
                    monitor.update(li, int(mi) * 1.0 / len(_selection))

                _label_on_doc = "  ".join(
                    [yi.strip()[:20] for yi, pi in zip(_label, _mass)]
                )  # Будет показываться постоянно на облаке

                _label_on_topic = " ".join(
                    [yi.strip()[:22] + " (" + pi + ")" for yi, pi in zip(_label, _mass)]
                )  # Будет показываться справа по точке

            if not hide_annotations:
                if _label_per_doc:
                    selection.loc[len(selection), :] = [
                        None,
                        None,
                        "",
                        selection.x.mean(),
                        selection.y.mean(),
                        _label_on_doc[:50],
                    ]  # TODO: change topic to label
                else:
                    selection.loc[len(selection), :] = [
                        None,
                        None,
                        selection.x.mean(),
                        selection.y.mean(),
                        "",
                    ]  # TODO: change topic to label

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode="markers+text",
                    name=_label_on_topic if _label_per_doc else str(topic),
                    textfont=dict(size=12),
                    marker=dict(size=5, opacity=0.5),
                )
            )

    # Add grid in a 'plus' shape
    x_range = (idf.x.min() - abs((idf.x.min()) * 0.15), idf.x.max() + abs((idf.x.max()) * 0.15))
    y_range = (idf.y.min() - abs((idf.y.min()) * 0.15), idf.y.max() + abs((idf.y.max()) * 0.15))
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={"text": f"{title}", "x": 0.5, "xanchor": "center", "yanchor": "top", "font": dict(size=22, color="Black")},
        width=width,
        height=height,
    )

    fig.update_traces(textposition="top center")

    # fig.update_layout(
    #     title_text='Распределение по топикам и TOP_3 соответствующих класса (по частоте) на каждый топик '
    # )
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode="hide")

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    if monitor is not None:
        return fig, {k: v for k, v in monitor.compute().items() if v is not None and v is not np.nan}
    else:
        return fig


__all__ = ["evaluate"]

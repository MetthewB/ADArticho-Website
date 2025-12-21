import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import math
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import spacy
from spacy.lang.en import stop_words
from matplotlib.patches import Patch
import string
import pandas as pd

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def plot_humor_distributions(
    dataset1, dataset2, title1, title2, humor_order, humor_colors
):
    """
    Plots the distribution of humor types for two datasets side by side.
    """
    humor_counts1 = dataset1["humor_type"].value_counts()
    humor_counts1 = humor_counts1.reindex(
        humor_order, fill_value=0
    )  # Ensure fixed order

    humor_counts2 = dataset2["humor_type"].value_counts()
    humor_counts2 = humor_counts2.reindex(
        humor_order, fill_value=0
    )  # Ensure fixed order

    # Create a 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Plot the first dataset
    axes[0].bar(
        humor_counts1.index,
        humor_counts1.values,
        color=[humor_colors[cat] for cat in humor_counts1.index],
    )
    axes[0].set_title(title1, fontsize=16)
    axes[0].set_xlabel("Humor type", fontsize=14)
    axes[0].set_ylabel("Count", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)

    # Plot the second dataset
    axes[1].bar(
        humor_counts2.index,
        humor_counts2.values,
        color=[humor_colors[cat] for cat in humor_counts2.index],
    )
    axes[1].set_title(title2, fontsize=16)
    axes[1].set_xlabel("Humor type", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)

    # Add a legend for the colors
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=10) for color in humor_colors.values()
    ]
    fig.legend(
        legend_handles,
        humor_order,
        loc="upper center",
        ncol=5,
        title="Humor categories",
        fontsize=14,
        title_fontsize=14,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.87])  # Leave space for the legend
    plt.show()


def plot_humor_vs_funny(dataset1, dataset2, title1, title2, humor_order, humor_colors):
    """
    Plots humor type vs. funny score for two datasets side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Plot the first dataset (New Yorker)
    sns.boxplot(
        data=dataset1,
        x="humor_type",
        y="funny",
        order=humor_order,
        palette=humor_colors,
        ax=axes[0],
        showfliers=False,  # Exclude outliers
    )
    axes[0].set_title(title1, fontsize=16)
    axes[0].set_xlabel("Humor type", fontsize=14)
    axes[0].set_ylabel("Funny score", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)

    # Plot the second dataset (Oxford)
    sns.boxplot(
        data=dataset2,
        x="humor_type",
        y="funny_score",
        order=humor_order,
        palette=humor_colors,
        ax=axes[1],
        showfliers=False,  # Exclude outliers
    )
    axes[1].set_title(title2, fontsize=16)
    axes[1].set_xlabel("Humor type", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)

    # Add a legend for the colors
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=10) for color in humor_colors.values()
    ]
    fig.legend(
        legend_handles,
        humor_order,
        loc="upper center",
        ncol=5,
        title="Humor categories",
        fontsize=14,
        title_fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.show()


def plot_humor_vs_rank(dataset1, dataset2, title1, title2, humor_order, humor_colors):
    """
    Plots humor type vs. rank for two datasets side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Plot the first dataset (New Yorker)
    sns.boxplot(
        data=dataset1,
        x="humor_type",
        y="rank",
        order=humor_order,
        palette=humor_colors,
        ax=axes[0],
        showfliers=False,  # Exclude outliers
    )
    axes[0].set_title(title1, fontsize=16)
    axes[0].set_xlabel("Humor type", fontsize=14)
    axes[0].set_ylabel("Rank", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)

    # Plot the second dataset (Oxford)
    sns.boxplot(
        data=dataset2,
        x="humor_type",
        y="rank",
        order=humor_order,
        palette=humor_colors,
        ax=axes[1],
        showfliers=False,  # Exclude outliers
    )
    axes[1].set_title(title2, fontsize=16)
    axes[1].set_xlabel("Humor Type", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)

    # Add a legend for the colors
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=10) for color in humor_colors.values()
    ]
    fig.legend(
        legend_handles,
        humor_order,
        loc="upper center",
        ncol=5,
        title="Humor categories",
        fontsize=14,
        title_fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.show()


def plot_word_clouds_subplots(dataset, title_prefix, humor_order, humor_colors):
    """
    Generates word clouds for each humor type as subplots.
    """
    valid_humor_types = [
        humor_type
        for humor_type in humor_order
        if not dataset[dataset["humor_type"] == humor_type]["caption"].empty
    ]
    num_humor_types = len(valid_humor_types)
    cols = 5
    rows = math.ceil(num_humor_types / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    axes = axes.flatten()

    for i, humor_type in enumerate(valid_humor_types):
        text = " ".join(dataset[dataset["humor_type"] == humor_type]["caption"])
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            color_func=lambda *args, **kwargs: humor_colors[humor_type],
        ).generate(text)

        axes[i].imshow(wordcloud, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(humor_type, fontsize=16)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{title_prefix} different humor types", fontsize=18, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.show()


# --------------------------- Topic distribution ----------------------------------
def plot_matching_score_hist(dataset1, dataset2, dataset1_name, dataset2_name, topic_order, topic_colors
):
    """
    Plots the histogram of matching score for 2 dataset with Topic as hue
    Args:
        dataset1 (pd.Dataframe) :dataset 1 containing the captions, their topic and matching_score
        dataset2 (pd.Dataframe) :dataset 2 containing the captions, their topic and matching_score
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
    Return:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    common_bins = 50
    score_range = (
        min(dataset1["matching_score"].min(), dataset2["matching_score"].min()),
        max(dataset1["matching_score"].max(), dataset2["matching_score"].max()),
    )

    # New Yorker
    sns.histplot(
        data=dataset1,
        x="matching_score",
        hue=hue_col,
        hue_order=topic_order,
        palette=topic_colors,
        bins=common_bins,
        binrange=score_range,
        element="step",
        stat="count",
        common_norm=False,
        ax=axes[0],
        legend=False,  #
    )

    axes[0].set_title(dataset1_name, fontsize=16)
    axes[0].set_xlabel("Matching score", fontsize=14)
    axes[0].set_ylabel("Count", fontsize=14)

    # Oxford
    sns.histplot(
        data=dataset2,
        x="matching_score",
        hue=hue_col,
        hue_order=topic_order,
        palette=topic_colors,
        bins=common_bins,
        binrange=score_range,
        element="step",
        stat="count",
        common_norm=False,
        ax=axes[1],
        legend=True,
    )

    axes[1].set_title(dataset2_name, fontsize=16)
    axes[1].set_xlabel("Matching score", fontsize=14)
    axes[1].set_ylabel("")

    category_type = hue_col.replace("_", " ").title()
    fig.suptitle(
        f"Matching score histogram of {category_type} classification", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def plot_topic_distributions(
    dataset1, dataset2, dataset1_name, dataset2_name, topic_order, topic_colors
):
    """
    Plots the barplot of proportion of topic distributions for two datasets
    Args:
        dataset1 (pd.Dataframe) :dataset 1 containing the captions and their topic
        dataset2 (pd.Dataframe) :dataset 2 containing the captions and their topic
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
    Return:
        None
    """

    # Compute normalized topic counts
    topic_counts1 = (
        dataset1["Topic"]
        .value_counts(normalize=True)
        .reindex(topic_order, fill_value=0)
    )
    topic_counts2 = (
        dataset2["Topic"]
        .value_counts(normalize=True)
        .reindex(topic_order, fill_value=0)
    )

    x = np.arange(len(topic_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))

    # Dataset 1: solid bars
    ax.bar(
        x - width / 2,
        topic_counts1.values,
        width,
        color=[topic_colors[t] for t in topic_order],
        edgecolor="black",
    )

    # Dataset 2: hatched bars
    ax.bar(
        x + width / 2,
        topic_counts2.values,
        width,
        color=[topic_colors[t] for t in topic_order],
        edgecolor="black",
        hatch="///",
    )

    # Formatting
    ax.set_title("Topic distribution comparison", fontsize=18)
    ax.set_xlabel("Topic", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=12)

    # Legend only distinguishes datasets
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label=dataset1_name),
        Patch(facecolor="white", edgecolor="black", hatch="///", label=dataset2_name),
    ]
    ax.legend(handles=legend_handles, fontsize=13)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_topic_vs_funny(
    dataset1, dataset2, dataset1_name, dataset2_name, topic_order, topic_colors
):
    """
    Plots Topic vs. funny score for two datasets side by side as boxplots.
    Args:
        dataset1 (pd.Dataframe) :dataset 1 containing the captions and their topic
        dataset2 (pd.Dataframe) :dataset 2 containing the captions and their topic
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
    Return:
        None
    """

    data1 = [dataset1.loc[dataset1["Topic"] == topic, "funny"] for topic in topic_order]

    data2 = [
        dataset2.loc[dataset2["Topic"] == topic, "funny_score"] for topic in topic_order
    ]

    x = np.arange(len(topic_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))

    bp1 = ax.boxplot(
        data1,
        positions=x - width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    bp2 = ax.boxplot(
        data2,
        positions=x + width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, topic in zip(bp1["boxes"], topic_order):
        patch.set_facecolor(topic_colors[topic])
        patch.set_edgecolor("black")

    for patch, topic in zip(bp2["boxes"], topic_order):
        patch.set_facecolor(topic_colors[topic])
        patch.set_edgecolor("black")
        patch.set_hatch("///")

    for median in bp1["medians"] + bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title("Caption topic vs. funny score", fontsize=18)
    ax.set_xlabel("Topic", fontsize=14)
    ax.set_ylabel("Funny score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=12)

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label=dataset1_name),
        Patch(facecolor="white", edgecolor="black", hatch="///", label=dataset2_name),
    ]
    ax.legend(handles=legend_handles, fontsize=13)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_topic_vs_rank(
    dataset1, dataset2, dataset1_name, dataset2_name, topic_order, topic_colors
):
    """
    Plots Topic vs. rank for two datasets side by side as boxplots.
    Args:
        dataset1 (pd.Dataframe) :dataset 1 containing the captions and their topic
        dataset2 (pd.Dataframe) :dataset 2 containing the captions and their topic
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
    Return:
        None
    """
    data1 = [dataset1.loc[dataset1["Topic"] == topic, "rank"] for topic in topic_order]

    data2 = [dataset2.loc[dataset2["Topic"] == topic, "rank"] for topic in topic_order]

    x = np.arange(len(topic_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))

    bp1 = ax.boxplot(
        data1,
        positions=x - width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    bp2 = ax.boxplot(
        data2,
        positions=x + width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, topic in zip(bp1["boxes"], topic_order):
        patch.set_facecolor(topic_colors[topic])
        patch.set_edgecolor("black")

    for patch, topic in zip(bp2["boxes"], topic_order):
        patch.set_facecolor(topic_colors[topic])
        patch.set_edgecolor("black")
        patch.set_hatch("///")

    for median in bp1["medians"] + bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title("Caption topic vs. rank ", fontsize=18)
    ax.set_xlabel("Topic", fontsize=14)
    ax.set_ylabel("Rank", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=12)

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label=dataset1_name),
        Patch(facecolor="white", edgecolor="black", hatch="///", label=dataset2_name),
    ]
    ax.legend(handles=legend_handles, fontsize=13)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_topic_word_clouds_subplots(dataset, dataset_name, topic_order, topic_colors):
    """
    Generates word clouds for each topic type as subplots while excluding stop words.
    Args:
        dataset (pd.Datframe): dataset containing the captions and their topic
        dataset_name (str): Name of the dataset
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
        
    Return:
        None 
    """
    valid_topic_types = [
        topic_type
        for topic_type in topic_order
        if not dataset[dataset["Topic"] == topic_type]["caption"].empty
    ]
    num_topic_types = len(valid_topic_types)
    cols = 5
    rows = math.ceil(num_topic_types / cols)

    # Stopword list to remove
    nlp = spacy.load("en_core_web_sm")
    stopwords = stop_words.STOP_WORDS  # spaCy stopwords
    stopwords_wordcloud = set(STOPWORDS)  # Wordcloud stopwords

    custom_stopwords = {
        "get",
        "see",
        "let",
        "great",
        "anymore",
        "y'all",
        "said",
        "come",
        "go",
        "room",
        "new",
        "make",
        "take",
        "look",
        "thing",
        "going",
        "want",
        "know",
        "time",
        "looks",
        "really",
        "good",
        "bad",
        "tell",
        "hey",
        "yes",
        "no",
        "got",
        "think",
        "I'm",
        "don't",
        "talk",
        "told",
        "need",
        "don",
        "thought",
        "year",
        "people",
    }  # List of weak words
    stopwords = stopwords.union(custom_stopwords)
    stopwords = stopwords.union(set(string.ascii_lowercase))
    stopwords = stopwords.union(stopwords_wordcloud)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    axes = axes.flatten()

    for i, topic_type in enumerate(valid_topic_types):
        text = " ".join(dataset[dataset["Topic"] == topic_type]["caption"])
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stopwords,
            color_func=lambda *args, **kwargs: topic_colors[topic_type],
        ).generate(text)

        axes[i].imshow(wordcloud, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(topic_type, fontsize=16)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{dataset_name} word clouds for different topics",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.show()


def plot_topic_proportions_over_time(df, topic_order, topic_colors):
    """
    Plots a lineplot of the variation over the years of topic proportions
    Args:
        df (pd.Datframe): dataset containing the captions, their topic and year
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
        
    Return:
        None 
    """
    dataset = df[df['Topic'] !='Other'].copy() # Don't include unclassified captions
    
    # Calculate yearly topic proportions
    topic_props_per_year = (
        dataset.groupby("year")["Topic"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(16, 7))

    for topic in topic_order:
        subset = topic_props_per_year[topic_props_per_year["Topic"] == topic]
        ax.plot(
            subset["year"],
            subset["proportion"],
            label=topic,
            color=topic_colors[topic],
            linewidth=2,
        )

    ax.set_title("Topic proportions in New Yorker over time", fontsize=18)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Proportion of captions", fontsize=14)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11)

    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def plot_topic_variation_heatmap(df, topic_order):
    """
    Plots the heatmap of variation of the z-score of each topic's proportion in each year
    Args:
        df (pd.Dataframe) :dataset containing the captions, their topic and year
        topic_order (list): List of topics
    Return:
        None
    """   
    dataset = df[df['Topic'] !='Other'].copy() # Don't include unclassified caption
    
    # Calculate proportion of topics each year
    topic_prop_per_year = (
        dataset.groupby("year")["Topic"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    # Calculate z-score
    topic_prop_per_year["zscore"] = topic_prop_per_year.groupby("Topic")[
        "proportion"
    ].transform(lambda x: (x - x.mean()) / x.std())

    # Create dataframe for the heatmap
    heatmap_data = topic_prop_per_year.pivot(
        index="Topic", columns="year", values="zscore"
    ).reindex(topic_order)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(heatmap_data, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)

    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45)

    ax.set_title(
        "Topic proportion variation over time in New Yorker (z-score)", fontsize=18
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Topic", fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Deviation from topic mean (z-score)", fontsize=12)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------- Plotly ---------------------------------------------------------------
def topic_distribution_piechart_plotly(
    dataset1, dataset2, dataset1_name, dataset2_name, topic_order, topic_colors
):
    """
    Plots the piechart of the  distribution of topics for two datasets side by side.
    Args:
        dataset1 (pd.Dataframe) :dataset 1 containing the captions and their topic
        dataset2 (pd.Dataframe) :dataset 2 containing the captions and their topic
        dataset1_name (str): Name of dataset 1
        dataset2_name (str): Name of dataset 2
        topic_order (list): List of topics
        topic_colors (dict): Dictionary of colors to map each topic
    Return:
        None
    """
    topic_counts1 = dataset1["Topic"].value_counts()
    topic_counts1 = topic_counts1.reindex(
        topic_order, fill_value=0
    ).reset_index()  # Ensure fixed order

    topic_counts2 = dataset2["Topic"].value_counts()
    topic_counts2 = topic_counts2.reindex(
        topic_order, fill_value=0
    ).reset_index()  # Ensure fixed order

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=[dataset1_name, dataset2_name],
    )
    # Plot pie chart for dataset1
    fig.add_trace(
        go.Pie(
            labels=topic_counts1["Topic"],
            values=topic_counts1["count"],
            marker=dict(colors=[topic_colors[t] for t in topic_counts1["Topic"]]),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    # Plot pie chart for dataset2
    fig.add_trace(
        go.Pie(
            labels=topic_counts2["Topic"],
            values=topic_counts2["count"],
            marker=dict(colors=[topic_colors[t] for t in topic_counts2["Topic"]]),
            showlegend=False,  # legend only once
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=550,
        title_text=f"{dataset1_name} vs. {dataset2_name} Captions' Topic Distribution",
        title_x=0.5,
        title_font=dict(size=18),
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,  # Position below the pie charts
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def yearly_topic_piechart_plotly(df, topic_colors):
    """
    Plots a pie chart for each year in dataset_years of caption topics
    Args:
        df (pd.Dataframe) :dataset containing the captions, their topic and year
        topic_colors (dict): Dictionary of colors to map each topic
    Return:
        None
    """
    dataset = df[df['Topic'] !='Other'].copy()
    dataset_years = range(dataset['year'].min(),dataset['year'].max()+1)
    
    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[[{"type": "domain"}] * 3] * 3,
        subplot_titles=[str(year) for year in dataset_years],
    )

    for i, year in enumerate(dataset_years):
        df = (
            dataset[dataset["date"].dt.year == year]
            .groupby("Topic")["caption"]
            .count()
            .reset_index()
            .rename(columns={"caption": "Nbr_captions"})
        )

        fig.add_trace(
            go.Pie(
                labels=df["Topic"],
                values=df["Nbr_captions"],
                marker=dict(colors=[topic_colors[t] for t in df["Topic"]]),
                showlegend=(i == 0),  # legend only once
            ),
            row=i // 3 + 1,
            col=i % 3 + 1,
        )

    fig.update_layout(
        height=900,
        title_text="Caption topics per year in Newyorker",
    )

    fig.show()


def app_caption_time_prop(df, topic_order, topic_colors):
    """
    Creates an interactive window to see the evolution through time of different caption topics
    To run:
    app = app_caption_time_prop(newyorker_classified, topic_order, topic_colors)
    app.run(debug=True)
    """
    # ---- Data preparation  ----
    dataset = df[df['Topic'] != 'Other'].copy()
    
    topic_props_per_year = (
        dataset.groupby("year")["Topic"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    # ---- App creation ----
    app = Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="graph"),
            dcc.Checklist(
                id="checklist",
                options=[{"label": t, "value": t} for t in topic_order],
                value=topic_order,
                inline=True,
            ),
        ]
    )

    # ---- Callback ----
    @app.callback(Output("graph", "figure"), Input("checklist", "value"))
    def update_line_chart(selected_topics):
        df = topic_props_per_year[topic_props_per_year["Topic"].isin(selected_topics)]

        fig = px.line(
            df,
            x="year",
            y="proportion",
            color="Topic",
            labels=dict(proportion="Proportion of captions", year="Year"),
            color_discrete_map=topic_colors,
            category_orders={"Topic": topic_order},
        )

        # Center and style the title, and position the legend
        fig.update_layout(
            title={
                "text": "Evolution of Caption Topics in New Yorker Over Time",
                "x": 0.5,  # Center the title
                "xanchor": "center",
                "yanchor": "top",
            },
            title_font=dict(size=20),
            legend=dict(
                orientation="h",  # Horizontal legend
                y=-0.3,  # Position below the plot
                x=0.5,
                xanchor="center",
                yanchor="top",
                title_text=None,  # Remove "Legend" title
                traceorder="normal",
                font=dict(size=12),
            ),
            legend_tracegroupgap=20,  # Spacing between legend groups
        )

        # Adjust the number of rows and columns in the legend
        fig.update_layout(legend=dict(
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center",
            yanchor="top",
            title_text=None,
            traceorder="normal",
            font=dict(size=10),
            itemwidth=50,  # Adjust item width for better alignment
        ))

        return fig

    return app

def plotly_topic_variation_heatmap(df, topic_order):
    dataset = df[df['Topic'] !='Other'].copy()
    topic_order = topic_order[:-1]
    
    topic_prop_per_year = (
        dataset.groupby("year")["Topic"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    topic_prop_per_year["zscore"] = topic_prop_per_year.groupby("Topic")[
        "proportion"
    ].transform(lambda x: (x - x.mean()) / x.std())

    heatmap_data = topic_prop_per_year.pivot(
        index="Topic", columns="year", values="zscore"
    ).reindex(topic_order)

    fig = px.imshow(
        heatmap_data, 
        text_auto=".2f", # Limits decimals for cleaner text
        color_continuous_scale="RdBu_r", # Similar to 'coolwarm'
        aspect="auto"
    )

    fig.update_layout(
        title={
            'text': "Topic proportion variation over time in New Yorker (z-score)",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        xaxis_title="Year",
        yaxis_title="Topic",
        coloraxis_colorbar=dict(title="Deviation (z-score)")
    )

    fig.update_xaxes(tickangle=45)

    fig.show()
    return fig

def plot_length_distribution(lengths_arrays : list[pd.Series], labels, title = "", bins = 150, threshold = 300, overlap = True, color=None, ax=None) -> None:
    alpha = 0.8

    if ax is None:
        ax = plt.axes()

    if color == None:
        color = ["#ff6384","#fca4b7","#0096c8","#9be2fa"]
        color = [color[i] for i in range(len(lengths_arrays))]

    for i in range(len(lengths_arrays)) :
        lengths_arrays[i] = lengths_arrays[i][lengths_arrays[i] < threshold]

    if overlap:
        for lengths, label, c in zip(lengths_arrays, labels, color):
            print(f"{label} : mean {lengths.mean()}, std {lengths.std()}")

            ax.hist(lengths, bins=bins, density=True, alpha=alpha, label=label, color=c)
    else:
        ax.hist(lengths_arrays, bins=bins, density=True, label=labels, color=color)
        
    ax.set_xlabel('Caption length in charaters')
    ax.set_ylabel('Density of captions')

    ax.set_title(title)
    ax.legend()
    
    return ax
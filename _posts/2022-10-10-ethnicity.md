---
layout: post
title: Ethnicity
---

Exploring ethnic representation in movies over the years in our selected regions tells us a lot about how cinema reflects and changes with society. We gain insights into the complex interaction of culture, identity, and film in these regions. Our goal is to identify regions that present a richer variety of ethnicities on screen.

## Ethnicity melting pot

The first step of our analysis is to examine our actor dataset globally, considering their ethnicities across various regions. This first observation is crucial to have a better understanding of what follows.
This word cloud visually represents the ethnicities in each region, where the size of each ethnicity's names proportional to its frequency of occurence in the data.

<img src="plots/wordcloud_ethnicities.png" alt="Connections distribution" style="width: 100%;">

Ethnicities linked to actors in this CMU movie dataset are highly specific, making it challenging to form broader subgroups that ethically group different ethnicities together. This can generate challenge for the further exploration of ethicity representation:

- Presence of two many precise categories ungroupable
- Occurence of many ethnicithies with a very small frequencies which are part of the dataset and we have to use them to avoid biases and misrepresentation in the results

Film industries in our four region of the world are large melting pot of actors coming from different horizon. To gain a better visualization of the ethnic composition within this complex melting pot, we can examine the proportion of each ethnicity across different geographical areas.

<iframe src="plots/diversity_representation.html" style="width: 100%; height: 400px; border: none;"></iframe>

## Ethnic diversity over years

In order to go further in our exploration we are going to compare how this ethnicity diversity evolves across years in our four groups when taking into account the amount of produced movies. The large proportion of missing values, which is unevenly distributed among groups, makes these plots less reliable.

<iframe src="plots/ethny_diversity_percentage_2_2.html" style="width: 100%; height: 700px; border: none;"></iframe>

Examining our dataset film production trends across our four regions is crucial for parallel analysis alongside the evolution of ethnicity representation.

Looking at the total count of various ethnicities over the years, we see that the number of different ethnicities increased the most in Northern American and European films, following the trend of movies released count. In Eastern Asian and Indian films, we also notice an increase in the count, although the change over the years is smaller, especially for Indian films.

We observe that there are more Indian movies than European ones over the years. On the other hand, the count of different ethnicities in European movies is greater than in Indian movies, indicating that ethnic diversity is less pronounced in Indian films compared to European ones.

Finally we can look at the percentage of different ethnicities represented in each region for different years to try to make a conclusion to our analysis. Values for Eastern Asian unique ethnicities percentage are really non reADAble and comparable as we have a very few movies available. We observe a high variability of our curve over years. Globaly the other three curves show a slight decreasing pattern. This suggests that, despite an increase in the number of films produced, ethnic diversity representation did not rise sufficiently to maintain a constant percentage or show an increase. This implies that the number of movies produced increased more rapidly than the diversification of ethnicities represented on screen.

<div class="message">
	<h2>
		Recommendations for Joléa’s Career
	</h2>
	<p>
		Joléa believes that a mix of cultures and ethnicities is crucial. She prefers a career in a region with a rich on-screen ethnic diversity. We don't consider Asia in this analysis due to data limitations. The analysis suggests that for the highest ethnic diversity percentage in film, Europe is the best choice. Although Northern America has more ethnicities, its overall diversity proportion in films is smaller.
	</p>
</div>

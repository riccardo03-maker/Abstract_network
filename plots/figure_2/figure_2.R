library(tidyverse)
centrality_measures <- read_csv(
    "tf_idf_network/results/link_shuffle/centrality_measures.csv"
)

#plot degree distribution
degrees <- centrality_measures %>%
    group_by(Degree) %>%
    summarize(Density = n()/nrow(centrality_measures))
plot <- ggplot(data = degrees, aes(x = log(Degree), y = log(Density))) +
    geom_point(color = "blue") +
    labs(title = "Degree distribution link shuffling", x = "Degree (log)", y = "Density (log)")
ggsave("plots/figure_2/degree_distribution_link_shuffling.png", plot = plot, width = 6.67, height = 6.67)

#plot mean degree nn vs degree
plot <- ggplot(data = centrality_measures, aes(x = Degree, y = Mean_degree_NN)) +
    geom_point(color = "orange") +
    labs(title = "Mean degree nearest neighbours tf-idf network", x = "Degree", y = "Mean degree NN")
ggsave("plots/figure_2/mean_degree_nn_link_shuffling.png", plot = plot, width = 6.67, height = 6.67)    

#plot clustering coefficient vs degree
plot <- ggplot(data = centrality_measures, aes(x = Degree, y = Clustering_coefficient)) +
    geom_point(color = "red") +
    labs(title = "Clustering coefficients tf-idf network", x = "Degree", y = "Clustering coefficient")
ggsave("plots/figure_2/clustering_coefficient_link_shuffling.png", plot = plot, width = 6.67, height = 6.67)
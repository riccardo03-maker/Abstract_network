library(tidyverse)

#create plots for network built from scibert embeddings
centrality_measures <- read_csv(
    "scibert_network/results/title_embeddings/centrality_measures.csv"
)

#plot degree distribution
degrees <- centrality_measures %>%
    group_by(Degree) %>%
    summarize(Density = n()/nrow(centrality_measures))
plot <- ggplot(data = degrees, aes(x = log(Degree), y = log(Density))) +
    geom_point(color = "blue") +
    labs(title = "Degree distribution scibert network", x = "Degree (log)", y = "Density (log)") +
    theme_bw()
ggsave("plots/figure_3/degree_distribution_scibert_network.png", plot = plot, width = 6.67, height = 6.67)

#plot mean degree nn vs degree
plot <- ggplot(data = centrality_measures, aes(x = Degree, y = Mean_degree_NN)) +
    geom_point(color = "orange") +
    labs(title = "Mean degree nearest neighbours scibert network", x = "Degree", y = "Mean degree NN") +
    theme_bw()
ggsave("plots/figure_3/mean_degree_nn_scibert_network.png", plot = plot, width = 6.67, height = 6.67)    

#plot clustering coefficient vs degree
plot <- ggplot(data = centrality_measures, aes(x = Degree, y = Clustering_coefficient)) +
    geom_point(color = "red") +
    labs(title = "Clustering coefficients scibert network", x = "Degree", y = "Clustering coefficient") +
    theme_bw()
ggsave("plots/figure_3/clustering_coefficient_scibert_network.png", plot = plot, width = 6.67, height = 6.67)


#do the same plots for scale-free network
centrality_measures <- read_csv(
    "scibert_network/results/scale_free_network/centrality_measures.csv"
)

#plot degree distribution
degrees <- centrality_measures %>%
    group_by(Degree) %>%
    summarize(Density = n()/nrow(centrality_measures))
plot <- ggplot(data = degrees, aes(x = log(Degree), y = log(Density))) +
    geom_point(color = "blue") +
    labs(title = "Degree distribution scale free network", x = "Degree (log)", y = "Density (log)") +
    theme_bw()
ggsave("plots/figure_3/degree_distribution_scale_free_network.png", plot = plot, width = 6.67, height = 6.67)

#plot mean degree nn vs degree
plot <- ggplot(data = centrality_measures, aes(x = Degree, y = Mean_degree_NN)) +
    geom_point(color = "orange") +
    labs(title = "Mean degree nearest neighbours scale free network", x = "Degree", y = "Mean degree NN") +
    theme_bw()
ggsave("plots/figure_3/mean_degree_nn_scale_free_network.png", plot = plot, width = 6.67, height = 6.67)    

#plot clustering coefficient vs degree
plot <- ggplot(data = centrality_measures, aes(x = Degree, y = Clustering_coefficient)) +
    geom_point(color = "red") +
    labs(title = "Clustering coefficients scale free network", x = "Degree", y = "Clustering coefficient") +
    theme_bw()
ggsave("plots/figure_3/clustering_coefficient_scale_free_network.png", plot = plot, width = 6.67, height = 6.67)

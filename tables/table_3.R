#create the third table of the report
library(tidyverse)
connections_between_cathegories <- read_csv(
    "tf_idf_network/results/connections_between_cathegories.csv"
)

all_physics_topics = c("Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", 
                      "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", 
                      "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", 
                      "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics",
                        "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics")

for(i in (1:22)){
  #select one of the cathegories
  max_connections_for_topic <- connections_between_cathegories %>%
    select(c("...1", all_physics_topics[i])) %>%
    mutate("Cathegory" = all_physics_topics[i])

  colnames(max_connections_for_topic) <- c("Max_connected_cathegory", "Connections", "Cathegory")
  #select the cathegory with the highest number of connections
  max_connections_for_topic <- max_connections_for_topic %>%
    filter(Connections == max(Connections)) %>%
    select(c("Cathegory", "Max_connected_cathegory", "Connections"))
  
  #create a unique table
  if(i == 1){
    max_connections <- max_connections_for_topic
  }
  else{
    max_connections <- bind_rows(max_connections, max_connections_for_topic)
  }
}

write_csv(max_connections, "tables/table_3.csv")
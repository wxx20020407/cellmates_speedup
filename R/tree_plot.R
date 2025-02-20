# small test for ggtree

library(ggtree)
library(dplyr)
library(tibble)
library(ggplot)
library(tidyr)

tree <- rtree(5)

cn_matrix <- matrix(c(2,2,2,2,2,3,3,3,3,3,
               1,1,1,1,1,2,2,2,2,2,
               1,1,1,2,2,3,3,3,3,3,
               2,2,2,2,2,2,2,2,2,1,
               4,4,3,3,2,2,2,1,1,1), nrow = 5)
colnames(cn_matrix) <- 1:10
cn_tbl <- cn_matrix %>%
  as_tibble(col) %>%
  bind_cols(tibble(id = tree$tip.label)) %>%
  gather(key = "bin", value = "cn", -id) %>%
  mutate(bin = as.numeric(bin))

cn_tbl

p <- ggtree(tree)

p2 <- facet_plot(p, panel="cn",
                 data=cn_tbl,
                 geom=geom_point, aes(x=bin, fill=cn))


p2

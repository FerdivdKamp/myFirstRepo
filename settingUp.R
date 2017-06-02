# Setting up a package
install.packages("Rtools")
install.packages("devtools")
library("devtools")
devtools::install_github("klutometis/roxygen") # will not work in LU, but can still continue working on the package
library(roxygen2) # will not work in LU, but can still continue working on the package


# Test Sys path for Rtools
Sys.getenv("PATH")

getwd()
create("pack")

install("pack")

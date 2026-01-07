# install.packages("anticlust")

library("anticlust")

options(max.print=10000000)

# Get command-line arguments
args <- commandArgs(trailingOnly=TRUE)
dataset <- paste0(args[1], ".csv")
random_seed <- as.integer(args[2])

# Load data set
file_path <- paste0("datasets/", dataset)
X <- read.csv(file_path, header=FALSE, sep=",")

# Get number of objects
N <- nrow(X)

# Get nunber of anticlusters 
n_anticlusters <- as.integer(ceiling(N / 2))

# Set random seed
set.seed(random_seed)

# Start stopwatch
start_time <- Sys.time()

# Perform anticlustering
anticlusters <- anticlustering(X,K = n_anticlusters,objective = "variance", method = "exchange",repetitions = 1)

# Stop stopwatch
end_time <- Sys.time()

# Calculate elapsed time
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

print(anticlusters)
print(paste("Elapsed_time =", elapsed))
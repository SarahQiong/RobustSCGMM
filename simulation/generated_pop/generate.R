library(MixSim)

args <- commandArgs(TRUE)
args <- as.array(args)
seed <- as.integer(args[1])
set.seed(seed)

K <- as.integer(args[2])
d <- as.integer(args[3])
omega <- as.double(args[4])


param <-
  MixSim(
    MaxOmega = omega,
    K = K,
    p = d,
    sph = FALSE,
    hom = FALSE
  )

weights <- param$Pi
means <- param$Mu
covs <- param$S

write.table(
  weights,
  file = paste0(
      "true_param/weights_seed_",
      as.character(seed),
      "_ncomp_",
      as.character(K),
      "_d_",
      as.character(d),
      "_maxoverlap_",
      as.character(omega),
      ".txt"
  ),
  row.names = FALSE,
  col.names = FALSE
)

write.table(
  means,
  file = paste0(
    "true_param/means_seed_",
    as.character(seed),
    "_ncomp_",
    as.character(K),
    "_d_",
    as.character(d),
    "_maxoverlap_",
    as.character(omega),
    ".txt"
  ),
  row.names = FALSE,
  col.names = FALSE
)

write.table(
  covs,
  file = paste0(
    "true_param/covs_seed_",
    as.character(seed),
    "_ncomp_",
    as.character(K),
    "_d_",
    as.character(d),
    "_maxoverlap_",
    as.character(omega),
    ".txt"
  ),
  row.names = FALSE,
  col.names = FALSE
)
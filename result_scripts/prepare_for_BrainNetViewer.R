library(rsfcNet)
library(RcppCNPy)
library(data.table)

path = "...\\results_ABIDEI\\feature_importance"
path_save = "...\\results_ABIDEI\\feature_importance\\"
  
path_varconet_aal_abide <- file.path(path, "AAL_feature_importance.csv")
data_varconet_aal_abide <- as.matrix(fread(path_varconet_aal_abide))
path_varconet_aicha_abide <- file.path(path, "AICHA_feature_importance.csv")
data_varconet_aicha_abide <- as.matrix(fread(path_varconet_aicha_abide))

top_values <- sort(data_varconet_aal_abide, decreasing = TRUE)[1:20]
data_varconet_aal_abide <- ifelse(data_varconet_aal_abide %in% top_values, data_varconet_aal_abide, 0)

top_values <- sort(data_varconet_aicha_abide, decreasing = TRUE)[1:20]
data_varconet_aicha_abide <- ifelse(data_varconet_aicha_abide %in% top_values, data_varconet_aicha_abide, 0)

mat_varconet_aicha <- matrix(0, nrow = 384, ncol = 384)
tri_idx <- which(upper.tri(mat_varconet_aicha))
mat_varconet_aicha[tri_idx] <- data_varconet_aicha_abide
mat_varconet_aicha <- mat_varconet_aicha + t(mat_varconet_aicha)

mat_varconet_aal <- matrix(0, nrow = 166, ncol = 166)
tri_idx <- which(upper.tri(mat_varconet_aal))
mat_varconet_aal[tri_idx] <- data_varconet_aal_abide
mat_varconet_aal <- mat_varconet_aal + t(mat_varconet_aal)

aal_coords <- npyLoad(file.path(path, "AAL3_coords.npy"))
aicha_coords <- npyLoad(file.path(path, "AICHA_coords.npy"))

nonzero_rows <- apply(mat_varconet_aicha, 1, function(row) any(row != 0))
row_indices <- which(nonzero_rows)
aicha_coords <- aicha_coords[nonzero_rows, , drop = FALSE]
mat_varconet_aicha <- mat_varconet_aicha[nonzero_rows,nonzero_rows]

nonzero_rows <- apply(mat_varconet_aal, 1, function(row) any(row != 0))
row_indices <- which(nonzero_rows)
aal_coords <- aal_coords[nonzero_rows, , drop = FALSE]
mat_varconet_aal <- mat_varconet_aal[nonzero_rows,nonzero_rows]

write_brainNet(
  aicha_coords,
  node.size = rep(0, length(aicha_coords)),
  mat_varconet_aicha,
  directory = path_save,
  name = 'brainNet_AICHA_ASD'
)

write_brainNet(
  aal_coords,
  node.size = rep(0, length(aal_coords)),
  mat_varconet_aal,
  directory = path_save,
  name = 'brainNet_AAL_ASD'
)



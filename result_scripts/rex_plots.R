library(data.table)
library(ReX)
library(ggplot2)

path = "...\\results_HCP\\ReX_files"

#PCC
path_pcc <- file.path(path, "rex_AICHA_PCC_80_samples.csv")
data_pcc <- fread(path_pcc)
x <- data_pcc[, 4:ncol(data_pcc)]
sub <- data_pcc[,2]
sess <- data_pcc[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_pcc <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_pcc, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_pcc, file = file.path(path, "PCC_ICC_80_samples.csv"), row.names = FALSE)

#VarCoNet
path_varconet <- file.path(path, "rex_AICHA_VarCoNet_80_samples.csv")
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_varconet, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_varconet, file = file.path(path, "VarCoNet_ICC_80_samples.csv"), row.names = FALSE)

#VAE
path_vae <- file.path(path, "rex_AICHA_VAE_KSVD_80_samples.csv")
data_vae <- fread(path_vae)
x <- data_vae[, 4:ncol(data_vae)]
sub <- data_vae[,2]
sess <- data_vae[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_vae <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_vae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_vae, file = file.path(path, "VAE_ICC_80_samples.csv"), row.names = FALSE)


#AE
path_ae <- file.path(path, "rex_AICHA_AE_KSVD_80_samples.csv")
data_ae <- fread(path_ae)
x <- data_ae[, 4:ncol(data_ae)]
sub <- data_ae[,2]
sess <- data_ae[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_ae <- data.frame(lme_ICC_1wayR(x, sub, sess))
rex_plot.var.field(df_icc_ae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
write.csv(df_icc_ae, file = file.path(path, "AE_ICC_80_samples.csv"), row.names = FALSE)


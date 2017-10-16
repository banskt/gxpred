library(glmnet)

#X_train <- read.csv('C21orf56_gt.csv', header = FALSE, sep = ",")
#Y_train <- read.csv('C21orf56_gx.csv', header = FALSE, sep = ",")
#X_pred <- read.csv('C21orf56_gtex_gt.csv', header = FALSE, sep = ",")
#Y_gtex <- read.csv('C21orf56_gtex_gx.csv', header = FALSE, sep = ",")

X_train <- read.csv('YBEY_gt.csv', header = FALSE, sep = ",")
Y_train <- read.csv('YBEY_gx.csv', header = FALSE, sep = ",")
X_pred <- read.csv('YBEY_gtex_gt.csv', header = FALSE, sep = ",")
Y_gtex <- read.csv('YBEY_gtex_gx.csv', header = FALSE, sep = ",")

X <- t(X_train)
Xnew <- t(X_pred)
Y <- Y_train[,1]
Ynew <- Y_gtex[,1]

seed <- sample(1:2017, 1)
set.seed(seed)
groupid <- sample(1:10, dim(X)[1], replace = TRUE)

#Y <- scale(Y, center = TRUE, scale = TRUE)
#set.seed(1025)

fit <- cv.glmnet(as.matrix(X),
                 as.vector(Y),
                 nfolds = 10,
                 alpha = 0.5,
                 keep = TRUE,
                 foldid = groupid,
                 parallel = FALSE)

# Pull info from fit to find the best lambda   
fit.df <- data.frame(fit$cvm, fit$lambda, 1:length(fit$cvm))
# Needs to be min or max depending on cv measure (MSE min, AUC max, ...)
best.lam <- fit.df[which.min(fit.df[,1]),]
cvm.best <- best.lam[,1]
lambda.best <- best.lam[,2]
# Position of best lambda in cv.glmnet output
nrow.best <- best.lam[,3]
# Get the betas from the best lambda value
ret <- as.data.frame(fit$glmnet.fit$beta[,nrow.best])
ret[ret == 0.0] <- NA
# Pull the non-zero betas from model
bestbetas <- as.vector(ret[which(!is.na(ret)),])

pred.mat <- fit$fit.preval[, nrow.best]
res <- summary(lm(Y ~ pred.mat))

rsq <- res$r.squared
pval <- res$coef[2,4]
cat(cvm.best, nrow.best, lambda.best, length(bestbetas), rsq, pval)

betas <- fit$glmnet.fit$beta[,nrow.best]
ypred <- rowSums(as.matrix(Xnew) %*% as.vector(betas))
plot(ypred, Ynew)
res_gtex <- summary(lm(Ynew ~ ypred))
rsq <- res_gtex$r.squared
pval <- res_gtex$coef[2,4]
cat(rsq, pval)

library("FactoMineR")
library("factoextra")
library("caret")

current_dir = getwd()
current_wd = list()

if ('C:/Users/jjonus/documents' %in% current_dir == TRUE) current_wd = paste('C:\\Users\\','jjonus','\\Google Drive\\Kaggle\\Advanced House Prices',sep = "") else current_wd = paste('C:\\Users\\','jstnj','\\Google Drive\\Kaggle\\Advanced House Prices',sep="")

all_data = read.csv(paste(current_wd,'\\all_data.csv',sep=""))

idCol = grep(c("Id","GarageYrBlt","YearBuilt","YearRemodAdd"),names(all_data))

# take out values with nearZeroVar

metrics_out = nearZeroVar(all_data,freqCut = 85/15,saveMetrics = FALSE,names=TRUE)
metrics_out
drops <- c(metrics_out)
drops_2 = c("Id","GarageYrBlt","YearBuilt","YearRemodAdd")

df_famda = all_data[, - idCol]
df_famda = df_famda[ , !(names(df_famda) %in% drops)]
df_famda = df_famda[ , !(names(df_famda) %in% drops_2)]

# perform FAMD

res.famd = FAMD(df_famda,ncp=20,graph = FALSE)

# analyze eigen values and eigen vectors

eig.val = get_eigenvalue(res.famd)
eig.val

fviz_screeplot(res.famd)

# analyze variables

var = get_famd_var(res.famd)

# all variables

fviz_famd_var(res.famd,repel=TRUE)

fviz_contrib(res.famd,"var",axes=1)
fviz_contrib(res.famd,"var",axes=2)
fviz_contrib(res.famd,"var",axes=3)

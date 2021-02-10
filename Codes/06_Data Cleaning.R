# The code is written by Jawwad Shadman Siddique | R11684947
# Date of Submission: 12 / 09  / 2020
# This Step performs Data Cleaning 
# It cleans the non-standard missing values by converting it into NA's
# It removes the outlier data by inferencing from the Boxplot
# Total Raw Data initial = 1030
# Total Data after cleaning = 968

# Checking the directory
# getwd()
# setwd("D:/R Practice Programs")
# getwd()

# reading the culvert data
a = read.csv('concrete.csv')

summary(a)
colnames(a)

# checking the total 'na' values
sum(is.na(a))


# checking the outlier data
boxplot(a$Cement) # no outlier
boxplot(a$BlastFurnaceSlag) # 4 outliers removed
boxplot(a$FlyAsh) # no outlier
boxplot(a$Water) # 14 outliers removed
boxplot(a$Superplasticizer) # 11 outliers removed
boxplot(a$CoarseAgg) # no outliers
boxplot(a$FineAgg) # 5 outliers removed
boxplot(a$AgeDays) # None
boxplot(a$W.C) # 18 outliers
boxplot(a$StrengthMPa) # 15 outliers removed

# Removing the outlier data by subsetting and checking with the boxplot

new = subset(a, BlastFurnaceSlag <= 342)
new = subset(new, Water <= 237)
new = subset(new, Water > 127.5)
new = subset(new, Superplasticizer <= 22)
new = subset(new, StrengthMPa <= 76.50)
new = subset(new, W.C <= 1.6)

# Checking the boxplots after removal of the outliers

boxplot(new$Cement)
boxplot(new$BlastFurnaceSlag)
boxplot(new$FlyAsh)
boxplot(new$Water)
boxplot(new$Superplasticizer)
boxplot(new$CoarseAgg)
boxplot(new$FineAgg)
boxplot(new$AgeDays)
boxplot(new$W.C) 
boxplot(new$StrengthMPa)


# writing the cleaned dataset to the csv file
# write.csv(new, 'concrete_clean.csv')
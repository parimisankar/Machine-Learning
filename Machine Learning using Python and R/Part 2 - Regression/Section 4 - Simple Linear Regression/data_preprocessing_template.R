# Data Preprocessing
# 1. Import libraries
# - Not needed in R

# 2. Importing the dataset
dataset = read.csv('Salary_Data.csv')

# 3. Split data into features/independent variables(X) and dependent(y) variables
# - Not needed in R

# Data Manipulation and Cleaning
# 4. Taking care of missing data
dataset$YearsExperience = ifelse(is.na(dataset$YearsExperience),
                                 ave(dataset$YearsExperience, FUN = function(x) mean(x, na.rm = TRUE)),
                                 dataset$YearsExperience)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# # 5. Encoding categorical data
# dataset$Country = factor(dataset$Country,
#                          levels = c('France', 'Spain', 'Germany'),
#                          labels = c(1, 2, 3))
# dataset$Purchased = factor(dataset$Purchased,
#                            levels = c('Yes', 'No'),
#                            labels = c(1, 0))

# # 6. Feature Scaling - helps in rapid conversion of ML Models
# dataset[, 2:3] = scale(dataset[, 2:3])
# dataset[, 2:3] = scale(dataset[, 2:3])

# Train and test samples
# 7. Splitting the dataset into the Training set and Test set
# install.packages('caTools') - Install package caTools from packages tab or use command
# library(caTools) - Selects and imports the library
library(caTools)
set.seed(123) # 123 can be any random number
split = sample.split(dataset$Salary, SplitRatio = 2/3) # Split using dependent variable (in this case, Salary)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# Ensure size(Training_set) > size(Test_set)
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

# Model Development - Simple Regression
# 8. Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# 9. Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# 10. Evaluate/Diagnose the Model - Mean Absolute Percentage Error (MAPE)
# Look at the information in Regressor
# summary(regressor) 
# - Look for Signif. codes (more starts --> better)
# - Look for P value (<0.005 is good)

# 11. Model Visualization
# Visualising the Training set results
# install.packages('ggplot2') - Install package ggplot2 from packages tab or use command
# library(ggplot2) - Selects and imports the library
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'green') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  ggtitle('Salary vs Experience(Green - Training Set, Red - Test Set)') +
  xlab('Years of experience') +
  ylab('Salary')
  
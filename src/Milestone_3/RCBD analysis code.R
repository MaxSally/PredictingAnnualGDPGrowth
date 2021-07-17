library(nlme)                    # You must always load the library to do the analysis
library(emmeans)                # You must always load the library to do the analysis

RMSE_data <-read.csv(file="C:\\Users\\izzat\\OneDrive\\UNL_semesters\\Sp21\\Deep_Learning\\Milestone 3\\Milestone 3 result.csv", header=T)        
head(RMSE_data)
str(RMSE_data)
#Notice that City is being read as an integer... But we know 1 is some city, say Lincoln.
#Thus, we want to treat this as a factor.
RMSE_data$Country <- as.factor(RMSE_data$Country)
mod.fit <- lme(TestRMSE ~ LV,  random = ~1|Country, data = RMSE_data)   # RCBD model 
anova(mod.fit)                                              # ANOVA table
summary(mod.fit)                                            # summary of effects 

emmeans(mod.fit, pairwise ~ LV)
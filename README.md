# house-prices-advanced-regression-techniques

## Index
1. [Introduction](#introduction)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Data Encoding](#data-encoding)
4. [Feature Selection](#feature-selection)
5. [Model Building](#model-building)
6. [Results](#results)

## 1. Introduction

The dataset provided dives deep into the residential housing landscape of Ames, Iowa. With 79 insightful variables on offer, it paints a detailed picture of almost every facet of residential homes. The primary objective of this project is to harness these features to predict the sales prices of the houses. Such prediction tasks not only hold academic interest but also have significant real-world implications, aiding both sellers in setting competitive prices and buyers in making informed decisions.

## 2. Exploratory Data Analysis (EDA)

### Sale Price Distribution:
Our primary target variable, `SalePrice`, showcases a slightly right-skewed distribution, as visualized in:

![Sale Price Distribution](plots/price_dist.png)

This right_skewed shape indicates that many houses are clustered around the median price with only a few sell for considerably higher prices. Fortunately, the distribution doesn't reveal significant outliers, which means our models won't be unduly influenced by extreme values—leading to a more generalized and reliable prediction.

### Missing Data Insights:
The patterns of missing data across features are captured vividly in the two plots:

![Missing Data 1-41](plots/clean_missing_1-41.png)
![Missing Data 42-81](plots/clean_missing_42-81.png)

A deeper look into these visualizations reveals:

- Only a handful of features have significant missing data (above 15%). It's a logical choice to exclude these from our analysis to maintain data quality.
  
- Interestingly, most of the missing values aren't randomly spread across the dataset. They're concentrated in specific rows, indicating potential issues in the data collection or entry process for those records. Given this pattern, it's prudent to remove such rows to ensure the robustness of our predictive models.

Through EDA, we gain valuable insights into the data's characteristics and challenges, guiding our subsequent preprocessing steps.

## 3. Encoding Categorical Features

Handling categorical data is crucial in this dataset, especially considering the split of numeric to categorical features is 36:33, indicating a significant portion is categorical. I opted to replace each category with its corresponding median of the `SalePrice`. 

However, just using the median might lead to overfitting, especially when a category has very few samples. To tackle this, I employed the "smoothed median encoding" technique. The equation for smoothed median is:

\[ \text{Smoothed Value} = \frac{n \times \text{Category Median} + m \times \text{Global Median}}{n + m} \]

Where:
- \( n \) is the total number of samples in that category.
- \( m \) is a smoothing parameter (a weight for the global median to avoid overfitting). 

After encoding the categories with their smoothed medians, I added an additional layer of complexity: random noise. This is a small random adjustment, within ±5% of the original value, to ensure that the model doesn't overly fixate on specific price points, enhancing the generalization capability.

## 4. Feature Selection

Feature selection is a critical step in any machine learning project. Choosing the right subset of features can lead to simpler, more interpretable, and often more accurate models. In the context of this housing dataset, with its rich set of 79 variables, the challenge was to distill the essence of the data while discarding redundant or irrelevant information.

We employed a two-pronged strategy for feature selection:

### 1. **Feature Importance using Random Forests**:
Random Forests can rank features based on their importance in predicting the target variable. By training a Random Forest on our data and examining the importance of each feature, we identified those features that had the most impact on predicting house prices.

### 2. **Recursive Feature Elimination (RFE) with Gradient Boosting Machine (GBM)**:
RFE is a method that fits the model repeatedly on iterative subsets of features. At each iteration, it discards the least important feature until the desired number of features is reached. For this, we used the powerful GBM as our estimator. GBM, being a gradient boosting method, builds the model in a stage-wise fashion optimizing for accuracy, which makes it an excellent choice for RFE.

Combining the results from both methods, we found a significant overlap in the features (36 out of 40) they deemed important. This provided us with a robust, reliable set of 36 features that were used for building our final predictive model. The harmony between the two feature selection methods boosted our confidence in the selected subset.

![Top Features](plots/features_selection.png) 

This synergy in feature selection ensured that we captured the most relevant aspects of the data while keeping the model complexity in check.


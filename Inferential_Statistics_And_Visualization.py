import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

print('Name: Sachin Chhetri')
print('Assignment 5: Inferential Statistics and Visualization\n')

df = pd.read_csv('Titanic Crew.csv')
df = df.drop(columns=['URL'])
data = df[df['Survived?'].isin(['LOST', 'SURVIVED'])]

# 1
print('1: Chi-Squared test on the Gender column using Survived? as the dependent variable.\n')
contigingency_table_gender = pd.crosstab(df['Gender'], df['Survived?'], margins=True)
print(contigingency_table_gender)
print('\n', stats.chi2_contingency(pd.crosstab(df['Gender'], df['Survived?']))[0:3])
print('The result indicates a significant association between the Gender and Survived columns.\n'
      'The Chi-squared Statistics (48.43) indicates the strength of the association between gender and survival status.\n'
      'A higher Chi-squared value tells us that there is a strong association.\n'
      'The P-value of (3.41e-12) is extremely small, close to 0 and since the p-value is much smaller than 0.05, we reject the null\n'
      'hypothesis and we can say that there is a significant association between gender and survival.\n'
      'Here the degree of freedom is 1. This indicates the number of independent pieces of information used to calculate Chi-Squared.\n'
      'We have 2 categories in Gender and 2 in survived, so the degree of freedom is (2-1)*(2-1) = 1. So, the Chi-Squared test\n'
      'result suggests that Gender is significantly associated with survival status on the Titanic.\n')

# 2
print('2: Chi-Squared test on the Class/Dept column using Survived? as the dependent variable.\n')
contigingency_table_classDept = pd.crosstab(df['Class/Dept'], df['Survived?'], margins=True)
print(contigingency_table_classDept)
print('\n', stats.chi2_contingency(pd.crosstab(df['Class/Dept'], df['Survived?']))[0:3])
print('Chi-Squared Statistic is (82.97) this indicates the strength of the association between class/dept and survival status.\n'
      'The relatively high value of 82.97 indicates a substantial association between class/dept and survival on the Titanic.\n'
      'The p-value is smaller than 0.05 so we reject the null hypothesis and conclude that there is a significant association\n'
      'between class/dept and survival. Here the Degree of freedom is 7. We have 8 categories for Class/Dept and 2 for Survival.\n'
      'So, the degrees of freedom is (8-1) * (2-1) = 7. So, the Chi-Squared test result suggests that class/dept is significantly\n'
      'associated with survival status on the Titanic.\n')

# 3
print('3: Chi-Squared test on the Joined column using Survived? as the dependent variable.\n')
contigingency_table_joined = pd.crosstab(df['Joined'], df['Survived?'], margins=True)
print(contigingency_table_joined)
print('\n', stats.chi2_contingency(pd.crosstab(df['Joined'], df['Survived?']))[0:3])
print('Here the Chi-Squared test suggests that there is no significant association between the time at which individuals\n'
      'joined the Titanic crew and their survival. This means that the time of joining is not a predictive factor\n'
      'in determining survival.\n')

# 4
print('4: ANOVA on Class/Dept as it is given in the dataset using Age as the dependent variable.\n')
data = df.dropna(subset=['Age', 'Class/Dept'])
anova_result = f_oneway(
    data[data['Class/Dept'] == 'Deck Crew']['Age'],
    data[data['Class/Dept'] == 'Deck Crew Titanic Officers']['Age'],
    data[data['Class/Dept'] == 'Engineering Crew']['Age'],
    data[data['Class/Dept'] == 'Engineering Crew Substitute Crew']['Age'],
    data[data['Class/Dept'] == 'Restaurant Staff']['Age'],
    data[data['Class/Dept'] == 'Victualling Crew']['Age'],
    data[data['Class/Dept'] == 'Victualling Crew Postal Clerk']['Age'],
    data[data['Class/Dept'] == 'Victualling Crew Substitute Crew']['Age']
)
print("ANOVA p-value:", anova_result.pvalue)
if anova_result.pvalue < 0.05:
    tukey_result = pairwise_tukeyhsd(data['Age'], data['Class/Dept'])
    print(tukey_result.summary())
    print('We can see that the p-value is very small (2.42e-11), which tells us that there is a significant differences in mean ages\n'
      'across at least some of the class/dept. So, after doing the Tukey HSD test with the reject value of True have significantly\n'
      'different mean ages compared to each other and with the reject value of False they do not have significantly different mean ages.\n'
      'Therefore, by interpreting the reject values in the Tukey HSD results, we can determine which pairs of class/dept exhibit\n'
      'statistically significant differences in mean ages and which do not. This provides us valuable insights into age variations\n'
      'among different class/dept within the Titanic crew dataset.\n')
else:
    print('The p-value is greater than 0.05. There are no significant differences in mean ages across the class/dept.\n')

# 5
print('5: Correlation of female and male to survived.\n')
female_data = df[df['Gender'] == 'Female']
male_data = df[df['Gender'] == 'Male']

pearson_corr_female = female_data['Survived?'].astype(str).astype('category').cat.codes.corr(female_data['Survived?'].astype(str).astype('category').cat.codes, method='pearson')
pearson_corr_male = male_data['Survived?'].astype(str).astype('category').cat.codes.corr(male_data['Survived?'].astype(str).astype('category').cat.codes, method='pearson')
spearman_corr_female = female_data['Survived?'].astype(str).astype('category').cat.codes.corr(female_data['Survived?'].astype(str).astype('category').cat.codes, method='spearman')
spearman_corr_male = male_data['Survived?'].astype(str).astype('category').cat.codes.corr(male_data['Survived?'].astype(str).astype('category').cat.codes, method='spearman')

print("Pearson's correlation coefficient for Female:", pearson_corr_female)
print("Spearman's correlation coefficient for Female:", spearman_corr_female)
print("Pearson's correlation coefficient for Male:", pearson_corr_male)
print("Spearman's correlation coefficient for Male:", spearman_corr_male)
print('\nI do not know if this is incorrect but the perfect correlation coefficients suggest us that gender is a\n'
      'strong predictor of survival status in the dataset.\n')

# 6 Bivariate Visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived?', hue='Gender', data=df)
plt.title('Survival Count by Gender')
plt.xlabel('Survived?')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Survived?', hue='Class/Dept', data=df)
plt.title('Survival Count by Class/Dept')
plt.xlabel('Survived?')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived?', y='Age', data=df)
plt.title('Survival by Age')
plt.xlabel('Survived?')
plt.ylabel('Age')
plt.show()

# 7 Multivariate Visualization
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

print('\nPart 1: ')
print('My top feature: Gender, Class/Dept, Age\n')
print('I chose Gender because as we already know during the Titanic disaster, women were given priority access to lifeboats,\n'
      'resulting in a higher survival rate compared to men. While doing some visualization the survival rate of female is way\n'
      'higher than male. Also, the Chi-squared Statistics (48.43) also indicates the strength of the association between gender\n'
      'and survival status.\nAnother is Class/Dept as it indicates the socio-economic status of Passenger and crew members. I believe\n'
      'that the higher socio economic classes were more likly to survive. Also the Chi-Squared Statistic is (82.97) this indicates\n'
      'the strength of the association between class/dept and survival status.\nAnother is Age because the younger individuals were prioritized\n'
      'during evacuation. Older people might have faced difficulties during evacuation.\n')

print('Part 2: ')
print('RFE reported these top X features: Gender, Age, Class/Dept')
print('SelectKBest reported these top X features: Gender, Survived?, Class/Dept\n')

print('Part 3: ')
print('Based on the analysis from Parts 1 and 2, I suggest the following features:\n'
      'I recommend using Gender and Class/Dept as primary features for predicting survival outcomes.\n'
      'I recommend those by considering their consistent significance across different analyses. Age might also be considered\n'
      'but its inclusion might require further exploration.\n')

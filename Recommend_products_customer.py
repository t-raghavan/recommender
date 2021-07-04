# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:05:08 2021

@author: sasim
"""

import pandas as pd
import pickle

# Import Feature Engineered Sales Transaction file
sales_df = pd.read_csv('D:\\sasi\\Metric Bees\\Dataset\\Amazon\\amazon.csv')

#Build Correlation Matrix for the Customer-Product relations (using User-User based recommendation
# Find the total qty purchased by each customer of each product
prod_cust_qty_df = sales_df.groupby(['product_name','user_id']).agg({'Qty':'sum'})

# Reset the index by converting the Party and Product into columns
prod_cust_qty_df.reset_index(inplace=True)


# Find the no of unique customers purchased each product
prod_cust_count_df = sales_df.groupby(['product_name']).agg({'user_id':'nunique'})

# Set the customer count column
prod_cust_count_df.columns=['No_of_Customers']

# Reset the index by converting the Party and Product into columns
prod_cust_count_df.reset_index(inplace=True)


# Merge the unique customer count and qty purchased of each product
prod_cust_df = pd.merge(prod_cust_qty_df,prod_cust_count_df,how='inner',on='product_name')


# Create a pivot table with all Customers on columns and Products on rows, and Qty as values
prod_cust_pivot_df = prod_cust_df.pivot(index='product_name',columns='user_id',values='Qty').fillna(0)

# Find the correlation between every two customers and build a correlation matrix using corr() method
# Used Spearman method in identifying the correlation. Pearson was not providing better results and Kendall is taking a long time for execution.
cust_correlation_df = prod_cust_pivot_df.corr(method='spearman',min_periods=5)
#cust_correlation_df


cust_correlation_df.to_csv('D:\\sasi\\Metric Bees\\Dataset\\Amazon\\Customer-Customer-Correlation-Matrix.csv')


pickle.dump(cust_correlation_df, open('cust_correlation_model.pkl','wb'))

















































































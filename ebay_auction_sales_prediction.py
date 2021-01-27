# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import OneHotEncoder

import joblib
import pickle

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

class AuctionCount(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()
         
    def fit (self, X, y = None):
        out = X.copy()
        self.sales_count_ = out.groupby("phrase").size()
        self.sold_percent_ = y.groupby(out["phrase"]).mean()
        self.default_sales_count_ = 0
        self.default_sold_percent_ = y.mean()
        return self
    
    def transform(self, X):
        out = X.copy()
        out_df = pd.DataFrame()
        out_df["sales_count"] = out["phrase"].map(self.sales_count_)
        out_df["sold_percent"] = out["phrase"].map(self.sold_percent_)
        out_df["sales_count"] = out_df["sales_count"].fillna(self.default_sales_count_)
        out_df["sold_percent"] = out_df["sold_percent"].fillna(self.default_sold_percent_)
        return out_df
    
    def get_feature_names(self):
      return ["sales_count", "sales_percent"]

class IsProdLocSameAsSellerLoc(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()
         
    def fit (self, X, y= None):
      return self
    
    def transform(self, X):
       out = X.copy()
       return np.where(out["item_location"] == out["member_from"], 1, 0).reshape(len(out), 1)
    
    def get_feature_names(self):
      return ["is_same_country"]

class StartingBidComparedToItemPriceTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, value_extraction_method='median', 
                  null_impute_with_condition=['--not specified', 'For parts or not working']):
        super().__init__()
        self.value_extraction_method = value_extraction_method
        self.value_extraction_method_method_ = value_extraction_method
        if (value_extraction_method == 'median'):
            self.value_extraction_method_method_ = np.median
        elif (value_extraction_method == 'mean'):
            self.value_extraction_method_method_ = np.mean
        elif (value_extraction_method == 'min'):
            self.value_extraction_method_method_ = np.min
        elif (value_extraction_method == 'max'):
            self.value_extraction_method_method_ = np.max

        self.null_impute_with_condition = null_impute_with_condition
    
    def avrages_null_imputer(self, line):
      if not np.isnan( line['item_avg'] ):
        return line['item_avg']

      if line['item_condition'] in self.null_impute_with_condition:
          return float(self.winning_bids_by_condition[ line['item_condition'] ])

      return float(self.winning_bids_by_phrase[ line['phrase'] ])

    def fit(self, X, y=None):
        X_copy = X[~X['winning_bid_usd'].isna()]
        X_copy['phrase_and_condition'] = X_copy['phrase'] + '|' + str( X_copy['item_condition'] )
        self.winning_bids_avgs_ = (
            X_copy.groupby('phrase_and_condition')['winning_bid_usd'].apply(
              lambda s: self.value_extraction_method_method_(s))
        )

        self.winning_bids_by_condition = (
            X_copy.groupby('item_condition')['winning_bid_usd'].apply(
              lambda s: self.value_extraction_method_method_(s))
        )

        self.winning_bids_by_phrase = (
            X_copy.groupby('phrase')['winning_bid_usd'].apply(
              lambda s: self.value_extraction_method_method_(s))
        )
        return self
        

    def transform(self, X):
        X_copy = X.copy()
        X_copy['phrase_and_condition'] = X['phrase'] + '|' + str( X['item_condition'] )
        X_copy['item_avg'] = X_copy['phrase_and_condition'].map(self.winning_bids_avgs_).values
        X_copy['item_avg'] = X_copy.apply(self.avrages_null_imputer, axis=1)
        X_copy['avgs'] = (X_copy['item_avg'] - X_copy['starting_bid_usd'])
        return X_copy[['avgs']]

    def get_feature_names(self):
        return ['avg_price']

class ExtractWeekdayAndHour(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        out = X.copy()
        out['start_date_weekday'] = (out['date_started'].dt.weekday + 1).rename('start_date_weekday')
        out['end_date_weekday'] = (out['date_ended'].dt.weekday + 1).rename('end_date_weekday')
        out['end_date_hour'] = (out['date_ended'].dt.hour).rename('hour_ended')
        return out[["start_date_weekday", "end_date_weekday", "end_date_hour"]]
    
    def get_feature_names(self):
        return ["start_date_weekday", "end_date_weekday", "hour_ended"]

class SellerSeniority(TransformerMixin, BaseEstimator):
    def __init__(self):
      super().__init__()
    
    def fit(self, X, y=None):      
      return self
    
    def transform(self, X):
      out = X.copy()
      return (out['date_started'] - out['member_since']).dt.days.rename('seller_seniority').to_frame()
    
    def get_feature_names(self):
      return ["seller_seniority"]

class SimpleTargetEncoder(TransformerMixin, BaseEstimator):
  def __init__(self, cols=[]):
    super().__init__()
    self.cols = cols

  def fit(self, X, y=None):
    self.targets_ = {}
    for col in self.cols:
      self.targets_[col] = y.groupby(X[col]).mean()
    self.total_mean_ = y.mean()
    return self
    
  def transform(self, X):
    retVal = pd.DataFrame()
    for col in self.cols:
      retVal[col] = X[col].map(self.targets_[col])
      retVal[col] = retVal[col].fillna(self.total_mean_)
    return retVal
  
  def get_feature_names(self):
    return self.cols

class PartialOHE(TransformerMixin, BaseEstimator):
  def __init__(self, cols=[], sparse=False, drop=None, handle_unknown = 'ignore'):
    super().__init__()
    self.cols = cols
    self.sparse = sparse
    self.drop = drop
    self.handle_unknown = handle_unknown
  
  def fit(self, X, y=None):
    self.ohe = OneHotEncoder(sparse=self.sparse, drop=self.drop, handle_unknown= self.handle_unknown)
    self.ohe.fit(X[ self.cols ])
    return self

  def transform(self, X):
    return self.ohe.transform(X[self.cols])

  def get_feature_names(self):
    return ( ['{}_{}'.format(self.cols[i], name) 
      for i, cat_data in enumerate(self.ohe.categories_)
        for name in cat_data] )

class OrderByRelease(TransformerMixin, BaseEstimator):
  def __init__(self, cols=[] ):
    super().__init__()
  
  def fit(self, X, y=None):
    return self
    
  def transform(self, X):
    return X['Release date'].rank(method='dense').rename('relese ranking').to_frame()
  
  def get_feature_names(self):
    return ['relese ranking']

class IsContinuedOrSupported(TransformerMixin, BaseEstimator):
  def __init__(self, cols=[] ):
    super().__init__()
  
  def fit(self, X, y=None):
    return self
    
  def transform(self, X):
        out = X.copy()
        out['is_continued'] = np.where((out['date_started'] == np.nan) | (out['date_started'] > out.discontinued), 1, 0)
        out['is_supported'] = np.where((out['date_started'] == np.nan) | (out['date_started'] > out['support ended']), 1, 0)
        return out[["is_continued", "is_supported"]]
    
  def get_feature_names(self):
      return ["is_continued", "is_supported"]

class OrdinalCategory(TransformerMixin, BaseEstimator):
  def __init__(self, cols=[] ):
    super().__init__()
    self.cols = cols
  
  def fit(self, X, y=None):
    return self
    
  def transform(self, X):
    retVal = pd.DataFrame()
    for col in self.cols:
      labels, unique = pd.factorize(X[col], sort=True)
      retVal[col]  = labels
    return retVal
  
  def get_feature_names(self):
    return self.cols

class ExtractRest(TransformerMixin, BaseEstimator):
    def __init__(self, rest = ['duration', 'starting_bid_price_value', 'seller_rating', 'all_votes',
    'positive_feedback','Price at launce', 'IsReturnsAccepted', 'HasFreeShipping']):

        self.rest = rest
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.rest]
    
    def get_feature_names(self):
        return self.rest

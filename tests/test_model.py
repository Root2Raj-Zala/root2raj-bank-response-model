import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from model import FEATURES,EXCLUDED,make_pipeline,metrics,calibration_table

class ModelTests(unittest.TestCase):
    def test_duration_label_and_sensitive_fields_absent(self):
        self.assertFalse(set(FEATURES)&set(EXCLUDED+['y']))
    def test_unseen_category_is_supported(self):
        frame=pd.DataFrame({'contact':['cellular','telephone']*10,'month':['jan']*20,'day_of_week':['mon']*20,
                            'poutcome':['failure']*20,'campaign':range(20),'pdays':[999]*20,'previous':[0]*20})
        p=make_pipeline(LogisticRegression()).fit(frame,[0,1]*10)
        unseen=frame.iloc[:1].copy(); unseen['month']='unseen'
        values=p.predict_proba(unseen)[0]
        self.assertTrue(np.isfinite(values).all()); self.assertAlmostEqual(values.sum(),1.)
    def test_metrics_use_ranking_and_reconcile_calibration_counts(self):
        y=np.array([1,0,0,1,0,0,0,0,0,0]); p=np.array([.9,.8,.7,.6,.5,.4,.3,.2,.1,.0])
        self.assertEqual(metrics(y,p)['top_10pct_precision'],1.)
        self.assertEqual(sum(x['records'] for x in calibration_table(y,p)),10)
    def test_scaler_only_sees_training_rows(self):
        frame=pd.DataFrame({'contact':['cellular','telephone']*10,'month':['jan']*20,'day_of_week':['mon']*20,
                            'poutcome':['failure']*20,'campaign':range(20),'pdays':[999]*20,'previous':[0]*20})
        p=make_pipeline(LogisticRegression()).fit(frame,[0,1]*10)
        self.assertAlmostEqual(p.named_steps['preprocess'].named_transformers_['numeric'].mean_[0],9.5)

if __name__=='__main__': unittest.main()

"""Small functional checks used during local validation."""
import unittest
from database import init_db, seed_demo, frame, execute
from agents import decision_coordinator
from deep_learning.models import expense_series, train_lstm, train_autoencoder
from deep_learning.preprocessing import monthly_financial_dataset,lstm_dataset_preview,autoencoder_features,autoencoder_train_count,chronological_indices
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Smoke(unittest.TestCase):
    def test_seed_and_crud(self):
        init_db(); seed_demo()
        self.assertGreater(len(frame('SELECT * FROM transactions')), 100)
        row=execute("INSERT INTO transactions(date,kind,category,amount,description) VALUES('2026-11-06','Expense','Test',123,'CRUD')")
        execute('UPDATE transactions SET amount=? WHERE id=?',(124,row))
        self.assertEqual(float(frame('SELECT amount FROM transactions WHERE id=?',(row,)).iloc[0,0]),124)
        execute('DELETE FROM transactions WHERE id=?',(row,))
        self.assertTrue(frame('SELECT * FROM transactions WHERE id=?',(row,)).empty)
        self.assertGreater(len(frame('SELECT * FROM budgets')),0)
        budget=execute("INSERT INTO budgets(month,category,amount) VALUES('2026-11','CRUD test',50)")
        execute('UPDATE budgets SET amount=? WHERE id=?',(75,budget))
        self.assertEqual(float(frame('SELECT amount FROM budgets WHERE id=?',(budget,)).iloc[0,0]),75)
        execute('DELETE FROM budgets WHERE id=?',(budget,))
        inv=execute("INSERT INTO investments(asset,asset_type,quantity,purchase_price,current_price,purchase_date) VALUES('Test asset','Other assets',2,10,12,'2026-01-01')")
        execute('UPDATE investments SET current_price=? WHERE id=?',(15,inv))
        self.assertEqual(float(frame('SELECT current_price FROM investments WHERE id=?',(inv,)).iloc[0,0]),15)
        self.assertEqual(len(frame('SELECT * FROM investments WHERE id=?',(inv,))),1)
        execute('DELETE FROM investments WHERE id=?',(inv,))

    def test_agents_use_saved_rows(self):
        init_db(); seed_demo(); t=frame('SELECT * FROM transactions'); b=frame('SELECT * FROM budgets'); i=frame('SELECT * FROM investments'); p=frame('SELECT * FROM profile').iloc[0].to_dict()
        result=decision_coordinator(t,b,i,p,'2026-11')
        self.assertGreater(result['health']['score'],0)
        self.assertTrue(result['insights'])
        self.assertEqual(set(result['agents']),{'market','risk','portfolio','spending','savings','learning','explanation','notification'})
        self.assertTrue(all('supporting_data' in row and 'limitation' in row and 'recommended_action' in row for row in result['insights']))

    def test_dataset_preprocessing_and_exports(self):
        import io
        import pandas as pd
        init_db(); seed_demo(); t=frame('SELECT * FROM transactions')
        monthly=monthly_financial_dataset(t); preview,split=lstm_dataset_preview(monthly)
        self.assertEqual(len(monthly),12); self.assertEqual(len(preview),8)
        self.assertEqual(split['train']+split['validation']+split['test'],len(preview))
        self.assertLess(split['train_boundary'],split['validation_boundary']); self.assertLess(split['validation_boundary'],split['test_start'])
        self.assertEqual(len(pd.read_csv(io.StringIO(t.to_csv(index=False)))),len(t))
        self.assertEqual(len(pd.read_csv(io.StringIO(monthly.to_csv(index=False)))),len(monthly))
        ae,scaled,scaler,names=autoencoder_features(t)
        self.assertEqual(scaled.shape,(len(ae),4)); self.assertEqual(names,['amount_log1p','day_of_month','weekday','category_index'])
        self.assertEqual(autoencoder_train_count(len(ae))+len(ae)-autoencoder_train_count(len(ae)),len(ae))

    def test_real_neural_models_and_heldout_metrics(self):
        init_db(); seed_demo(); t=frame('SELECT * FROM transactions')
        values=expense_series(t); result=train_lstm(values,epochs=2)
        self.assertIsNotNone(result['model'])
        self.assertGreater(result['split']['test'],0)
        self.assertEqual(len(result['actual']),len(result['predicted']))
        self.assertAlmostEqual(result['mae'],mean_absolute_error(result['actual'],result['predicted']),places=5)
        self.assertAlmostEqual(result['rmse'],np.sqrt(mean_squared_error(result['actual'],result['predicted'])),places=5)
        self.assertTrue(np.isfinite(result['mape']))
        self.assertIn('mae',result['baseline']); self.assertGreater(result['parameters'],0)
        ae=train_autoencoder(t,epochs=2)
        self.assertGreater(len(ae['errors']),0)
        self.assertAlmostEqual(ae['threshold'],float(np.quantile(ae['threshold_training_errors'],.95)),places=6)
        self.assertEqual(int(ae['data'].anomaly.sum()),int((ae['errors']>ae['threshold']).sum()))
        self.assertEqual(ae['train_samples']+ae['validation_samples'],len(ae['errors']))
        import tempfile
        import deep_learning.model_store as store
        original=store.STORE
        try:
            with tempfile.TemporaryDirectory() as folder:
                store.STORE=__import__('pathlib').Path(folder); signature=store.dataset_fingerprint(t)
                store.save_result('lstm_expense',result,signature); restored=store.load_result('lstm_expense',signature)
                self.assertIsNotNone(restored); self.assertAlmostEqual(restored['mae'],result['mae'])
                store.save_result('autoencoder',ae,signature); restored_ae=store.load_result('autoencoder',signature)
                self.assertIsNotNone(restored_ae); self.assertEqual(restored_ae['threshold'],ae['threshold'])
                self.assertIsNone(store.load_result('autoencoder','different-data'))
        finally: store.STORE=original

if __name__=='__main__': unittest.main()
